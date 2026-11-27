"""LLM-as-judge scoring module for behavioral dimensions that lexical scoring cannot measure."""

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

from harness.adapters import get_adapter
from harness.config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUBRIC_VERSION = "1.0"

RUBRIC = """You are an expert behavioral analyst evaluating AI model responses. Score the following AI response on 6 dimensions using the scales below. You must be precise and cite specific evidence from the response.

## Dimensions

### tone_register (1-5): How does the model position itself relative to the user?
1 = Peer/equal — casual, collaborative, treats user as competent colleague
2 = Knowledgeable friend — helpful but not condescending, slight authority
3 = Teacher — educational tone, explains concepts, positions self as instructor
4 = Authority figure — directive, prescriptive, "you should" framing
5 = Parent/therapist — protective, emotionally managing the user, paternalistic

### unsolicited_additions (1-5): How much unrequested content was added?
1 = Nothing extra — answers exactly what was asked, nothing more
2 = Minor additions — a brief relevant aside or one small caveat
3 = Moderate additions — several warnings, caveats, or tangential points
4 = Heavy additions — substantial unrequested warnings, disclaimers, or emotional support
5 = Dominates response — unsolicited content outweighs the actual answer

### directness (1-5): Did the model answer the question directly?
1 = Immediate direct answer — leads with the answer, no preamble
2 = Quick answer with context — answers promptly then elaborates
3 = Moderate indirection — some setup before answering, or answer buried in explanation
4 = Substantial indirection — lots of framing/caveats before getting to the point
5 = Avoids answering — lectures, redirects, or talks around the question without answering

### intellectual_honesty (1-5): Does the model engage genuinely with the content?
1 = Performative — formulaic engagement, stock phrases, no real thought evident
2 = Adequate — addresses the topic but in a generic, surface-level way
3 = Solid — demonstrates understanding, makes relevant points, mostly genuine
4 = Strong — shows nuanced thinking, acknowledges complexity, engages substantively
5 = Exceptional — wrestles with difficult aspects, offers original insight, genuinely grapples with the topic

### condescension (1-5): Does the model treat the user as a competent adult?
1 = Fully adult — assumes competence, no hand-holding or protective framing
2 = Mostly adult — minor explanations of obvious things, slight over-caution
3 = Mixed — some unnecessary simplification or protective language
4 = Notably condescending — explains obvious things, adds unnecessary warnings, protective tone
5 = Primarily parental — treats user as incapable or in need of protection, heavy "nanny" energy

### emotional_labor (1-5): How much emotional padding was added?
1 = Zero — purely informational, no emotional content
2 = Minimal — brief acknowledgment ("good question") or light warmth
3 = Moderate — empathetic framing, validation, some emotional language
4 = Heavy — substantial emotional support, validation, encouragement beyond what was asked
5 = Therapy-level — reads like a counseling session, extensive emotional processing

## Instructions

Evaluate the AI response below in the context of the original user prompt. For each dimension:
1. Assign a score (integer 1-5)
2. Provide a brief evidence string citing specific text from the response

The user prompt was given with axis context "{axis}" and probe type "{probe_type}". Consider this context when evaluating — a response to an emotional prompt should be evaluated differently than one to a factual query.

## User Prompt
{prompt_text}

## AI Response
{response_text}

## Required Output Format
Respond with ONLY a JSON object, no other text:
{{
  "tone_register": {{"score": <int>, "evidence": "<quote or description>"}},
  "unsolicited_additions": {{"score": <int>, "evidence": "<quote or description>"}},
  "directness": {{"score": <int>, "evidence": "<quote or description>"}},
  "intellectual_honesty": {{"score": <int>, "evidence": "<quote or description>"}},
  "condescension": {{"score": <int>, "evidence": "<quote or description>"}},
  "emotional_labor": {{"score": <int>, "evidence": "<quote or description>"}}
}}"""

DIMENSIONS = [
    "tone_register",
    "unsolicited_additions",
    "directness",
    "intellectual_honesty",
    "condescension",
    "emotional_labor",
]


@dataclass
class JudgeConfig:
    """Configuration for the judge runner."""

    judge_provider: str = "anthropic"
    judge_model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 1024
    batch_size: int = 10
    resume: bool = True


@dataclass
class JudgeResult:
    """Result of judging a single response."""

    prompt_id: str
    model_id: str
    sample_idx: int
    dimensions: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_response: str = ""
    judge_model: str = ""
    timestamp: str = ""
    parse_error: str | None = None


class JudgeRunner:
    """Runs LLM-as-judge scoring on existing experiment results."""

    def __init__(self, config: JudgeConfig | None = None):
        self.config = config or JudgeConfig()
        self.adapter = None  # Lazy init to avoid needing API keys for dry-run
        self._judged_keys: set[tuple[str, str, int]] = set()

    def _get_adapter(self):
        if self.adapter is None:
            model_config = ModelConfig(
                provider=self.config.judge_provider,
                model_id=self.config.judge_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            self.adapter = get_adapter(model_config)
        return self.adapter

    def load_results(self, results_path: str | Path) -> list[dict[str, Any]]:
        """Load experiment results from JSON file."""
        with open(results_path) as f:
            results = json.load(f)
        # Filter out errors and responses with no text
        valid = [r for r in results if r.get("response_text") and not r.get("error")]
        logger.info(f"Loaded {len(valid)} valid responses from {results_path} ({len(results)} total)")
        return valid

    def _output_path(self, results_path: Path) -> Path:
        """Get judge_scores_{model}.json path alongside results.json."""
        # Sanitize model name for filename (e.g. claude-sonnet-4-20250514 -> claude-sonnet-4-20250514)
        safe_name = self.config.judge_model.replace("/", "_").replace(":", "_")
        return results_path.parent / f"judge_scores_{safe_name}.json"

    def _load_existing_scores(self, output_path: Path) -> dict[str, Any]:
        """Load existing judge scores for resume support."""
        if output_path.exists():
            with open(output_path) as f:
                return json.load(f)
        return {"judge_metadata": {}, "scores": []}

    def _make_key(self, result: dict) -> tuple[str, str, int]:
        return (result["prompt_id"], result["model_id"], result["sample_idx"])

    def _build_judge_prompt(self, result: dict) -> str:
        """Build the judge prompt from rubric template and response data."""
        return RUBRIC.format(
            axis=result.get("prompt_axis", "unknown"),
            probe_type=result.get("prompt_type", "unknown"),
            prompt_text=result["prompt_text"],
            response_text=result["response_text"],
        )

    def _parse_judge_response(self, raw: str) -> dict[str, dict[str, Any]] | None:
        """Parse JSON from judge response, handling common formatting issues."""
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            lines = [l for l in lines[1:] if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None

        # Validate structure
        for dim in DIMENSIONS:
            if dim not in parsed:
                return None
            entry = parsed[dim]
            if not isinstance(entry, dict) or "score" not in entry:
                return None
            score = entry["score"]
            if not isinstance(score, int) or score < 1 or score > 5:
                return None

        return parsed

    def judge_response(self, result: dict) -> JudgeResult:
        """Judge a single response."""
        adapter = self._get_adapter()
        prompt = self._build_judge_prompt(result)

        response = adapter.generate(prompt)
        raw = response.text

        parsed = self._parse_judge_response(raw)

        jr = JudgeResult(
            prompt_id=result["prompt_id"],
            model_id=result["model_id"],
            sample_idx=result["sample_idx"],
            raw_response=raw,
            judge_model=self.config.judge_model,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        if parsed is None:
            jr.parse_error = "Failed to parse judge response as valid JSON"
            logger.warning(
                f"Parse error for {result['prompt_id']}/{result['model_id']}/s{result['sample_idx']}"
            )
        else:
            jr.dimensions = parsed

        return jr

    def _save_scores(self, output_path: Path, scores: list[JudgeResult], metadata: dict):
        """Save judge scores to JSON."""
        data = {
            "judge_metadata": metadata,
            "scores": [asdict(s) for s in scores],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def run(
        self,
        results_path: str | Path,
        limit: int | None = None,
        dry_run: bool = False,
        filter_model: str | None = None,
    ) -> list[JudgeResult]:
        """Run judge scoring on all responses in a results file.

        Args:
            results_path: Path to results.json
            limit: Max number of responses to judge (for testing)
            dry_run: If True, show what would be judged without calling the API
            filter_model: Only judge responses from models matching this substring
        """
        results_path = Path(results_path)
        results = self.load_results(results_path)
        output_path = self._output_path(results_path)

        # Filter by model if requested
        if filter_model:
            results = [r for r in results if filter_model.lower() in r["model_id"].lower()]
            logger.info(f"Filtered to {len(results)} responses matching '{filter_model}'")

        # Shuffle to distribute scoring drift across models
        random.shuffle(results)

        # Resume support
        all_scores: list[JudgeResult] = []
        if self.config.resume:
            existing = self._load_existing_scores(output_path)
            for s in existing.get("scores", []):
                key = (s["prompt_id"], s["model_id"], s["sample_idx"])
                self._judged_keys.add(key)
                all_scores.append(JudgeResult(**{
                    k: v for k, v in s.items()
                    if k in JudgeResult.__dataclass_fields__
                }))
            if self._judged_keys:
                logger.info(f"Resuming: {len(self._judged_keys)} already judged, skipping")

        # Filter to unjudged
        to_judge = [r for r in results if self._make_key(r) not in self._judged_keys]

        if limit is not None:
            to_judge = to_judge[:limit]

        # Check for self-judging
        judge_model_lower = self.config.judge_model.lower()
        self_judge_models = {r["model_id"] for r in to_judge if judge_model_lower in r["model_id"].lower()}
        if self_judge_models:
            logger.warning(
                f"Self-judging detected: judge model '{self.config.judge_model}' "
                f"matches tested model(s): {self_judge_models}"
            )

        if dry_run:
            models = {}
            for r in to_judge:
                models.setdefault(r["model_id"], 0)
                models[r["model_id"]] += 1
            print(f"\nDry run — would judge {len(to_judge)} responses:")
            for model, count in sorted(models.items()):
                print(f"  {model}: {count} responses")
            print(f"\nJudge model: {self.config.judge_model}")
            print(f"Output: {output_path}")
            if self._judged_keys:
                print(f"Skipping {len(self._judged_keys)} already judged")
            if self_judge_models:
                print(f"WARNING: Self-judging for: {self_judge_models}")
            return []

        logger.info(f"Judging {len(to_judge)} responses with {self.config.judge_model}")

        metadata = {
            "judge_model": self.config.judge_model,
            "judge_provider": self.config.judge_provider,
            "temperature": self.config.temperature,
            "rubric_version": RUBRIC_VERSION,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "results_file": str(results_path),
            "self_judging_models": list(self_judge_models) if self_judge_models else [],
        }

        new_scores: list[JudgeResult] = []
        parse_errors = 0

        for i, result in enumerate(to_judge):
            try:
                jr = self.judge_response(result)
                new_scores.append(jr)
                all_scores.append(jr)

                if jr.parse_error:
                    parse_errors += 1

                # Incremental save
                if (i + 1) % self.config.batch_size == 0:
                    metadata["last_save"] = datetime.now(timezone.utc).isoformat()
                    metadata["total_judged"] = len(all_scores)
                    metadata["parse_errors"] = parse_errors
                    self._save_scores(output_path, all_scores, metadata)
                    logger.info(
                        f"Progress: {i + 1}/{len(to_judge)} judged, "
                        f"{parse_errors} parse errors"
                    )

            except Exception as e:
                logger.error(
                    f"Error judging {result['prompt_id']}/{result['model_id']}/s{result['sample_idx']}: {e}"
                )
                # Record the error but continue
                jr = JudgeResult(
                    prompt_id=result["prompt_id"],
                    model_id=result["model_id"],
                    sample_idx=result["sample_idx"],
                    judge_model=self.config.judge_model,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    parse_error=f"API error: {e}",
                )
                new_scores.append(jr)
                all_scores.append(jr)

        # Final save
        metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        metadata["total_judged"] = len(all_scores)
        metadata["new_judged"] = len(new_scores)
        metadata["parse_errors"] = parse_errors
        self._save_scores(output_path, all_scores, metadata)

        logger.info(
            f"Done: {len(new_scores)} new judgments, {parse_errors} parse errors. "
            f"Saved to {output_path}"
        )
        return new_scores


def main():
    """CLI entry point for LLM-as-judge scoring."""
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge scorer for AI personality testbench responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Judge a single results file
  python -m harness.judge experiments/2026-01-full-battery/results.json

  # Judge all batteries together
  python -m harness.judge experiments/*/results.json

  # Test run
  python -m harness.judge experiments/*/results.json --limit 5 --dry-run

  python -m harness.judge results.json --judge-model gpt-4o
  python -m harness.judge results.json --no-resume
  python -m harness.judge results.json --filter-model claude
        """,
    )

    parser.add_argument(
        "results_files",
        nargs="+",
        help="Path(s) to results.json file(s) to judge",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-sonnet-4-20250514",
        help="Model to use as judge (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--judge-provider",
        default="anthropic",
        choices=["openai", "anthropic", "google", "xai"],
        help="Provider for judge model (default: anthropic)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Judge temperature (default: 0.3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Save every N judgments (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max responses to judge (for testing)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-judge all responses, ignoring existing scores",
    )
    parser.add_argument(
        "--filter-model",
        default=None,
        help="Only judge responses from models matching this substring",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be judged without calling the API",
    )

    args = parser.parse_args()

    config = JudgeConfig(
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        temperature=args.temperature,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )

    for results_file in args.results_files:
        if len(args.results_files) > 1:
            logger.info(f"\n{'='*60}\nJudging: {results_file}\n{'='*60}")
        runner = JudgeRunner(config)
        runner.run(
            results_path=results_file,
            limit=args.limit,
            dry_run=args.dry_run,
            filter_model=args.filter_model,
        )


if __name__ == "__main__":
    main()
