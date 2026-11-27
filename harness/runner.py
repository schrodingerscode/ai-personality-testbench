"""Main runner for executing experiments."""

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv(override=True)

from harness.adapters import Response, get_adapter
from harness.config import ModelConfig, RunConfig, load_config, load_prompts
from harness.scoring import ResponseScorer, ScoreResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default batteries directory
PROMPTS_DIR = Path("prompts")
EXPERIMENTS_DIR = Path("experiments")


class Runner:
    """Orchestrates experiment runs across models and prompts."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.scorer = ResponseScorer()
        self.results: list[dict[str, Any]] = []

    def run(self) -> list[dict[str, Any]]:
        """Execute the experiment run."""
        prompts = load_prompts(self.config.prompt_file)

        # Filter prompts if specific IDs requested
        if self.config.prompt_ids:
            prompts = [p for p in prompts if p["id"] in self.config.prompt_ids]

        logger.info(
            f"Starting run '{self.config.name}' with {len(prompts)} prompts, "
            f"{len(self.config.models)} models, {self.config.samples_per_prompt} samples each"
        )

        total_calls = len(prompts) * len(self.config.models) * self.config.samples_per_prompt

        with tqdm(total=total_calls, desc="Running prompts") as pbar:
            for prompt_data in prompts:
                for model_config in self.config.models:
                    adapter = get_adapter(model_config)

                    for sample_idx in range(self.config.samples_per_prompt):
                        try:
                            response = adapter.generate(prompt_data["prompt"])
                            score = self.scorer.score(response.text)

                            result = self._build_result(
                                prompt_data=prompt_data,
                                model_config=model_config,
                                sample_idx=sample_idx,
                                response=response,
                                score=score,
                            )
                            self.results.append(result)

                        except Exception as e:
                            logger.error(
                                f"Error on {prompt_data['id']}/{model_config.model_id}: {e}"
                            )
                            self.results.append(
                                self._build_error_result(
                                    prompt_data, model_config, sample_idx, str(e)
                                )
                            )

                        pbar.update(1)

        return self.results

    def _build_result(
        self,
        prompt_data: dict,
        model_config: ModelConfig,
        sample_idx: int,
        response: Response,
        score: ScoreResult,
    ) -> dict[str, Any]:
        """Build a result record."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "run_name": self.config.name,
            "prompt_id": prompt_data["id"],
            "prompt_axis": prompt_data.get("axis"),
            "prompt_type": prompt_data.get("probe_type"),
            "prompt_text": prompt_data["prompt"],
            "model_provider": model_config.provider,
            "model_id": model_config.model_id,
            "temperature": model_config.temperature,
            "sample_idx": sample_idx,
            "response_text": response.text,
            "latency_ms": response.latency_ms,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "finish_reason": response.finish_reason,
            "scores": asdict(score),
            "error": None,
        }

    def _build_error_result(
        self,
        prompt_data: dict,
        model_config: ModelConfig,
        sample_idx: int,
        error: str,
    ) -> dict[str, Any]:
        """Build an error result record."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "run_name": self.config.name,
            "prompt_id": prompt_data["id"],
            "prompt_axis": prompt_data.get("axis"),
            "prompt_type": prompt_data.get("probe_type"),
            "prompt_text": prompt_data["prompt"],
            "model_provider": model_config.provider,
            "model_id": model_config.model_id,
            "temperature": model_config.temperature,
            "sample_idx": sample_idx,
            "response_text": None,
            "latency_ms": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish_reason": None,
            "scores": None,
            "error": error,
        }

    def save_results(self, output_dir: str | Path | None = None, filename: str | None = None) -> Path:
        """Save results to JSON file in experiment directory.

        Args:
            output_dir: Directory to save results. Defaults to config.output_dir/config.name
            filename: Custom filename. Defaults to 'results.json' or 'results_{run_name}.json'
                      if saving to shared directory
        """
        if output_dir is None:
            output_dir = Path(self.config.output_dir) / self.config.name
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = "results.json"

        output_file = output_dir / filename

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to {output_file}")
        return output_file


def run_experiment(config_path: str | Path, output_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Convenience function to run an experiment from a config file.

    Args:
        config_path: Path to the experiment config YAML file
        output_dir: Optional override for output directory

    Returns:
        List of result dictionaries
    """
    config = load_config(config_path)
    runner = Runner(config)
    results = runner.run()
    runner.save_results(output_dir=output_dir)
    return results


def list_batteries() -> list[Path]:
    """List all available battery configs in the experiments directory."""
    configs = list(EXPERIMENTS_DIR.glob("*_config.yaml"))
    return sorted(configs)


def list_prompt_files() -> list[Path]:
    """List all available prompt battery files."""
    prompts = list(PROMPTS_DIR.glob("*.yaml"))
    return sorted(prompts)


def run_all_batteries(exclude: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Run all available battery configs.

    Args:
        exclude: List of config names to exclude (without _config.yaml suffix)

    Returns:
        Dictionary mapping config names to their results
    """
    exclude = exclude or []
    configs = list_batteries()
    all_results = {}

    for config_path in configs:
        config_name = config_path.stem.replace("_config", "")
        if config_name in exclude or config_path.stem in exclude:
            logger.info(f"Skipping excluded config: {config_path.name}")
            continue

        logger.info(f"\n{'='*60}\nRunning battery: {config_path.name}\n{'='*60}")
        try:
            results = run_experiment(config_path)
            all_results[config_name] = results
        except Exception as e:
            logger.error(f"Failed to run {config_path.name}: {e}")
            all_results[config_name] = {"error": str(e)}

    return all_results


def main():
    """CLI entry point with flexible battery running options."""
    parser = argparse.ArgumentParser(
        description="AI Personality Testbench Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a specific config
  python -m harness.runner experiments/political_edgy_boundary_config.yaml

  # Run multiple configs
  python -m harness.runner config1.yaml config2.yaml

  # Run all available batteries
  python -m harness.runner --all

  # Run all except specific ones
  python -m harness.runner --all --exclude full_battery

  # List available batteries
  python -m harness.runner --list
        """,
    )

    parser.add_argument(
        "configs",
        nargs="*",
        help="Config file(s) to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available battery configs in experiments/",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Config names to exclude when using --all",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available batteries and prompt files",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("\nAvailable battery configs (experiments/*_config.yaml):")
        for config in list_batteries():
            print(f"  - {config.name}")

        print("\nAvailable prompt files (prompts/*.yaml):")
        for prompt_file in list_prompt_files():
            print(f"  - {prompt_file.name}")
        return

    # Run all mode
    if args.all:
        results = run_all_batteries(exclude=args.exclude)
        print(f"\nCompleted {len(results)} battery runs")
        for name, result in results.items():
            if isinstance(result, dict) and "error" in result:
                print(f"  - {name}: FAILED - {result['error']}")
            else:
                print(f"  - {name}: {len(result)} results")
        return

    # Run specific configs
    if not args.configs:
        parser.print_help()
        print("\nError: Provide config file(s) or use --all to run all batteries")
        return

    for config_path in args.configs:
        logger.info(f"\n{'='*60}\nRunning: {config_path}\n{'='*60}")
        run_experiment(config_path)


if __name__ == "__main__":
    main()
