# AI Personality Testbench

This repository contains a testbench for exploring persistent behavioral patterns in large language models under controlled prompt conditions.

The goal is not to rank models or ascribe human-like personality traits, but to probe whether stable, model-specific tendencies emerge across repeated interactions, and how those tendencies vary with prompt structure, alignment layers, and context.

## What this is

- An experimental harness for structured probing of LLM behavior
- A set of defined behavioral axes and prompt probes
- Instrumentation for repeatability, variance analysis, and artifact generation
- A place to record observations, surprises, and open questions

## What this is not

- A benchmark or leaderboard
- A claim that models possess human personality
- A comprehensive evaluation framework
- A product or library intended for general consumption

## Approach

The testbench operates by:
- Defining behavioral axes as explicit, inspectable specifications
- Applying controlled prompt sets across models and conditions
- Capturing outputs, variance, and stability over repeated runs
- Separating raw artifacts from interpretation

Emphasis is placed on methodological clarity and epistemic limits rather than definitive conclusions.

## Repository structure

- `axes/` — Declarative definitions of behavioral dimensions
- `prompts/` — Base prompts, controls, and stress tests
- `harness/` — Execution, model adapters, and scoring utilities
- `experiments/` — Experiment configs, raw responses (`results.json`), and per-judge scores (`judge_scores_{model}.json`)
- `analysis/` — Re-analysis script and its output tables (`output_2a/`)
- `docs/` — Methodology, limitations, and the November 2026 re-analysis note

## January 2026 study and November 2026 re-analysis

The January 2026 runs cover 143 prompts, each answered three times by four models (1,716 responses), with every response scored by all four models acting as judges (6,767 usable judgments).

The first write-up was published in March 2026. A stricter re-analysis in November 2026 confirmed the overall model profiles and corrected several specific claims. See [`docs/methods_note_2a.md`](docs/methods_note_2a.md).

To reproduce every number in that note:

```bash
pip install -e ".[analysis]"
python analysis/reanalysis_2a.py
```

Tables are written to `analysis/output_2a/`, along with excerpts of the responses the note quotes.

## Limitations

Findings from this testbench are inherently constrained by:
- Prompt design and evaluator subjectivity
- Hidden changes in model providers and alignment layers
- Non-determinism and sampling variance
- The inability to directly observe internal representations

Results should be interpreted as exploratory signals, not definitive claims.

## Status

This is an evolving experimental workspace. Structure, axes, and methods may change as understanding improves.

## License

MIT License.
