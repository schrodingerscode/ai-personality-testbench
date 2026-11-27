"""Configuration handling for the testbench."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    provider: str  # openai, anthropic, google, etc.
    model_id: str  # gpt-4, claude-3-opus, gemini-pro, etc.
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str | None = None  # Minimal by default to measure base behavior
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Configuration for an experiment run."""

    name: str
    models: list[ModelConfig]
    samples_per_prompt: int = 5
    prompt_file: str = "prompts/battery.yaml"
    output_dir: str = "experiments"
    seed: int | None = None  # For reproducibility where supported
    prompt_ids: list[str] | None = None  # None = run all prompts


def load_config(path: str | Path) -> RunConfig:
    """Load run configuration from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    models = [ModelConfig(**m) for m in data.get("models", [])]

    return RunConfig(
        name=data["name"],
        models=models,
        samples_per_prompt=data.get("samples_per_prompt", 5),
        prompt_file=data.get("prompt_file", "prompts/battery.yaml"),
        output_dir=data.get("output_dir", "experiments"),
        seed=data.get("seed"),
        prompt_ids=data.get("prompt_ids"),
    )


def load_prompts(path: str | Path) -> list[dict]:
    """Load prompts from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("prompts", [])


# Example config for reference
EXAMPLE_CONFIG = """
name: "2026-01-initial-scan"
samples_per_prompt: 5
prompt_file: "prompts/battery.yaml"
output_dir: "experiments"

models:
  - provider: openai
    model_id: gpt-4-turbo
    temperature: 0.7
    max_tokens: 1024

  - provider: anthropic
    model_id: claude-3-opus-20240229
    temperature: 0.7
    max_tokens: 1024

  - provider: google
    model_id: gemini-pro
    temperature: 0.7
    max_tokens: 1024

# Optional: limit to specific prompts
# prompt_ids:
#   - def_01
#   - epi_01
"""
