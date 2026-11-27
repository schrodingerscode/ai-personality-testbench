"""Model adapters for different LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import time

from harness.config import ModelConfig


@dataclass
class Response:
    """Standardized response from any model."""

    text: str
    model_id: str
    provider: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    raw_response: Any = None


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str) -> Response:
        """Generate a response for the given prompt."""
        pass

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def provider(self) -> str:
        return self.config.provider


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI models (GPT-4, etc.)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from openai import OpenAI

        self.client = OpenAI()  # Uses OPENAI_API_KEY env var

    def generate(self, prompt: str) -> Response:
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self.config.extra_params,
        )
        latency = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        return Response(
            text=choice.message.content or "",
            model_id=self.config.model_id,
            provider="openai",
            latency_ms=latency,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            finish_reason=choice.finish_reason,
            raw_response=response,
        )


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic models (Claude)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from anthropic import Anthropic

        self.client = Anthropic()  # Uses ANTHROPIC_API_KEY env var

    def generate(self, prompt: str) -> Response:
        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.config.model_id,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            **self.config.extra_params,
        )
        latency = (time.perf_counter() - start) * 1000

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        return Response(
            text=text,
            model_id=self.config.model_id,
            provider="anthropic",
            latency_ms=latency,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason,
            raw_response=response,
        )


class XAIAdapter(ModelAdapter):
    """Adapter for xAI models (Grok). Uses OpenAI-compatible API."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        import os
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.environ.get("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )

    def generate(self, prompt: str) -> Response:
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self.config.extra_params,
        )
        latency = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        return Response(
            text=choice.message.content or "",
            model_id=self.config.model_id,
            provider="xai",
            latency_ms=latency,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            finish_reason=choice.finish_reason,
            raw_response=response,
        )


class GoogleAdapter(ModelAdapter):
    """Adapter for Google models (Gemini)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        import google.generativeai as genai

        genai.configure()  # Uses GOOGLE_API_KEY env var
        self.model = genai.GenerativeModel(
            model_name=self.config.model_id,
            system_instruction=self.config.system_prompt,
        )
        self.generation_config = genai.types.GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )

    def generate(self, prompt: str) -> Response:
        start = time.perf_counter()
        response = self.model.generate_content(
            prompt,
            generation_config=self.generation_config,
        )
        latency = (time.perf_counter() - start) * 1000

        return Response(
            text=response.text if response.text else "",
            model_id=self.config.model_id,
            provider="google",
            latency_ms=latency,
            prompt_tokens=None,  # Gemini doesn't always expose this
            completion_tokens=None,
            finish_reason=None,
            raw_response=response,
        )


# Registry of available adapters
ADAPTERS: dict[str, type[ModelAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "xai": XAIAdapter,
    "google": GoogleAdapter,
}


def get_adapter(config: ModelConfig) -> ModelAdapter:
    """Get the appropriate adapter for a model configuration."""
    adapter_class = ADAPTERS.get(config.provider)
    if adapter_class is None:
        raise ValueError(
            f"Unknown provider: {config.provider}. "
            f"Available: {list(ADAPTERS.keys())}"
        )
    return adapter_class(config)


def register_adapter(provider: str, adapter_class: type[ModelAdapter]) -> None:
    """Register a custom adapter for a provider."""
    ADAPTERS[provider] = adapter_class
