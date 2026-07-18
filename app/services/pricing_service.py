"""Configuration-backed token pricing with explicit unknown-cost semantics."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.exceptions import ConfigurationError

_MILLION = Decimal(1_000_000)


class ModelPricing(BaseModel):
    """Per-million-token input and output prices in US dollars."""

    model_config = ConfigDict(extra="forbid")

    input_per_million_tokens: Decimal = Field(ge=0)
    output_per_million_tokens: Decimal = Field(ge=0)


class PricingFile(BaseModel):
    """Validated pricing file root."""

    model_config = ConfigDict(extra="forbid")

    pricing: dict[str, dict[str, ModelPricing]] = Field(default_factory=dict)


class PricingCatalog:
    """Look up exact provider/model prices without inventing fallbacks."""

    def __init__(self, pricing: dict[str, dict[str, ModelPricing]] | None = None) -> None:
        self._pricing = pricing or {}

    @classmethod
    def from_file(cls, path: Path) -> PricingCatalog:
        """Load a YAML catalog; a missing file is a configuration error."""
        try:
            raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
            parsed = PricingFile.model_validate(raw or {"pricing": {}})
        except (OSError, yaml.YAMLError, ValidationError) as exc:
            raise ConfigurationError(
                f"Invalid pricing configuration: {path}",
                details={"path": str(path)},
            ) from exc
        return cls(parsed.pricing)

    def estimate(
        self,
        *,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> Decimal | None:
        """Return exact configured cost, or None when the price is unknown."""
        if prompt_tokens < 0 or completion_tokens < 0:
            msg = "token counts must be non-negative"
            raise ValueError(msg)
        price = self._pricing.get(provider, {}).get(model)
        if price is None:
            return None
        input_cost = Decimal(prompt_tokens) * price.input_per_million_tokens / _MILLION
        output_cost = Decimal(completion_tokens) * price.output_per_million_tokens / _MILLION
        return input_cost + output_cost
