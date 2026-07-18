"""Framework-independent usage facts emitted by model operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class EmbeddingBatchUsage:
    """Measured input volume and provider usage for one embedding batch."""

    model: str
    provider: str
    input_text_count: int
    input_character_count: int
    prompt_tokens: int
    latency_ms: float

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.provider.strip():
            raise ValueError("embedding usage model and provider must not be blank")
        counts = (
            self.input_text_count,
            self.input_character_count,
            self.prompt_tokens,
        )
        if any(value < 0 for value in counts):
            raise ValueError("embedding usage counts must be non-negative")
        if self.latency_ms < 0:
            raise ValueError("embedding usage latency must be non-negative")
