"""Shared database batching primitives for parameter-heavy repository writes."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

# A chunk insert currently binds fewer than 25 values per row. Keeping a batch at
# 500 rows stays comfortably below PostgreSQL's bind-parameter ceiling while still
# amortising network and statement-planning overhead.
DATABASE_WRITE_BATCH_SIZE = 500


def database_batches[T](
    values: Sequence[T],
    *,
    batch_size: int = DATABASE_WRITE_BATCH_SIZE,
) -> Iterator[Sequence[T]]:
    """Yield stable, bounded slices without copying the complete input sequence."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]
