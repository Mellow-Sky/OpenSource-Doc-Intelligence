"""Common contracts and failures for asynchronous document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.documents import RawDocument


class LoaderError(RuntimeError):
    """Raised when a source cannot be loaded safely or completely."""


class DocumentLoader(ABC):
    """Asynchronous contract implemented by every ingestion source loader."""

    @abstractmethod
    async def load(self) -> list[RawDocument]:
        """Load the current source snapshot as canonical raw documents."""
