"""Deterministic, side-effect-free fusion of keyword and vector candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from app.domain.retrieval import FusionMode, RetrievalCandidate, RetrievalMode

_Channel = Literal["keyword", "vector"]


@dataclass(frozen=True, slots=True)
class _ChannelCandidate:
    candidate: RetrievalCandidate
    rank: int
    score: float | None
    position: int


def _finite_score(score: float | None) -> float | None:
    if score is None or not math.isfinite(score):
        return None
    return score


def _channel_candidates(
    candidates: list[RetrievalCandidate],
    channel: _Channel,
) -> dict[UUID, _ChannelCandidate]:
    """Deduplicate one channel, retaining the best declared rank."""
    result: dict[UUID, _ChannelCandidate] = {}
    for position, candidate in enumerate(candidates, start=1):
        declared_rank = candidate.keyword_rank if channel == "keyword" else candidate.vector_rank
        rank = declared_rank or position
        score = _finite_score(
            candidate.keyword_score if channel == "keyword" else candidate.vector_score
        )
        record = _ChannelCandidate(candidate=candidate, rank=rank, score=score, position=position)
        existing = result.get(candidate.chunk_id)
        if existing is None:
            result[candidate.chunk_id] = record
            continue
        existing_score = existing.score if existing.score is not None else -math.inf
        record_score = score if score is not None else -math.inf
        if (rank, -record_score, position) < (
            existing.rank,
            -existing_score,
            existing.position,
        ):
            result[candidate.chunk_id] = record
    return result


def _merge_candidate(
    chunk_id: UUID,
    keyword: dict[UUID, _ChannelCandidate],
    vector: dict[UUID, _ChannelCandidate],
    fused_score: float,
) -> RetrievalCandidate:
    keyword_record = keyword.get(chunk_id)
    vector_record = vector.get(chunk_id)
    base = keyword_record or vector_record
    if base is None:  # pragma: no cover - guarded by callers' key union
        msg = f"missing candidate for chunk {chunk_id}"
        raise ValueError(msg)
    return base.candidate.model_copy(
        deep=True,
        update={
            "keyword_rank": keyword_record.rank if keyword_record else None,
            "keyword_score": keyword_record.score if keyword_record else None,
            "vector_rank": vector_record.rank if vector_record else None,
            "vector_score": vector_record.score if vector_record else None,
            "fused_rank": None,
            "fused_score": fused_score,
        },
    )


def _sort_key(candidate: RetrievalCandidate) -> tuple[float, float, float, float, str]:
    keyword_rank = float(candidate.keyword_rank) if candidate.keyword_rank is not None else math.inf
    vector_rank = float(candidate.vector_rank) if candidate.vector_rank is not None else math.inf
    return (
        -(candidate.fused_score or 0.0),
        min(keyword_rank, vector_rank),
        keyword_rank,
        vector_rank,
        str(candidate.chunk_id),
    )


def _finalize(
    candidates: list[RetrievalCandidate],
    top_k: int | None,
) -> list[RetrievalCandidate]:
    if top_k is not None and top_k < 1:
        msg = "top_k must be positive when provided"
        raise ValueError(msg)
    ordered = sorted(candidates, key=_sort_key)
    if top_k is not None:
        ordered = ordered[:top_k]
    return [
        candidate.model_copy(update={"fused_rank": rank})
        for rank, candidate in enumerate(ordered, start=1)
    ]


def reciprocal_rank_fusion(
    keyword_candidates: list[RetrievalCandidate],
    vector_candidates: list[RetrievalCandidate],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalCandidate]:
    """Fuse both channels using ``sum(1 / (k + rank))``."""
    if k < 1:
        msg = "RRF k must be positive"
        raise ValueError(msg)
    keyword = _channel_candidates(keyword_candidates, "keyword")
    vector = _channel_candidates(vector_candidates, "vector")
    fused: list[RetrievalCandidate] = []
    for chunk_id in keyword.keys() | vector.keys():
        score = 0.0
        if chunk_id in keyword:
            score += 1.0 / (k + keyword[chunk_id].rank)
        if chunk_id in vector:
            score += 1.0 / (k + vector[chunk_id].rank)
        fused.append(_merge_candidate(chunk_id, keyword, vector, score))
    return _finalize(fused, top_k)


def min_max_normalize(scores: list[float | None]) -> list[float]:
    """Normalize finite values to ``[0, 1]`` and safely handle ties.

    Missing and non-finite values receive zero. When all finite scores are
    equal, each finite value receives one because the channel still supplied a
    valid (albeit tied) signal.
    """
    finite = [score for score in scores if score is not None and math.isfinite(score)]
    if not finite:
        return [0.0] * len(scores)
    minimum = min(finite)
    maximum = max(finite)
    if math.isclose(maximum, minimum, rel_tol=0.0, abs_tol=1e-12):
        return [1.0 if score is not None and math.isfinite(score) else 0.0 for score in scores]
    scale = maximum - minimum
    return [
        (score - minimum) / scale if score is not None and math.isfinite(score) else 0.0
        for score in scores
    ]


def _normalized_scores(records: dict[UUID, _ChannelCandidate]) -> dict[UUID, float]:
    """Min-max normalize finite scores, with safe rank fallback."""
    finite = {
        chunk_id: record.score for chunk_id, record in records.items() if record.score is not None
    }
    if not finite:
        # Retrievers should normally provide scores. Reciprocal rank keeps the
        # weighted mode useful during graceful degradation when they do not.
        return {chunk_id: 1.0 / record.rank for chunk_id, record in records.items()}

    chunk_ids = list(records)
    normalized = min_max_normalize([records[chunk_id].score for chunk_id in chunk_ids])
    return dict(zip(chunk_ids, normalized, strict=True))


def weighted_score_fusion(
    keyword_candidates: list[RetrievalCandidate],
    vector_candidates: list[RetrievalCandidate],
    *,
    keyword_weight: float = 0.5,
    vector_weight: float = 0.5,
    top_k: int | None = None,
) -> list[RetrievalCandidate]:
    """Fuse per-channel min-max scores using normalized non-negative weights."""
    if not math.isfinite(keyword_weight) or not math.isfinite(vector_weight):
        msg = "fusion weights must be finite"
        raise ValueError(msg)
    if keyword_weight < 0 or vector_weight < 0:
        msg = "fusion weights cannot be negative"
        raise ValueError(msg)
    total_weight = keyword_weight + vector_weight
    if total_weight <= 0:
        msg = "at least one fusion weight must be positive"
        raise ValueError(msg)

    normalized_keyword_weight = keyword_weight / total_weight
    normalized_vector_weight = vector_weight / total_weight
    keyword = _channel_candidates(keyword_candidates, "keyword")
    vector = _channel_candidates(vector_candidates, "vector")
    keyword_scores = _normalized_scores(keyword)
    vector_scores = _normalized_scores(vector)
    fused = [
        _merge_candidate(
            chunk_id,
            keyword,
            vector,
            normalized_keyword_weight * keyword_scores.get(chunk_id, 0.0)
            + normalized_vector_weight * vector_scores.get(chunk_id, 0.0),
        )
        for chunk_id in keyword.keys() | vector.keys()
    ]
    return _finalize(fused, top_k)


def _single_channel(
    candidates: list[RetrievalCandidate],
    channel: _Channel,
    top_k: int | None,
) -> list[RetrievalCandidate]:
    records = _channel_candidates(candidates, channel)
    empty: dict[UUID, _ChannelCandidate] = {}
    ordered_records = sorted(
        records.items(),
        key=lambda item: (
            item[1].rank,
            -(item[1].score if item[1].score is not None else -math.inf),
            str(item[0]),
        ),
    )
    if top_k is not None:
        if top_k < 1:
            msg = "top_k must be positive when provided"
            raise ValueError(msg)
        ordered_records = ordered_records[:top_k]
    fused: list[RetrievalCandidate] = []
    for fused_rank, (chunk_id, record) in enumerate(ordered_records, start=1):
        # The original score remains available as both its channel score and
        # fused score, but the authoritative retriever rank determines order.
        score = record.score if record.score is not None else 1.0 / record.rank
        if channel == "keyword":
            candidate = _merge_candidate(chunk_id, records, empty, score)
        else:
            candidate = _merge_candidate(chunk_id, empty, records, score)
        fused.append(candidate.model_copy(update={"fused_rank": fused_rank}))
    return fused


def fuse_candidates(
    keyword_candidates: list[RetrievalCandidate],
    vector_candidates: list[RetrievalCandidate],
    *,
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
    fusion_mode: FusionMode = FusionMode.RRF,
    rrf_k: int = 60,
    keyword_weight: float = 0.5,
    vector_weight: float = 0.5,
    top_k: int | None = None,
) -> list[RetrievalCandidate]:
    """Dispatch keyword-only, vector-only, RRF, or weighted fusion."""
    if retrieval_mode == RetrievalMode.KEYWORD:
        return _single_channel(keyword_candidates, "keyword", top_k)
    if retrieval_mode == RetrievalMode.VECTOR:
        return _single_channel(vector_candidates, "vector", top_k)
    if retrieval_mode != RetrievalMode.HYBRID:
        msg = f"unsupported retrieval mode: {retrieval_mode}"
        raise ValueError(msg)
    if fusion_mode == FusionMode.RRF:
        return reciprocal_rank_fusion(keyword_candidates, vector_candidates, k=rrf_k, top_k=top_k)
    if fusion_mode == FusionMode.WEIGHTED:
        return weighted_score_fusion(
            keyword_candidates,
            vector_candidates,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            top_k=top_k,
        )
    msg = f"unsupported fusion mode: {fusion_mode}"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class HybridFusion:
    """Configurable facade suitable for dependency injection."""

    mode: FusionMode = FusionMode.RRF
    rrf_k: int = 60
    keyword_weight: float = 0.5
    vector_weight: float = 0.5

    def fuse(
        self,
        keyword_candidates: list[RetrievalCandidate],
        vector_candidates: list[RetrievalCandidate],
        *,
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int | None = None,
    ) -> list[RetrievalCandidate]:
        """Fuse candidates using this instance's configured strategy."""
        return fuse_candidates(
            keyword_candidates,
            vector_candidates,
            retrieval_mode=retrieval_mode,
            fusion_mode=self.mode,
            rrf_k=self.rrf_k,
            keyword_weight=self.keyword_weight,
            vector_weight=self.vector_weight,
            top_k=top_k,
        )
