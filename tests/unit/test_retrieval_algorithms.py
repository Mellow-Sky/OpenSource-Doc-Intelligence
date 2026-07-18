"""Unit tests for query preprocessing, filter extraction, and hybrid fusion."""

from __future__ import annotations

import math
from uuid import UUID

import pytest

from app.domain.retrieval import FusionMode, QueryFilters, RetrievalCandidate, RetrievalMode
from app.retrieval.filters import extract_query_filters, merge_query_filters
from app.retrieval.hybrid_fusion import (
    HybridFusion,
    fuse_candidates,
    min_max_normalize,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from app.retrieval.query_preprocessor import QueryPreprocessor, detect_language, normalize_query


def _candidate(
    number: int,
    *,
    keyword_rank: int | None = None,
    vector_rank: int | None = None,
    keyword_score: float | None = None,
    vector_score: float | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=UUID(int=number),
        document_id=UUID(int=1000 + number),
        document_title=f"Document {number}",
        document_type="official_documentation",
        heading_path=["Workloads", "Deployments"],
        content=f"Chunk {number}",
        keyword_rank=keyword_rank,
        vector_rank=vector_rank,
        keyword_score=keyword_score,
        vector_score=vector_score,
    )


def test_query_preprocessor_normalizes_without_losing_identifiers() -> None:
    result = QueryPreprocessor().preprocess(
        "  \uff2b\uff55\uff42\uff45\uff52\uff4e\uff45\uff54\uff45\uff53\n v1.34  中的 "
        "apps/v1 Deployment `spec.template` 怎么配置\uff1f  "
    )

    assert result.normalized == "Kubernetes v1.34 中的 apps/v1 Deployment `spec.template` 怎么配置?"
    assert result.language == "mixed"
    assert result.filters.versions == ["1.34"]
    assert result.filters.api_groups == ["apps"]
    assert result.filters.kinds == ["Deployment"]
    assert result.filters.api_versions == ["v1"]
    assert {"Kubernetes", "v1.34", "apps/v1", "Deployment", "spec.template"}.issubset(
        set(result.protected_terms)
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Deployment rollout", "en"),
        ("如何回滚部署", "zh"),
        ("Deployment 如何回滚", "mixed"),
        ("1.34?", "unknown"),
    ],
)
def test_language_detection(query: str, expected: str) -> None:
    assert detect_language(query) == expected


def test_normalization_rejects_empty_and_oversized_queries() -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        normalize_query("\u0000  \n")
    with pytest.raises(ValueError, match="maximum length"):
        normalize_query("abcd", max_length=3)


def test_filter_extraction_covers_kubernetes_metadata() -> None:
    filters = extract_query_filters(
        "Kubernetes 1.30 closed GitHub Issue 中 release notes 1.31.2 对 "
        "apiGroup rbac.authorization.k8s.io 的 kind ClusterRole 有什么变化?"
    )

    assert filters.versions == ["1.30"]
    assert filters.document_types == ["github_issue", "release_note"]
    assert filters.api_groups == ["rbac.authorization.k8s.io"]
    assert filters.kinds == ["ClusterRole"]
    assert filters.issue_states == ["closed"]
    assert filters.release_versions == ["v1.31.2"]


def test_filter_extraction_accepts_explicit_state_and_version_qualifiers() -> None:
    filters = extract_query_filters("version: 1.28, release version 1.29.3, issue status: opened")

    assert filters.versions == ["1.28"]
    assert filters.release_versions == ["v1.29.3"]
    assert filters.issue_states == ["open"]
    assert filters.document_types == ["github_issue", "release_note"]


def test_supplied_filters_take_precedence_without_losing_extracted_metadata() -> None:
    extracted = extract_query_filters("apps/v1 Deployment API reference")
    supplied = QueryFilters(
        document_types=["official_documentation"],
        versions=["1.29"],
        api_versions=["v1beta1"],
        metadata={"tenant": "acme"},
    )

    merged = merge_query_filters(extracted, supplied)

    assert merged.document_types == ["official_documentation"]
    assert merged.versions == ["1.29"]
    assert merged.api_groups == ["apps"]
    assert merged.kinds == ["Deployment"]
    assert merged.api_versions == ["v1beta1"]
    assert merged.metadata == {"tenant": "acme"}


def test_rrf_retains_complete_provenance_and_does_not_mutate_inputs() -> None:
    keyword = [
        _candidate(1, keyword_rank=1, keyword_score=8.0),
        _candidate(2, keyword_rank=2, keyword_score=4.0),
    ]
    vector = [
        _candidate(2, vector_rank=1, vector_score=0.95),
        _candidate(3, vector_rank=2, vector_score=0.80),
    ]

    fused = reciprocal_rank_fusion(keyword, vector, k=60)

    assert [candidate.chunk_id for candidate in fused] == [UUID(int=2), UUID(int=1), UUID(int=3)]
    assert fused[0].keyword_rank == 2
    assert fused[0].vector_rank == 1
    assert fused[0].keyword_score == pytest.approx(4.0)
    assert fused[0].vector_score == pytest.approx(0.95)
    assert fused[0].fused_score == pytest.approx(1 / 62 + 1 / 61)
    assert [candidate.fused_rank for candidate in fused] == [1, 2, 3]
    assert all(candidate.fused_rank is None for candidate in keyword + vector)


def test_rrf_uses_declared_rank_and_best_duplicate() -> None:
    keyword = [
        _candidate(1, keyword_rank=5, keyword_score=0.5),
        _candidate(1, keyword_rank=2, keyword_score=0.4),
        _candidate(2, keyword_rank=3, keyword_score=0.8),
    ]

    fused = reciprocal_rank_fusion(keyword, [], k=60)

    assert fused[0].chunk_id == UUID(int=1)
    assert fused[0].keyword_rank == 2
    assert len(fused) == 2


def test_weighted_fusion_normalizes_scores_and_handles_constant_channel() -> None:
    keyword = [
        _candidate(1, keyword_rank=1, keyword_score=10.0),
        _candidate(2, keyword_rank=2, keyword_score=5.0),
    ]
    vector = [
        _candidate(2, vector_rank=1, vector_score=0.7),
        _candidate(3, vector_rank=2, vector_score=0.7),
    ]

    fused = weighted_score_fusion(keyword, vector, keyword_weight=3, vector_weight=1)

    by_id = {candidate.chunk_id: candidate for candidate in fused}
    assert by_id[UUID(int=1)].fused_score == pytest.approx(0.75)
    assert by_id[UUID(int=2)].fused_score == pytest.approx(0.25)
    assert by_id[UUID(int=3)].fused_score == pytest.approx(0.25)
    assert all(math.isfinite(candidate.fused_score or 0.0) for candidate in fused)
    # Ties are stable: best source rank, then keyword rank, vector rank, UUID.
    assert [candidate.chunk_id for candidate in fused] == [UUID(int=1), UUID(int=2), UUID(int=3)]


def test_min_max_normalize_handles_negative_missing_nonfinite_and_tied_scores() -> None:
    assert min_max_normalize([-2.0, 0.0, 2.0, None, math.nan]) == [0.0, 0.5, 1.0, 0.0, 0.0]
    assert min_max_normalize([7.0, 7.0, None]) == [1.0, 1.0, 0.0]
    assert min_max_normalize([None, math.inf]) == [0.0, 0.0]


def test_weighted_fusion_falls_back_to_rank_when_scores_are_missing() -> None:
    fused = weighted_score_fusion(
        [_candidate(1, keyword_rank=1), _candidate(2, keyword_rank=2)],
        [],
    )

    assert [candidate.chunk_id for candidate in fused] == [UUID(int=1), UUID(int=2)]
    assert fused[0].fused_score == pytest.approx(0.5)
    assert fused[1].fused_score == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (RetrievalMode.KEYWORD, [1, 2]),
        (RetrievalMode.VECTOR, [3, 2]),
    ],
)
def test_keyword_and_vector_only_modes(mode: RetrievalMode, expected: list[int]) -> None:
    keyword = [
        _candidate(1, keyword_rank=1, keyword_score=3.0),
        _candidate(2, keyword_rank=2, keyword_score=2.0),
    ]
    vector = [
        _candidate(3, vector_rank=1, vector_score=0.9),
        _candidate(2, vector_rank=2, vector_score=0.8),
    ]

    fused = fuse_candidates(keyword, vector, retrieval_mode=mode)

    assert [candidate.chunk_id.int for candidate in fused] == expected
    assert all(candidate.fused_rank is not None for candidate in fused)


def test_single_channel_mode_treats_declared_rank_as_authoritative() -> None:
    fused = fuse_candidates(
        [
            _candidate(1, keyword_rank=1, keyword_score=0.1),
            _candidate(2, keyword_rank=2, keyword_score=100.0),
        ],
        [],
        retrieval_mode=RetrievalMode.KEYWORD,
    )

    assert [candidate.chunk_id.int for candidate in fused] == [1, 2]
    assert [candidate.keyword_score for candidate in fused] == [0.1, 100.0]


def test_hybrid_facade_dispatches_weighted_fusion_and_top_k() -> None:
    fusion = HybridFusion(mode=FusionMode.WEIGHTED, keyword_weight=0.6, vector_weight=0.4)
    candidates = fusion.fuse(
        [_candidate(1, keyword_rank=1, keyword_score=1.0)],
        [_candidate(2, vector_rank=1, vector_score=1.0)],
        top_k=1,
    )

    assert len(candidates) == 1
    assert candidates[0].chunk_id == UUID(int=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"keyword_weight": -1.0},
        {"keyword_weight": 0.0, "vector_weight": 0.0},
        {"keyword_weight": math.inf},
    ],
)
def test_weighted_fusion_rejects_invalid_weights(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="weight"):
        weighted_score_fusion([], [], **kwargs)
