"""End-to-end chat application service with grounding and atomic audit persistence."""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.exceptions import ValidationError
from app.domain.chat import ChatLatency, ChatResult, ChatUsage
from app.domain.citations import BuiltContext, CitationReport
from app.domain.evaluation import ConversationTurn
from app.domain.retrieval import NoAnswerDecision, QueryFilters, RetrievalMode
from app.ingestion.chunkers import TokenCounter
from app.providers.base import EmbeddingProvider, LLMProvider, RerankerProvider, TokenUsage
from app.repositories.conversation_repository import (
    ConversationCreate,
    ConversationRepository,
    MessageCreate,
)
from app.repositories.retrieval_run_repository import (
    AnswerCitationCreate,
    RetrievalResultCreate,
    RetrievalRunCreate,
    RetrievalRunRepository,
)
from app.repositories.usage_repository import UsageRecordCreate, UsageRepository
from app.services.answer_service import AnswerGeneration, AnswerService
from app.services.citation_service import CitationService
from app.services.context_builder import ContextBuilder
from app.services.llm_judges import LLMCitationValidator, LLMEvidenceSufficiencyJudge
from app.services.no_answer_service import NoAnswerService
from app.services.pricing_service import PricingCatalog
from app.services.query_rewrite_service import QueryRewriteService, RewriteResult
from app.services.retrieval_service import build_retrieval_service


class ChatService:
    """Coordinate rewrite, retrieval, refusal, generation, citations, and persistence."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        embedding_provider: EmbeddingProvider | None,
        reranker_provider: RerankerProvider | None,
        llm_provider: LLMProvider | None,
        context_token_counter: TokenCounter | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._embedding_provider = embedding_provider
        self._reranker_provider = reranker_provider
        self._llm_provider = llm_provider
        self._pricing = PricingCatalog.from_file(settings.pricing_config_path)
        prompt_directory = settings.prompt_directory
        self._rewriter = QueryRewriteService(
            provider=llm_provider,
            prompt_path=prompt_directory / "query_rewrite.md",
            enabled=settings.enable_query_rewrite,
            history_turns=settings.query_rewrite_history_turns,
            max_tokens=settings.query_rewrite_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
            max_query_length=settings.max_query_length,
        )
        evidence_judge = (
            LLMEvidenceSufficiencyJudge(
                llm_provider,
                prompt_path=prompt_directory / "no_answer.md",
                max_tokens=settings.judge_max_tokens,
                timeout_seconds=settings.judge_timeout_seconds,
            )
            if llm_provider is not None
            else None
        )
        citation_validator = (
            LLMCitationValidator(
                llm_provider,
                prompt_path=prompt_directory / "citation_check.md",
                max_tokens=settings.judge_max_tokens,
                timeout_seconds=settings.judge_timeout_seconds,
            )
            if llm_provider is not None and settings.enable_citation_validation
            else None
        )
        self._no_answer = NoAnswerService.from_settings(settings, judge=evidence_judge)
        self._context_builder = ContextBuilder(
            max_context_tokens=settings.max_context_tokens,
            token_counter=context_token_counter,
        )
        self._answer = AnswerService(
            provider=llm_provider,
            prompt_path=prompt_directory / "answer.md",
            max_tokens=settings.answer_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._citations = CitationService(validator=citation_validator)

    async def complete(
        self,
        query: str,
        *,
        request_id: UUID | None = None,
        conversation_id: UUID | None = None,
        filters: QueryFilters | None = None,
        mode: RetrievalMode | None = None,
        top_k: int | None = None,
        history_override: Sequence[ConversationTurn] | None = None,
    ) -> ChatResult:
        """Execute one grounded turn and persist its complete trace atomically."""
        total_started = time.perf_counter()
        resolved_request_id = request_id or uuid4()
        resolved_conversation_id = conversation_id or uuid4()
        history = (
            list(history_override)
            if history_override is not None
            else await self._load_history(conversation_id)
        )
        rewrite = await self._rewriter.rewrite(query, history)

        async with (
            self._session_factory() as keyword_session,
            self._session_factory() as vector_session,
        ):
            retrieval_service = build_retrieval_service(
                keyword_session,
                self._settings,
                vector_session=vector_session,
                embedding_provider=self._embedding_provider,
                reranker_provider=self._reranker_provider,
            )
            retrieval = await retrieval_service.retrieve(
                query,
                rewritten_query=rewrite.rewritten_query,
                filters=filters,
                mode=mode,
                top_k=top_k,
            )

        decision = await self._no_answer.assess(rewrite.rewritten_query, retrieval)
        context = BuiltContext(text="", sources=[], token_count=0)
        generation: AnswerGeneration | None = None
        citation_report: CitationReport | None = None
        if decision.answerable:
            context = self._context_builder.build(retrieval.candidates)
            if not context.sources:
                decision = decision.model_copy(
                    update={
                        "answerable": False,
                        "reason": "context_budget_empty",
                        "confidence": 0.0,
                        "evidence_sufficiency_score": 0.0,
                    }
                )
        if decision.answerable:
            generation = await self._answer.generate(rewrite.rewritten_query, context)
            citation_report = await self._citations.analyze(generation.text, context)
            decision = self._no_answer.apply_citation_report(decision, citation_report)

        if decision.answerable and generation is not None and citation_report is not None:
            answer = generation.text
            citations = [citation for citation in citation_report.citations if citation.valid]
        else:
            answer = _refusal_text(retrieval.query.language)
            citations = []

        usage = _chat_usage(
            rewrite,
            generation,
            decision,
            citation_report,
            retrieval.embedding_prompt_tokens,
            retrieval.embedding_model,
            had_rerank=self._reranker_provider is not None and bool(retrieval.trace_candidates),
            embedding_provider=self._embedding_provider,
            reranker_provider=self._reranker_provider,
            llm_provider=self._llm_provider,
            pricing=self._pricing,
        )
        latency = ChatLatency(
            rewrite_ms=rewrite.latency_ms,
            keyword_retrieval_ms=retrieval.timings.keyword_ms,
            vector_retrieval_ms=retrieval.timings.vector_ms,
            fusion_ms=retrieval.timings.fusion_ms,
            rerank_ms=retrieval.timings.rerank_ms,
            generation_ms=generation.latency_ms if generation is not None else 0.0,
            citation_validation_ms=(
                citation_report.judge_latency_ms if citation_report is not None else 0.0
            ),
            total_ms=(time.perf_counter() - total_started) * 1000,
        )
        message_id = uuid4()
        result = ChatResult(
            request_id=resolved_request_id,
            conversation_id=resolved_conversation_id,
            message_id=message_id,
            original_query=query.strip(),
            rewritten_query=rewrite.rewritten_query,
            answer=answer,
            answerable=decision.answerable,
            confidence=decision.confidence,
            citations=citations,
            retrieval=retrieval,
            usage=usage,
            latency=latency,
            no_answer=decision,
            citation_report=citation_report,
        )
        await self._persist(
            result,
            new_conversation=conversation_id is None,
            rewrite=rewrite,
            generation=generation,
            context=context,
            history_to_seed=(
                history if conversation_id is None and history_override is not None else []
            ),
        )
        return result

    async def _load_history(
        self,
        conversation_id: UUID | None,
    ) -> list[ConversationTurn]:
        if conversation_id is None:
            return []
        async with self._session_factory() as session:
            repository = ConversationRepository(session)
            conversation = await repository.get(conversation_id)
            if conversation is None:
                raise ValidationError("Conversation does not exist")
            messages = await repository.list_recent_messages(
                conversation_id,
                limit=self._settings.query_rewrite_history_turns * 2,
            )
        return [ConversationTurn(role=item.role, content=item.content) for item in messages]

    async def _persist(
        self,
        result: ChatResult,
        *,
        new_conversation: bool,
        rewrite: RewriteResult,
        generation: AnswerGeneration | None,
        context: BuiltContext,
        history_to_seed: Sequence[ConversationTurn],
    ) -> None:
        now = datetime.now(UTC)
        selected_ids = {
            reference.chunk_id for source in context.sources for reference in source.chunks
        }
        trace_candidates = [
            item.model_copy(update={"selected_for_context": item.chunk_id in selected_ids})
            for item in result.retrieval.trace_candidates
        ]
        retrieval_run_id = uuid4()
        async with self._session_factory.begin() as session:
            conversations = ConversationRepository(session)
            if new_conversation:
                await conversations.create(
                    ConversationCreate(
                        id=result.conversation_id,
                        title=result.original_query[:512],
                    ),
                    created_at=now,
                )
            await conversations.add_messages(
                [
                    *[
                        MessageCreate(
                            conversation_id=result.conversation_id,
                            role=turn.role,
                            content=turn.content,
                        )
                        for turn in history_to_seed
                    ],
                    MessageCreate(
                        conversation_id=result.conversation_id,
                        role="user",
                        content=result.original_query,
                    ),
                    MessageCreate(
                        id=result.message_id,
                        conversation_id=result.conversation_id,
                        role="assistant",
                        original_query=result.original_query,
                        rewritten_query=result.rewritten_query,
                        content=result.answer,
                        token_usage=result.usage.model_dump(),
                        cost=(
                            Decimal(str(result.usage.estimated_cost_usd))
                            if result.usage.estimated_cost_usd is not None
                            else None
                        ),
                        latency_ms=round(result.latency.total_ms),
                    ),
                ],
                created_at=now,
            )
            runs = RetrievalRunRepository(session)
            await runs.create_run(
                RetrievalRunCreate(
                    id=retrieval_run_id,
                    message_id=result.message_id,
                    query=result.original_query,
                    rewritten_query=result.rewritten_query,
                    filters=result.retrieval.query.filters.model_dump(mode="json"),
                    keyword_latency_ms=result.retrieval.timings.keyword_ms,
                    vector_latency_ms=result.retrieval.timings.vector_ms,
                    rerank_latency_ms=result.retrieval.timings.rerank_ms,
                    total_latency_ms=result.retrieval.timings.total_ms,
                    retrieved_count=len(trace_candidates),
                    reranked_count=result.retrieval.reranked_count,
                    no_answer_score=result.no_answer.evidence_sufficiency_score,
                    metadata={
                        "answerable": result.answerable,
                        "confidence": result.confidence,
                        "reason": result.no_answer.reason,
                        "degraded_channels": result.retrieval.degraded_channels,
                        "reranker_degraded": result.retrieval.reranker_degraded,
                        "fusion_latency_ms": result.retrieval.timings.fusion_ms,
                        "citation_metrics": (
                            result.citation_report.model_dump(mode="json")
                            if result.citation_report is not None
                            else None
                        ),
                    },
                ),
                created_at=now,
            )
            await runs.add_results(
                [
                    RetrievalResultCreate.from_candidate(retrieval_run_id, candidate)
                    for candidate in trace_candidates
                ]
            )
            await runs.add_citations(_citation_records(result))
            await UsageRepository(session).add_many(
                _usage_records(
                    result,
                    rewrite=rewrite,
                    generation=generation,
                    embedding_provider=self._embedding_provider,
                    reranker_provider=self._reranker_provider,
                    llm_provider=self._llm_provider,
                    pricing=self._pricing,
                    created_at=now,
                )
            )


def _citation_records(result: ChatResult) -> list[AnswerCitationCreate]:
    if result.citation_report is None or not result.answerable:
        return []
    records: list[AnswerCitationCreate] = []
    seen: set[tuple[int, UUID]] = set()
    for validation in result.citation_report.validations:
        citation = validation.citation
        if citation is None or (citation.number, citation.chunk_id) in seen:
            continue
        seen.add((citation.number, citation.chunk_id))
        records.append(
            AnswerCitationCreate.from_citation(
                result.message_id,
                citation,
                claim_text=validation.claim.text,
            )
        )
    return records


def _usage_records(
    result: ChatResult,
    *,
    rewrite: RewriteResult,
    generation: AnswerGeneration | None,
    embedding_provider: EmbeddingProvider | None,
    reranker_provider: RerankerProvider | None,
    llm_provider: LLMProvider | None,
    pricing: PricingCatalog,
    created_at: datetime,
) -> list[UsageRecordCreate]:
    records: list[UsageRecordCreate] = []
    if rewrite.usage.total_tokens or rewrite.model is not None:
        records.append(
            _usage_record(
                result.request_id,
                "query_rewrite",
                rewrite.model or "unknown",
                llm_provider.name if llm_provider is not None else "unavailable",
                rewrite.usage,
                rewrite.latency_ms,
                created_at,
                pricing,
            )
        )
    if result.retrieval.embedding_model is not None:
        records.append(
            _usage_record(
                result.request_id,
                "query_embedding",
                result.retrieval.embedding_model,
                embedding_provider.name if embedding_provider is not None else "unavailable",
                TokenUsage(prompt_tokens=result.retrieval.embedding_prompt_tokens),
                result.retrieval.timings.vector_ms,
                created_at,
                pricing,
            )
        )
    if reranker_provider is not None and result.retrieval.trace_candidates:
        records.append(
            _usage_record(
                result.request_id,
                "rerank",
                reranker_provider.model,
                reranker_provider.name,
                TokenUsage(),
                result.retrieval.timings.rerank_ms,
                created_at,
                pricing,
            )
        )
    if generation is not None:
        records.append(
            _usage_record(
                result.request_id,
                "answer_generation",
                generation.model,
                llm_provider.name if llm_provider is not None else "unavailable",
                generation.usage,
                generation.latency_ms,
                created_at,
                pricing,
            )
        )
    judge_model = result.no_answer.diagnostics.get("judge_model")
    if isinstance(judge_model, str):
        judge_usage = TokenUsage(
            prompt_tokens=int(result.no_answer.diagnostics.get("judge_prompt_tokens", 0)),
            completion_tokens=int(result.no_answer.diagnostics.get("judge_completion_tokens", 0)),
        )
        records.append(
            _usage_record(
                result.request_id,
                "evidence_judge",
                judge_model,
                llm_provider.name if llm_provider is not None else "unavailable",
                judge_usage,
                float(result.no_answer.diagnostics.get("judge_latency_ms", 0.0)),
                created_at,
                pricing,
            )
        )
    citation_report = result.citation_report
    if citation_report is not None and citation_report.judge_model is not None:
        records.append(
            _usage_record(
                result.request_id,
                "citation_validation",
                citation_report.judge_model,
                llm_provider.name if llm_provider is not None else "unavailable",
                TokenUsage(
                    prompt_tokens=citation_report.judge_prompt_tokens,
                    completion_tokens=citation_report.judge_completion_tokens,
                ),
                citation_report.judge_latency_ms,
                created_at,
                pricing,
            )
        )
    return records


def _usage_record(
    request_id: UUID,
    operation: str,
    model: str,
    provider: str,
    usage: TokenUsage,
    latency_ms: float,
    created_at: datetime,
    pricing: PricingCatalog,
) -> UsageRecordCreate:
    return UsageRecordCreate(
        request_id=request_id,
        operation=operation,
        model=model,
        provider=provider,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        estimated_cost=pricing.estimate(
            provider=provider,
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        ),
        latency_ms=latency_ms,
        created_at=created_at,
    )


def _chat_usage(
    rewrite: RewriteResult,
    generation: AnswerGeneration | None,
    decision: NoAnswerDecision,
    citation_report: CitationReport | None,
    embedding_tokens: int,
    embedding_model: str | None,
    *,
    had_rerank: bool,
    embedding_provider: EmbeddingProvider | None,
    reranker_provider: RerankerProvider | None,
    llm_provider: LLMProvider | None,
    pricing: PricingCatalog,
) -> ChatUsage:
    prompt_tokens = rewrite.usage.prompt_tokens + embedding_tokens
    completion_tokens = rewrite.usage.completion_tokens
    if generation is not None:
        prompt_tokens += generation.usage.prompt_tokens
        completion_tokens += generation.usage.completion_tokens
    prompt_tokens += int(decision.diagnostics.get("judge_prompt_tokens", 0))
    completion_tokens += int(decision.diagnostics.get("judge_completion_tokens", 0))
    if citation_report is not None:
        prompt_tokens += citation_report.judge_prompt_tokens
        completion_tokens += citation_report.judge_completion_tokens
    costs: list[Decimal | None] = []
    if rewrite.usage.total_tokens or rewrite.model is not None:
        costs.append(
            pricing.estimate(
                provider=llm_provider.name if llm_provider is not None else "unavailable",
                model=rewrite.model or "unknown",
                prompt_tokens=rewrite.usage.prompt_tokens,
                completion_tokens=rewrite.usage.completion_tokens,
            )
        )
    if embedding_model is not None:
        costs.append(
            pricing.estimate(
                provider=(
                    embedding_provider.name if embedding_provider is not None else "unavailable"
                ),
                model=embedding_model,
                prompt_tokens=embedding_tokens,
                completion_tokens=0,
            )
        )
    if had_rerank and reranker_provider is not None:
        costs.append(
            pricing.estimate(
                provider=reranker_provider.name,
                model=reranker_provider.model,
                prompt_tokens=0,
                completion_tokens=0,
            )
        )
    if generation is not None:
        costs.append(
            pricing.estimate(
                provider=llm_provider.name if llm_provider is not None else "unavailable",
                model=generation.model,
                prompt_tokens=generation.usage.prompt_tokens,
                completion_tokens=generation.usage.completion_tokens,
            )
        )
    judge_model = decision.diagnostics.get("judge_model")
    if isinstance(judge_model, str):
        costs.append(
            pricing.estimate(
                provider=llm_provider.name if llm_provider is not None else "unavailable",
                model=judge_model,
                prompt_tokens=int(decision.diagnostics.get("judge_prompt_tokens", 0)),
                completion_tokens=int(decision.diagnostics.get("judge_completion_tokens", 0)),
            )
        )
    if citation_report is not None and citation_report.judge_model is not None:
        costs.append(
            pricing.estimate(
                provider=llm_provider.name if llm_provider is not None else "unavailable",
                model=citation_report.judge_model,
                prompt_tokens=citation_report.judge_prompt_tokens,
                completion_tokens=citation_report.judge_completion_tokens,
            )
        )
    estimated_cost = (
        float(sum((cost for cost in costs if cost is not None), start=Decimal(0)))
        if costs and all(cost is not None for cost in costs)
        else None
    )
    return ChatUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated_cost_usd=estimated_cost,
    )


def _refusal_text(language: str) -> str:
    if language == "en":
        return (
            "The current knowledge base does not contain enough verifiable evidence to answer "
            "this question. Keyword and vector retrieval were checked where available. Please "
            "narrow the question with a Kubernetes version, resource Kind, API group, release, "
            "or issue number."
        )
    return (
        "当前知识库中没有足够的可验证证据回答该问题。系统已检查可用的关键词与向量检索"
        "结果。请补充 Kubernetes 版本、资源 Kind、API Group、Release 版本或 Issue 编号, "
        "以便缩小检索范围。"
    )
