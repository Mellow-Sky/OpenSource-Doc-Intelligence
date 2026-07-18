"""The initial ORM metadata contains every required persistence entity."""

from app.db.base import Base
from app.db.models import *  # noqa: F403


def test_required_tables_are_registered() -> None:
    required = {
        "sources",
        "documents",
        "document_versions",
        "chunks",
        "conversations",
        "messages",
        "retrieval_runs",
        "retrieval_results",
        "answer_citations",
        "usage_records",
        "evaluation_cases",
        "evaluation_runs",
        "evaluation_results",
    }

    assert required <= set(Base.metadata.tables)


def test_document_identity_constraint_exists() -> None:
    constraints = {constraint.name for constraint in Base.metadata.tables["documents"].constraints}
    assert "uq_documents_source_external" in constraints


def test_evaluation_case_identity_constraint_exists() -> None:
    constraints = {
        constraint.name for constraint in Base.metadata.tables["evaluation_cases"].constraints
    }
    assert "uq_evaluation_cases_dataset_external_id" in constraints


def test_citation_identity_keeps_all_chunks_from_a_merged_source() -> None:
    table = Base.metadata.tables["answer_citations"]
    constraint = next(
        item
        for item in table.constraints
        if item.name == "uq_answer_citations_message_number_chunk"
    )

    assert {column.name for column in constraint.columns} == {
        "message_id",
        "citation_number",
        "chunk_id",
    }


def test_usage_records_keep_embedding_input_volume() -> None:
    table = Base.metadata.tables["usage_records"]

    assert {"input_text_count", "input_character_count"} <= set(table.columns.keys())
    constraint_names = {constraint.name for constraint in table.constraints}
    assert "ck_usage_records_input_text_count_nonnegative" in constraint_names
    assert "ck_usage_records_input_character_count_nonnegative" in constraint_names
