"""Unit tests for structure-preserving parsing, cleaning, chunking, and deduplication."""

from __future__ import annotations

from itertools import pairwise

from app.domain.documents import RawDocument
from app.ingestion.chunkers import ChunkingConfig, StructureAwareChunker
from app.ingestion.cleaners import CleaningConfig, DocumentCleaner
from app.ingestion.deduplication import (
    ContentDeduplicator,
    DeduplicationCandidate,
    DeduplicationMethod,
    normalize_content_for_hash,
    normalized_content_hash,
    simhash64,
)
from app.ingestion.parsers import (
    HTMLDocumentParser,
    MarkdownDocumentParser,
    RSTDocumentParser,
    StructuredTextParser,
)


def _raw(content: str, **metadata: str) -> RawDocument:
    return RawDocument(
        source_type="github_repo",
        external_id="docs/workloads/deployment.md",
        title="Kubernetes Workloads",
        content=content,
        canonical_url="https://kubernetes.io/docs/concepts/workloads/",
        source_version="commit-a",
        metadata={"document_type": "official_documentation", **metadata},
    )


def test_markdown_parser_preserves_structure_links_and_source_offsets() -> None:
    content = (
        "# Deployments\r\n\r\n"
        "See [rollout guidance](https://kubernetes.io/docs/rollout).\r\n\r\n"
        "```yaml\r\napiVersion: apps/v1\r\nkind: Deployment\r\n```\r\n\r\n"
        "| Field | Type |\r\n| --- | --- |\r\n| replicas | integer |\r\n"
    )

    parsed = MarkdownDocumentParser().parse(_raw(content))

    assert parsed.content.startswith("# Deployments\n")
    assert parsed.headings[0].text == "Deployments"
    assert (
        parsed.content[parsed.headings[0].start_offset : parsed.headings[0].end_offset]
        == "# Deployments"
    )
    assert parsed.code_blocks[0].language == "yaml"
    assert parsed.code_blocks[0].content.startswith("```yaml\napiVersion")
    assert parsed.tables[0].content == ("| Field | Type |\n| --- | --- |\n| replicas | integer |\n")
    assert parsed.links[0].target == "https://kubernetes.io/docs/rollout"
    assert parsed.links[0].start_offset == parsed.content.index("[rollout guidance]")
    assert parsed.source_map[-1].source_end == len(content)


def test_html_parser_removes_page_chrome_and_retains_semantic_blocks() -> None:
    content = """
    <html><body><nav>Global navigation</nav><main>
      <h1>Deployment</h1>
      <p>Read <a href="https://kubernetes.io/guide">the guide</a>.</p>
      <pre><code class="language-yaml">apiVersion: apps/v1
kind: Deployment
      </code></pre>
      <table><tr><th>Field</th><th>Meaning</th></tr>
      <tr><td>replicas</td><td>desired count</td></tr></table>
    </main><footer>Legal footer</footer></body></html>
    """
    parsed = HTMLDocumentParser().parse(_raw(content))

    assert "Global navigation" not in parsed.content
    assert "Legal footer" not in parsed.content
    assert parsed.headings[0].text == "Deployment"
    assert parsed.code_blocks[0].language == "yaml"
    assert "kind: Deployment" in parsed.code_blocks[0].content
    assert "| replicas | desired count |" in parsed.tables[0].content
    assert parsed.links[0].target == "https://kubernetes.io/guide"
    assert parsed.source_map[0].source_start == content.index("<h1>")


def test_rst_parser_retains_heading_code_and_link_structure() -> None:
    content = (
        "Deployment Guide\n================\n\n"
        "See `the API <https://kubernetes.io/docs/reference/>`_.\n\n"
        ".. code-block:: yaml\n\n"
        "   apiVersion: apps/v1\n"
        "   kind: Deployment\n"
    )
    raw = _raw(content).model_copy(update={"metadata": {"format": "rst"}})
    parser = RSTDocumentParser()

    parsed = parser.parse(raw)
    cleaned = DocumentCleaner(CleaningConfig(minimum_content_characters=1)).clean(
        parsed, parser=parser
    )

    assert cleaned.headings[0].text == "Deployment Guide"
    assert cleaned.code_blocks[0].language == "yaml"
    assert "   kind: Deployment" in cleaned.code_blocks[0].content
    assert cleaned.links[0].target == "https://kubernetes.io/docs/reference/"


def test_yaml_parser_keeps_top_level_nodes_and_indentation_atomic() -> None:
    content = (
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n  name: example\n"
        "data:\n  Next: '<!-- program data -->'\n"
    )
    raw = _raw(content).model_copy(update={"metadata": {"format": "yaml"}})
    parser = StructuredTextParser()

    parsed = parser.parse(raw)
    cleaned = DocumentCleaner(CleaningConfig(minimum_content_characters=1)).clean(
        parsed, parser=parser
    )
    chunks = StructureAwareChunker(
        ChunkingConfig(target_tokens=12, max_tokens=24, overlap_tokens=2, min_tokens=2)
    ).chunk(cleaned)

    assert [heading.text for heading in cleaned.headings] == [
        "apiVersion",
        "kind",
        "metadata",
        "data",
    ]
    assert "metadata:\n  name: example\n" in cleaned.content
    assert "Next: '<!-- program data -->'" in cleaned.content
    assert all(chunk.metadata["oversized_atomic"] or chunk.token_count <= 24 for chunk in chunks)


def test_cleaner_removes_toc_issue_template_and_noise_without_mutating_code_or_table() -> None:
    code = '```yaml\napiVersion: v1\n\n\ndata: {"key":  1}\n```\n'
    table = "| Key | Value |\n| --- | --- |\n| a   |  b    |\n"
    content = (
        "# Guide\n\n"
        "## Table of Contents\n- [Configuration](#configuration)\n- [Usage](#usage)\n\n"
        "<!-- Describe the issue here. -->\n_No response_\nNext\n\n"
        "## Configuration {#configuration}\n\n"
        f"{code}\n{table}\n"
        "Useful prose remains available to the retrieval system.\x00\n"
    )
    parsed = MarkdownDocumentParser().parse(_raw(content))

    cleaned = DocumentCleaner(CleaningConfig(minimum_content_characters=1)).clean(parsed)

    assert "Table of Contents" not in cleaned.content
    assert "Describe the issue" not in cleaned.content
    assert "No response" not in cleaned.content
    assert "\x00" not in cleaned.content
    assert "Next" not in cleaned.content
    assert "{#configuration}" not in cleaned.content
    assert code in cleaned.content
    assert table in cleaned.content
    assert cleaned.code_blocks[0].content == code
    assert cleaned.tables[0].content == table
    assert cleaned.metadata["quality_status"] == "ready"


def test_cleaner_marks_too_short_documents_instead_of_silently_discarding_them() -> None:
    parsed = MarkdownDocumentParser().parse(_raw("# Tiny\n\nShort.\n"))
    cleaned = DocumentCleaner(CleaningConfig(minimum_content_characters=100)).clean(parsed)

    assert cleaned.content
    assert cleaned.metadata["quality_status"] == "too_short"


def test_cleaner_preserves_indented_code_and_its_template_like_lines() -> None:
    indented_code = "    Next\n    <!-- this is program data -->\n    value:  1\n"
    parsed = MarkdownDocumentParser().parse(
        _raw(f"# Guide\n\nAn indented example follows.\n\n{indented_code}\nUseful prose.\n")
    )

    cleaned = DocumentCleaner(CleaningConfig(minimum_content_characters=1)).clean(parsed)

    assert indented_code in cleaned.content
    assert cleaned.code_blocks[0].content == indented_code


def test_cleaner_source_map_keeps_original_offsets_and_line_numbers() -> None:
    content = "Next\n\n_No response_\n\n# Original heading\n\nUseful evidence remains.\n"
    parsed = MarkdownDocumentParser().parse(_raw(content))

    cleaned = DocumentCleaner(CleaningConfig(minimum_content_characters=1)).clean(parsed)
    heading_offset = cleaned.content.index("# Original heading")
    mapping = next(
        entry
        for entry in cleaned.source_map
        if entry.normalized_start <= heading_offset < entry.normalized_end
    )

    assert mapping.source_start == content.index("# Original heading")
    assert mapping.source_start_line == 5

    chunks = StructureAwareChunker().chunk(cleaned)
    heading_chunk = next(chunk for chunk in chunks if "Original heading" in chunk.content)
    assert heading_chunk.position.start_offset == content.index("# Original heading")
    assert heading_chunk.position.start_line == 5


def test_chunker_uses_heading_hierarchy_overlap_and_atomic_code_and_tables() -> None:
    first = " ".join(f"workload{i}" for i in range(90))
    second = " ".join(f"deployment{i}" for i in range(90))
    code = "```yaml\napiVersion: apps/v1\nkind: Deployment\nspec:\n  replicas: 3\n```\n"
    table = "| Field | Type |\n| --- | --- |\n| replicas | integer |\n"
    content = f"# Workloads\n\n{first}\n\n## Deployment\n\n{second}\n\n{code}\n{table}"
    parsed = MarkdownDocumentParser().parse(_raw(content))
    config = ChunkingConfig(target_tokens=30, max_tokens=42, overlap_tokens=6, min_tokens=5)

    chunks = StructureAwareChunker(config).chunk(parsed)

    assert len(chunks) > 4
    assert all(
        chunk.token_count <= config.max_tokens or chunk.metadata["oversized_atomic"]
        for chunk in chunks
    )
    assert sum(code.rstrip("\n") in chunk.content for chunk in chunks) == 1
    assert sum(table.rstrip("\n") in chunk.content for chunk in chunks) == 1
    deployment_root = next(
        chunk
        for chunk in chunks
        if chunk.heading_path == ["Workloads", "Deployment"]
        and chunk.metadata["chunk_role"] == "section_root"
    )
    assert deployment_root.parent_index is not None
    assert chunks[deployment_root.parent_index].heading_path == ["Workloads"]
    continuations = [
        chunk
        for chunk in chunks
        if chunk.heading_path == ["Workloads", "Deployment"]
        and chunk.metadata["chunk_role"] == "continuation"
    ]
    assert continuations
    assert all(chunk.parent_index == deployment_root.chunk_index for chunk in continuations)
    assert any(0 < int(chunk.metadata["overlap_tokens"]) <= 6 for chunk in chunks[1:])
    assert all(chunk.position.end_offset > chunk.position.start_offset for chunk in chunks)


def test_chunk_overlap_repeats_the_configured_tail_tokens() -> None:
    words = " ".join(f"token{i}" for i in range(120))
    parsed = MarkdownDocumentParser().parse(_raw(words))
    chunks = StructureAwareChunker(
        ChunkingConfig(target_tokens=24, max_tokens=34, overlap_tokens=5, min_tokens=5)
    ).chunk(parsed)

    overlapping_pairs = 0
    for previous, current in pairwise(chunks):
        if int(current.metadata["overlap_tokens"]) == 0:
            continue
        previous_words = previous.content.split()
        current_words = current.content.split()
        assert set(previous_words[-5:]).intersection(current_words[:5])
        assert current.position.start_offset < previous.position.end_offset
        overlapping_pairs += 1
    assert overlapping_pairs >= 2


def test_normalized_sha256_ignores_prose_layout_but_not_code_changes() -> None:
    left = "Deployment   rollout\r\n\r\nUse kubectl.\r\n"
    right = " Deployment rollout\n\n\nUse kubectl. "

    assert normalize_content_for_hash(left) == normalize_content_for_hash(right)
    assert normalized_content_hash(left) == normalized_content_hash(right)
    assert normalized_content_hash("```yaml\na:  1\n```") != normalized_content_hash(
        "```yaml\na: 1\n```"
    )


def test_normalized_sha256_preserves_unfenced_yaml_indentation() -> None:
    nested = "spec:\n  template:\n    replicas: 3\n"
    flattened = "spec:\ntemplate:\n  replicas: 3\n"

    assert normalized_content_hash(nested) != normalized_content_hash(flattened)


def test_chunk_hash_includes_context_used_for_embedding() -> None:
    content = "Evidence about a Kubernetes controller and its desired state."
    parser = MarkdownDocumentParser()
    first = StructureAwareChunker().chunk(
        parser.parse(_raw(content).model_copy(update={"title": "First"}))
    )
    second = StructureAwareChunker().chunk(
        parser.parse(_raw(content).model_copy(update={"title": "Second"}))
    )

    assert first[0].content == second[0].content
    assert first[0].content_hash != second[0].content_hash


def test_simhash_detects_near_duplicates_and_records_the_reason() -> None:
    sentence = (
        "Kubernetes Deployment rollout creates ReplicaSets, replaces old Pods gradually, "
        "and supports kubectl rollout status and kubectl rollout undo. "
    )
    original = sentence * 8
    near_copy = original.replace("old Pods", "older Pods", 1)
    deduplicator = ContentDeduplicator(similarity_threshold=0.90)
    existing = DeduplicationCandidate(
        external_id="docs/deployment.md",
        content=original,
        source_type="github_repo",
        source_version="commit-a",
        document_type="official_documentation",
    )
    candidate = DeduplicationCandidate(
        external_id="docs/deployment-copy.md",
        content=near_copy,
        source_type="github_repo",
        source_version="commit-a",
        document_type="official_documentation",
    )

    decision = deduplicator.compare(candidate, existing)

    assert decision.is_duplicate
    assert decision.method is DeduplicationMethod.SIMHASH
    assert decision.similarity is not None and decision.similarity >= 0.90
    assert "threshold" in decision.reason


def test_deduplicator_accepts_a_persisted_fingerprint_without_the_document_body() -> None:
    content = "Kubernetes Deployment rollback evidence with a revision history."
    existing = DeduplicationCandidate(
        external_id="docs/deployment.md",
        content="",
        source_type="github_repo",
        source_version="commit-a",
        document_type="official_documentation",
    )
    candidate = DeduplicationCandidate(
        external_id="docs/deployment-copy.md",
        content=content,
        source_type="github_repo",
        source_version="commit-a",
        document_type="official_documentation",
    )
    deduplicator = ContentDeduplicator(similarity_threshold=0.90)

    deduplicator.add_fingerprint(
        existing,
        exact_hash=normalized_content_hash(content),
        simhash=simhash64(content),
    )
    decision = deduplicator.find_duplicate(candidate)

    assert decision.is_duplicate
    assert decision.method is DeduplicationMethod.EXACT_SHA256
    assert decision.matched_external_id == "docs/deployment.md"


def test_deduplication_protects_versions_api_kinds_and_changed_logical_documents() -> None:
    content = "The replicas field specifies the desired number of Pods."
    v1 = DeduplicationCandidate(
        external_id="api/deployment-v1",
        content=content,
        source_type="api_reference",
        source_version="v1.30",
        document_type="api_reference",
        metadata={"api_group": "apps", "api_version": "v1", "kind": "Deployment"},
    )
    v1beta = DeduplicationCandidate(
        external_id="api/deployment-v1beta1",
        content=content,
        source_type="api_reference",
        source_version="v1.30",
        document_type="api_reference",
        metadata={"api_group": "apps", "api_version": "v1beta1", "kind": "Deployment"},
    )
    changed = DeduplicationCandidate(
        external_id="api/deployment-v1",
        content=f"{content} Defaults depend on the controller.",
        source_type="api_reference",
        source_version="v1.31",
        document_type="api_reference",
        metadata={"api_group": "apps", "api_version": "v1", "kind": "Deployment"},
    )
    deduplicator = ContentDeduplicator(similarity_threshold=0.50)

    version_decision = deduplicator.compare(v1beta, v1)
    changed_decision = deduplicator.compare(changed, v1)

    assert not version_decision.is_duplicate
    assert version_decision.method is DeduplicationMethod.PROTECTED
    assert "api_version" in version_decision.reason
    assert not changed_decision.is_duplicate
    assert changed_decision.method is DeduplicationMethod.PROTECTED
    assert "new document version" in changed_decision.reason
