# OpenSource Doc Intelligence

OpenSource Doc Intelligence 是面向大型开源项目的企业级 RAG 文档助手。默认配置采集 Kubernetes 官方文档、仓库文档、GitHub Issues、Release Notes 和 API Reference；数据源、模型和检索策略均由配置替换，因此同一套架构也可用于 PyTorch、PostgreSQL 等项目。

原始版本、结构化 Chunk、全文索引、向量索引、检索轨迹、回答引用、会话、Token、成本、任务状态和离线评测结果都持久化到 PostgreSQL。外部模型通过 Provider 接口隔离

## 1. 功能列表

- Git 仓库、GitHub Issues、GitHub Releases、HTML/API Reference 多源采集；支持分页、限流重试、include/exclude glob 和增量游标。
- Markdown、RST、HTML、YAML、JSON 解析，保留标题层级、代码块、表格、链接及原始位置。
- 结构优先切分、父子 Chunk、Token 上限、Overlap、代码块保护、SHA-256 精确去重和 SimHash 近似去重。
- 幂等同步、文档版本历史、软删除、稳定 Chunk 对齐，以及只为变化 Chunk 重新生成 Embedding。
- PostgreSQL 加权 FTS + trigram、pgvector cosine HNSW、metadata filter、RRF/加权融合和 Cross-Encoder Rerank。
- 多轮问题改写、引用编号上下文、提示词注入隔离、组合式无答案检测、引用解析与校验。
- 普通聊天、引用溯源、会话查询、SSE、后台采集任务、评测任务、Prometheus 和使用量汇总 API。
- 52 条可复现 Kubernetes 示例评测集，覆盖 13 类问题，并提供生成器、确定性指标、独立 LLM Judge、配置对比和多格式报告。
- 结构化 JSON 日志、统一错误响应、request ID、可选 API Key、超时/退避/并发限制和优雅关闭。
- pytest、Ruff、mypy、pre-commit、GitHub Actions、Dockerfile 和 Docker Compose。

## 2. 总体架构

```mermaid
flowchart LR
    subgraph Sources[Knowledge sources]
        GH[Git repositories]
        Issues[GitHub Issues]
        Releases[Release Notes]
        APIRef[API Reference / HTML]
    end

    subgraph Pipeline[Ingestion pipeline]
        Load[Loaders]
        Parse[Parse and normalize]
        Clean[Clean and deduplicate]
        Chunk[Structure-aware chunking]
        Embed[Batch embedding]
    end

    subgraph Store[PostgreSQL 16]
        Raw[(documents and versions)]
        FTS[(weighted tsvector and GIN)]
        Vector[(pgvector and HNSW)]
        Audit[(conversation, retrieval, usage, evaluation)]
    end

    subgraph Online[Online RAG]
        Rewrite[Query rewrite]
        Hybrid[Keyword plus vector]
        Fusion[RRF / weighted fusion]
        Rerank[Reranker]
        Gate[No-answer gate]
        Context[Bounded context]
        LLM[LLM generation]
        Cite[Citation validation]
    end

    Sources --> Load --> Parse --> Clean --> Chunk --> Embed
    Parse --> Raw
    Chunk --> FTS
    Embed --> Vector
    Rewrite --> Hybrid
    FTS --> Hybrid
    Vector --> Hybrid
    Hybrid --> Fusion --> Rerank --> Gate --> Context --> LLM --> Cite
    Cite --> Audit
```

分层依赖方向如下：

```text
API Layer                 FastAPI routes, schemas, auth, error mapping
Application Layer         ingestion/chat/retrieval/evaluation/usage services
Domain Layer              provider-neutral Pydantic contracts and algorithms
Infrastructure Layer      SQLAlchemy repositories, PostgreSQL, GitHub and model adapters
Evaluation Layer          datasets, deterministic metrics, judge and report writers
```

Router 只负责传输和依赖组装；检索、拒答、引用及事务逻辑位于 Service/Domain 层。关键架构决策见 [`docs/architecture-decisions.md`](docs/architecture-decisions.md)。

## 3. 数据流

```mermaid
sequenceDiagram
    participant U as Client
    participant A as FastAPI
    participant Q as Query rewrite
    participant R as Hybrid retrieval
    participant N as No-answer detector
    participant G as LLM
    participant C as Citation validator
    participant P as PostgreSQL

    U->>A: query and conversation_id and filters
    A->>P: recent conversation turns
    A->>Q: original query and history
    Q-->>A: standalone query or safe fallback
    par keyword
        A->>R: parameterized FTS
    and vector
        A->>R: query embedding and cosine search
    end
    R-->>A: ranks and scores and timing
    A->>N: top evidence and channel signals
    alt evidence insufficient
        N-->>A: refusal reason and confidence
    else evidence sufficient
        A->>G: system prompt plus bounded SOURCE blocks
        G-->>A: answer with citation markers
        A->>C: claims and source mapping
        C-->>A: validated citations and coverage
    end
    A->>P: one transaction for messages, retrieval trace, citations and usage
    A-->>U: answer or grounded refusal
```

文档同步先比较内容哈希；未变化文档只更新时间，变化文档写入 `document_versions` 并重新解析。Chunk 使用内容哈希和稳定序列对齐，未变化向量继续复用；完整快照中消失的文档与 Chunk 被软删除，检索语句统一排除 `deleted_at IS NOT NULL` 的记录。

## 4. 技术选型

| 区域 | 实现 | 选择理由 |
| --- | --- | --- |
| API | Python 3.12、FastAPI、Pydantic v2 | 异步 I/O、类型化契约、OpenAPI |
| 持久化 | SQLAlchemy 2、Alembic、asyncpg | 显式事务、可迁移、异步连接池 |
| 搜索 | PostgreSQL FTS、GIN、pg_trgm、pgvector HNSW | 在同一事务域保存内容、过滤器与索引 |
| 采集 | httpx、受控 Git CLI、BeautifulSoup、trafilatura | 异步 API、可复用 Git 缓存、多格式解析；Git CLI 用于精确控制 shallow/fetch/超时和非交互行为 |
| 模型 | sentence-transformers、FlagEmbedding、OpenAI-compatible HTTP | 本地/远程 Provider 可替换 |
| 可观测 | structlog、Prometheus、持久化 usage records | request ID、低基数指标、成本审计 |
| 质量 | pytest、pytest-asyncio、Ruff、mypy、pre-commit | 无收费 API 的可重复验证 |

默认 BGE-M3 向量维度为 1024。初始迁移把 `chunks.embedding` 固定为 `vector(1024)`；换成不同维度时必须创建迁移并重建向量索引，不能只修改环境变量。

## 5. 仓库结构

```text
app/api                 HTTP routes, dependencies and errors
app/domain              documents, chunks, retrieval, citations and evaluation contracts
app/ingestion           loaders, parsers, cleaners, chunkers, dedup and incremental planner
app/providers           LLM, embedding and reranker ports/adapters
app/repositories        parameterized persistence and retrieval queries
app/services            application orchestration
evaluation              dataset, metrics, judge, runner and report writer
migrations              Alembic schema migrations
prompts                 versioned prompts; no business prompt is embedded in Python
scripts                 ingestion, dataset, evaluation, seed and benchmark CLIs
tests                   unit, integration and fixture coverage
```

## 6. 本地运行

前置条件：Python 3.12、[`uv`](https://docs.astral.sh/uv/)、PostgreSQL 16，并且数据库用户可创建 `vector` 和 `pg_trgm` 扩展。

```bash
cp .env.example .env
make install
```

若使用默认本地 Embedding 与 Reranker，还需安装模型可选依赖：

```bash
make install-models
```

`make install-models` 同时保留 telemetry extra；若只使用远程模型，启用 `ENABLE_TELEMETRY=true` 前执行 `make install-telemetry`。生产 Docker 镜像已包含但默认关闭该模块。

`.env.example` 的数据库主机 `postgres` 用于 Compose。在宿主机直接运行 API 时改为：

```env
DATABASE_URL=postgresql+asyncpg://rag:rag@127.0.0.1:5432/rag
```

配置可用的 LLM 以及与其匹配的 `LLM_TOKENIZER_MODEL`，再迁移、写入默认 source 配置并启动 API/worker：

```bash
make migrate
make seed
make dev
```

另开终端运行后台同步 worker：

```bash
uv run python -m app.worker
```

评测 API 的持久化任务由独立 worker 消费：

```bash
uv run python -m app.evaluation_worker
```

`GET /health` 只表示进程存活；`GET /ready` 会实际检查 PostgreSQL、pgvector、Embedding、Reranker 和 LLM。pgvector 检查还会读取 `chunks.embedding` 的物理 `vector(N)` typmod，并与 `DATABASE_VECTOR_DIMENSION` 比对；未迁移或维度不一致时不会就绪。只有 `/ready` 返回 200 才表示系统可以处理完整 RAG 请求。

远程 Provider 的默认探针遵循各自契约：OpenAI-compatible LLM/Embedding 读取模型目录并确认配置的 model ID，Azure 以 1 token/一条短文本的最小请求验证真实 deployment，远程 Reranker 使用单文档 `POST /rerank`，不假设 Reranker 服务提供 `/models`。可通过 `*_HEALTHCHECK_MODE` 覆盖；Reranker 也可选 `endpoint` 并配置同一 base URL 下的 `RERANKER_HEALTHCHECK_RESOURCE`。Azure 和 inference 探针可能产生极少量 token/推理费用，不要用高频 liveness 调用 `/ready`。

## 7. Docker 运行

复制 `.env.example` 并至少配置 LLM 与匹配的 `LLM_TOKENIZER_MODEL`。然后选择一种模型部署方式：

- 远程方式：设置 `EMBEDDING_PROVIDER=openai_compatible`、`EMBEDDING_BASE_URL`，以及 `RERANKER_PROVIDER=remote`、`RERANKER_BASE_URL`。
- 本地方式（Compose 默认）：保持 `INSTALL_LOCAL_MODELS=true`。这会把 PyTorch、sentence-transformers 和 FlagEmbedding 打入镜像，镜像更大，首次 readiness 还会下载模型权重。只有当 Embedding 与 Reranker 都改成远程 Provider 时才设置为 `false`。

```bash
cp .env.example .env
export GIT_COMMIT="$(git rev-parse --verify HEAD 2>/dev/null || true)"
docker compose config
docker compose up --build
```

Compose 包含：

- `postgres`：`pgvector/pgvector:pg16`；
- `migrate`：一次性执行 `alembic upgrade head`；
- `api`：以 UID 10001 的非 root 用户运行单 Uvicorn worker；
- `worker`：租约/心跳驱动的持久化采集任务 worker。
- `evaluation-worker`：领取评测任务，运行实际 RAG/Judge，并在同一事务中批量 upsert
  数据集样本、写入带样本外键的逐条结果、完成运行状态。

API、worker 等待数据库健康和迁移成功。Git 缓存、模型缓存、评测报告及数据库分别使用命名卷。默认单 API worker 避免本地大模型在同一容器内重复占用内存；远程模型部署可在编排层横向扩展 API。

Compose 网络中的数据库主机名是 `postgres`，数据采集和评测命令也应在应用容器中执行：

```bash
docker compose exec api python scripts/ingest_kubernetes.py --config config/sources.yaml
docker compose exec api python scripts/seed_demo_data.py \
  --evaluation-fixtures evaluation/datasets/kubernetes_source_catalog.jsonl
docker compose exec api python scripts/run_evaluation.py \
  --dataset evaluation/datasets/kubernetes_eval.jsonl \
  --output evaluation/reports/latest
```

`make docker-up` 会在当前仓库已有有效 commit 时，自动把 Git HEAD 作为 OCI revision 和 `GIT_COMMIT` 注入镜像。直接执行 `docker compose up --build` 时应像上例先导出该变量；新建但尚未 commit 的仓库会安全地留空。容器构建上下文不包含 `.git`，未注入时报告会诚实地把 revision 留空。

停止服务不会删除数据卷：

```bash
make docker-down
```

不要把“容器处于 running”视作模型已就绪；启动后仍应检查：

```bash
curl -fsS http://localhost:8000/health
curl -i http://localhost:8000/ready
```

## 8. 环境变量

完整模板在 [`.env.example`](.env.example)。空字符串由 Settings 忽略；密钥只从环境读取，不会写入配置快照或结构化日志。

| 变量组 | 关键变量 | 说明 |
| --- | --- | --- |
| 应用 | `APP_ENV`, `APP_HOST`, `APP_PORT`, `LOG_LEVEL`, `CORS_ORIGINS`, `GIT_COMMIT` | 环境、监听、日志、允许来源与评测 VCS provenance；生产环境关闭 Swagger |
| 鉴权 | `API_KEY`, `ADMIN_API_KEY` | 空值表示关闭；管理路由优先使用 admin key，否则回退 API key |
| 数据库 | `DATABASE_URL`, `DATABASE_POOL_SIZE`, `DATABASE_MAX_OVERFLOW`, `POSTGRES_PASSWORD`, `POSTGRES_PORT` | asyncpg URL、连接池和 Compose PostgreSQL 参数；使用内置数据库时两处密码必须一致，URL 中的保留字符需百分号编码 |
| GitHub | `GITHUB_TOKEN`, `GITHUB_REPOSITORY`, `GITHUB_BRANCH` | Token 可选但匿名限额较低；不会记录 Token |
| LLM | `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_TOKENIZER_MODEL`, `LLM_HEALTHCHECK_MODE` | OpenAI-compatible Chat Completions；Tokenizer 必须与服务模型匹配；含超时、重试、并发和 readiness 探针设置；URL 禁止内嵌 userinfo 凭据 |
| Judge | `JUDGE_PROVIDER`, `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL` | 与被评测 LLM 独立；空值表示不配置 Judge |
| Embedding | `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_DIMENSION`, `EMBEDDING_BATCH_SIZE`, `EMBEDDING_HEALTHCHECK_MODE` | `local` 或 OpenAI-compatible `/embeddings` |
| Reranker | `RERANKER_PROVIDER`, `RERANKER_MODEL`, `RERANKER_BASE_URL`, `RERANKER_API_KEY`, `RERANKER_BATCH_SIZE`, `RERANKER_HEALTHCHECK_*` | `local` 或通用 `/rerank`；可使用推理探针或专用 JSON 健康端点 |
| 检索 | `KEYWORD_TOP_K`, `VECTOR_TOP_K`, `RERANK_TOP_K`, `RRF_K`, `RETRIEVAL_MODE`, `FUSION_MODE` | 支持 keyword/vector/hybrid 和 RRF/weighted |
| Chunk | `CHUNK_TARGET_TOKENS`, `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `CHUNK_MIN_TOKENS` | 默认 500/800/80/80 |
| Tokenizer | `CHUNK_TOKENIZER_PROVIDER`, `CHUNK_TOKENIZER_MODEL`, `CHUNK_TOKENIZER_ALLOW_REGEX_FALLBACK` | 默认复用本地 Embedding tokenizer；远程模型需明确指定 tokenizer |
| 拒答 | `NO_ANSWER_*`, `EVIDENCE_SUFFICIENCY_THRESHOLD`, `CITATION_COVERAGE_THRESHOLD` | 分数阈值集中配置；无通用值的字段允许为空待校准 |
| 开关 | `ENABLE_QUERY_REWRITE`, `ENABLE_RERANKER`, `ENABLE_CITATION_VALIDATION`, `ENABLE_TELEMETRY` | 按实验或部署关闭可选阶段 |
| Trace | `OTEL_SERVICE_NAME`, `OTEL_EXPORTER_OTLP_ENDPOINT` | 可选 OpenTelemetry service name 与 OTLP gRPC endpoint |
| 运行边界 | `MAX_CONTEXT_TOKENS`, `MAX_QUERY_LENGTH`, `INGESTION_WORKER_POLL_SECONDS` | 防止无限输入、上下文和轮询 |
| 任务背压 | `INGESTION_MAX_OUTSTANDING_JOBS`, `EVALUATION_MAX_OUTSTANDING_RUNS`, `*_MAX_CONCURRENT_*`, `QUEUE_RETRY_AFTER_SECONDS` | PostgreSQL 原子门控待处理容量和跨 Worker 实例运行并发；超限返回 429 |
| 成本 | `PRICING_CONFIG_PATH` | 默认 `config/pricing.yaml`；未知价格返回 `null` |

## 9. 数据库迁移与索引

```bash
make migrate
make migration name=add_feature_name
```

初始迁移创建所有领域表、`vector`/`pg_trgm` 扩展及以下核心索引：

- `documents(source_id, external_id)` 唯一约束；
- `evaluation_cases(dataset_name, external_id)` 唯一约束；同一数据集重复评测会刷新样本快照，
  不会复制样本行；
- 文档/Chunk JSONB GIN；
- 标题 A、heading path B、正文 C、contextualized content D 的加权 `tsvector` 触发器和 GIN；
- 正文 `gin_trgm_ops`，补足连字符、点号和代码标识符的 FTS 匹配；
- `embedding vector_cosine_ops` HNSW，`m=16`、`ef_construction=64`；
- 检索运行、结果、引用、usage、评测、任务状态和同步游标索引。

索引由迁移创建，Chunk 写入时 `search_vector` 由触发器更新，Embedding 由批量 ingestion 写入；无需手工执行 SQL。生产迁移前应备份并在影子库运行 `alembic upgrade head`。

## 10. Kubernetes 数据采集

默认源定义在 [`config/sources.yaml`](config/sources.yaml)。先用 dry-run 验证网络、glob 和解析，不写数据库也不加载 Embedding：

```bash
uv run python scripts/ingest_kubernetes.py --config config/sources.yaml --dry-run
```

正式同步需要可用 Embedding Provider：

```bash
make ingest
```

其他选择：

```bash
uv run python scripts/ingest_kubernetes.py --all
uv run python scripts/ingest_kubernetes.py --source-id 00000000-0000-0000-0000-000000000000
uv run python scripts/ingest_kubernetes.py --config config/sources.yaml --no-delete
```

也可通过 API 入队，HTTP 请求不会执行长任务；`app.worker` 使用租约、心跳和 compare-and-set 状态恢复失败任务。重复 `idempotency_key` 即使在队列满时也返回已有任务。入队容量和运行并发由 PostgreSQL transaction advisory lock 原子控制，跨 API/Worker 副本生效；超限返回 `429 RATE_LIMITED` 与 `Retry-After`。同步统计包括 scanned、created、updated、unchanged、deleted、Chunk 变化、重复跳过和错误数。

正式同步只为 planner 判定为新增或变化的 Chunk 生成 Embedding。每个 Provider 批次都会在 `usage_records` 写入 `ingestion_embedding`，Worker 直接使用 ingestion job UUID 作为 `request_id`；CLI 输出独立的 `request_id`。审计记录包含输入文本条数、Unicode 字符数、Provider 报告的输入 Token、实测批次延迟和按价格表计算的成本。Provider 不返回 Token 时以 0 表示“未报告”，此时即使价格表存在也不会伪算为零成本，而是保留文本量并把成本记为 `null`。价格未知同样保持 `null`。重复同步没有 Embedding 工作时不会创建虚假 usage。

替换项目时复制 `config/sources.yaml`，修改仓库、分支、include/exclude、source type 和 document type；领域与检索代码无需修改。Git checkout 只允许落在配置的 cache root 下，避免路径穿越。Git clone/fetch/pull/log 以非交互方式运行，并受每个 source 的 `git_timeout_seconds` 约束，防止网络或凭据提示永久占用 worker。

## 11. API

开发环境的 OpenAPI 位于 `/docs`；生产环境默认关闭。受保护路由使用 `X-API-Key`，管理路由使用 `X-Admin-API-Key`。

| 方法 | 路径 | 鉴权 | 用途 |
| --- | --- | --- | --- |
| GET | `/health` | 无 | 进程存活 |
| GET | `/ready` | 无 | 数据库、pgvector 和模型 readiness |
| POST | `/api/v1/chat` | API | 完整、引用校验后的回答 |
| POST | `/api/v1/chat/stream` | API | SSE metadata/delta/done |
| GET | `/api/v1/conversations/{id}` | API | 会话及消息 |
| GET | `/api/v1/documents` | API | 文档分页与筛选 |
| GET | `/api/v1/documents/{id}` | API | 文档详情 |
| GET | `/api/v1/chunks/{id}` | API | Chunk、offset 和元数据 |
| GET | `/api/v1/citations/{chunk_id}` | API | Chunk 全文、相邻 Chunk、URL 和原始位置 |
| POST | `/api/v1/ingestion/jobs` | Admin | 幂等入队 |
| GET | `/api/v1/ingestion/jobs/{id}` | Admin | 任务状态/统计 |
| POST | `/api/v1/ingestion/sources/{id}/sync` | Admin | source-scoped 入队快捷方式 |
| POST | `/api/v1/evaluations` | Admin | 创建持久化评测运行 |
| GET | `/api/v1/evaluations/{id}` | Admin | 评测状态与摘要 |
| GET | `/api/v1/evaluations/{id}/report` | Admin | 受目录约束的报告下载 |
| GET | `/api/v1/usage/summary` | API | Token、成本和延迟聚合 |
| GET | `/metrics` | 无 | Prometheus exposition |

统一错误格式：

```json
{
  "error": {
    "code": "RETRIEVAL_FAILED",
    "message": "Human-readable message",
    "request_id": "2c8b79c2-9491-4b5c-947a-4d8db6bb5c68",
    "details": {}
  }
}
```

响应头 `X-Request-ID` 可用于关联 JSON 日志；普通路由只在长度安全时保留客户端值，聊天链路会进一步规范为 UUID，使响应体、usage 记录、日志和响应头使用同一标识。任务队列达到配置容量时返回 `429`，并附带整数秒 `Retry-After`，客户端应在该时间后使用相同 idempotency key 重试。

## 12. curl 示例

聊天与结构化过滤：

```bash
curl -sS http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "query": "Kubernetes Deployment 如何回滚到上一版本？",
    "filters": {
      "document_types": ["official_documentation"],
      "versions": ["1.34"],
      "kinds": ["Deployment"]
    },
    "mode": "hybrid",
    "top_k": 8,
    "debug": false
  }'
```

继续会话时把首次响应的 `conversation_id` 放入下一请求。查询引用原文：

```bash
curl -sS "http://localhost:8000/api/v1/citations/${CHUNK_ID}" \
  -H "X-API-Key: ${API_KEY}"
```

入队同步：

```bash
curl -sS -X POST "http://localhost:8000/api/v1/ingestion/sources/${SOURCE_ID}/sync" \
  -H 'Content-Type: application/json' \
  -H "X-Admin-API-Key: ${ADMIN_API_KEY}" \
  -d '{
    "idempotency_key": "k8s-nightly-2026-07-18",
    "requested_by": "operator",
    "options": {"dry_run": false, "allow_delete_missing": true}
  }'
```

## 13. 流式回答

```bash
curl -N http://localhost:8000/api/v1/chat/stream \
  -H 'Accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${API_KEY}" \
  -d '{"query":"它失败后怎么回滚？","conversation_id":"CONVERSATION_UUID"}'
```

事件顺序是 `metadata`、若干 `delta`、`done`。实现会先完成无答案和引用校验，再向客户端分段发送答案，因此不会在最终拒答前泄漏未经验证的模型 token；它是 citation-safe SSE，而不是直接透传上游模型 token。

## 14. 回答、引用与无答案策略

上下文按 rerank 顺序、Token 预算和去重规则选择；同一文档相邻 Chunk 可合并，但每个合并片段仍保留 Chunk/offset 映射。每段使用 `[SOURCE n]` 边界，包含标题、章节、URL、文档类型和内容。系统提示明确把文档视为不可信参考资料，文档中的角色切换、密钥请求或操作指令不会成为系统指令。

无答案检测组合使用：空通道、Top-1、Top-K 平均、Top-1 margin、主题词重叠、双通道信号、灰区 evidence judge 和生成后的引用覆盖率。阈值均在 Settings 中；未经数据校准的 provider-specific 分数阈值默认留空。证据不足时返回检索范围和可补充的版本、Kind、API group、Release 或 Issue 信息，不生成无来源答案。

引用编号只可指向本次提供的 SOURCE，引用 API 返回标题、章节、URL、具体 Chunk、前后邻居、行号及字符 offset。Issue 讨论的 `document_type` 会保留，回答层不得把个人观点表述为官方结论。

## 15. 自动评测数据集

仓库内 [`evaluation/datasets/kubernetes_eval.jsonl`](evaluation/datasets/kubernetes_eval.jsonl) 有 52 条样本：36 条可回答、16 条不可回答，easy/medium/hard 基本均衡。13 个分类包括事实、How-to、API 字段、版本差异、Release、Issue 状态、多跳、多轮指代、模糊、不可回答、范围外、幻觉陷阱和相似 API 区分。

所有自动生成样本保留 source chunk/hash，`human_reviewed` 明确表示审核状态，不会把模型生成内容伪装成人工真值。使用固定种子重建：

```bash
make eval-dataset

uv run python scripts/build_eval_dataset.py \
  --input evaluation/datasets/kubernetes_source_catalog.jsonl \
  --output evaluation/datasets/kubernetes_eval.jsonl \
  --count 52 \
  --seed 20250717
```

也可以通过 `--database-url` 从已索引的活动 Chunk 生成候选集。生成器去重问题、生成困难负样本和不可回答样本，并拒绝无法满足分类覆盖的输入。

## 16. 运行评测与对比实验

仓库内的 52 条 portable 数据集引用固定 source content hash。首次评测前必须把配套 source catalog 建成可检索索引；普通 `make seed` 只写 source 配置，不会写这些 fixture Chunk：

```bash
make seed-eval
# 等价于：
uv run python scripts/seed_demo_data.py \
  --evaluation-fixtures evaluation/datasets/kubernetes_source_catalog.jsonl
```

该命令使用当前 Embedding Provider 为 fixture 批量建立向量，因此 hybrid/vector 评测会与 JSONL 的 source hash 对齐。仅做 keyword-only 烟雾测试时可运行 `make seed-eval SEED_EVAL_ARGS=--no-embeddings`；这不能作为向量或混合检索评测。也可以改用真实同步后的知识库，但只有内容 hash 与数据集 provenance 匹配的证据才会计为相关命中。

配置 LLM、数据库和所需 Provider 后运行：

```bash
uv run python scripts/run_evaluation.py \
  --dataset evaluation/datasets/kubernetes_eval.jsonl \
  --output evaluation/reports/latest
```

使用独立 Judge 时配置 `JUDGE_*`；若只运行确定性指标可加 `--no-judge`。对比内置基线：

```bash
uv run python scripts/run_evaluation.py \
  --dataset evaluation/datasets/kubernetes_eval.jsonl \
  --output evaluation/reports/comparison \
  --compare \
  --concurrency 2
```

`--compare` 包含 keyword-only、vector-only、hybrid、hybrid-reranked、原始/改写问题和 Top-K 5/10/20。更多融合权重、Reranker 阈值或 Top-K 组合可通过 `--experiment-config experiments.yaml` 提供 Settings override。Chunk 大小会改变索引本身，因此此类实验必须先分别重建索引，并在各实验中指定对应的 `database_url`；仅修改报告中的 Chunk 参数不会被表述成有效实验。

每个运行保存：数据集指纹、UTC 时间、可获取时的 Git commit、脱敏配置快照、Prompt SHA-256、模型输出、检索证据、引用、各项指标、usage、错误和失败案例。报告输出 `report.json`、`results.jsonl`、`results.csv`、`report.md`；多实验还输出 comparison JSON/CSV/Markdown。

API 任务执行完成时，JSONL 中的完整样本字段也会批量写入 `evaluation_cases`，每条
`evaluation_results.evaluation_case_id` 指向对应的 `(dataset_name, external_id)` 稳定身份。
报告内仍保留 `_evaluation.case` 快照，便于离线审计；`0002` migration 会利用该快照回填
旧 worker 已写入但尚未关联的历史结果。

## 17. 指标定义

| 指标 | 定义 |
| --- | --- |
| Recall@1/3/5/10/20 | Top-K 是否包含任一 relevant Chunk |
| Relevant-set Recall@K | Top-K 覆盖 relevant Chunk 集合的比例 |
| MRR | 第一个 relevant Chunk 排名倒数的均值；未命中为 0 |
| Exact Match | Unicode、大小写和空白规范化后的严格相等 |
| Token F1 | 技术标识符感知的 multiset token precision/recall 调和平均 |
| Keyword Coverage | 要求关键词在生成答案中的覆盖比例 |
| Numeric/version consistency | 参考答案与生成答案数字/版本集合的一致性 F1 |
| Answer correct | Judge 的 factual correctness >= 4 且 groundedness >= 4；未配置 Judge 时为 `null` |
| Citation Precision | 有效引用数 / 全部引用数 |
| Citation Recall/Completeness | 需要外部证据的 Claim 中被正确引用覆盖的比例 |
| Citation Correctness | 规则有效性与 support score 的组合 |
| No-answer Accuracy/Precision/Recall/F1 | `answerable=true` 为正类的二分类指标 |
| FPR/FNR | 错答不可回答问题 / 错拒可回答问题的比例 |
| Rewrite delta | 改写前后 Recall@K 差，以及主题切换错误/不必要改写率 |
| Latency | mean、P50/P90/P95/P99、min/max、吞吐量和错误率；另按阶段汇总 |
| Token/cost | prompt、completion、judge、总量、单请求与全数据集成本；未知价格为 `null` |

Judge 必须同时接收 Question、Reference Answer、Generated Answer 和 Retrieved Evidence，并输出严格 JSON。Judge 与被测模型单独配置，且 Judge 不是唯一正确性指标。

无答案阈值应由真实报告校准，而不是凭经验填写：

```bash
uv run python scripts/calibrate_no_answer.py evaluation/reports/latest/report.json --top-k 3
```

脚本给出 `NO_ANSWER_TOP1_THRESHOLD` 和 `NO_ANSWER_AVG_THRESHOLD` 建议及对应二分类指标；上线前仍必须在独立、人工审核的 hold-out 集上复核。

## 18. 示例评测报告

[`docs/evaluation-report-example.md`](docs/evaluation-report-example.md) 展示实际报告生成器的字段与解读方式，所有尚未执行的数值均标为 `null`，不冒充真实评测结果。要得到可比较分数，必须对同一数据集指纹实际运行上一节命令；仓库不会提交虚构模型输出、延迟或成本。

## 19. 指标、成本与基准测试

Prometheus `/metrics` 包含按规范化 route template 标记的请求数、耗时直方图和响应体大小，避免把 UUID 放入 label。`/api/v1/usage/summary` 可按 request ID、operation、model、provider 和 UTC 时间范围过滤，并返回 operation/model/provider 分组、Token、`input_text_count`、`input_character_count`、请求延迟和检索阶段延迟。离线向量补建使用 `indexing_embedding`，同步使用 `ingestion_embedding`，固定评测 fixture 建索引使用 `seed_embedding`；每个逻辑批次形成一条 usage，但由 Repository 以最多 500 行的 SQL 批次写入，不逐 Chunk 提交。

`config/pricing.yaml` 初始为 `pricing: {}`。只有填入经供应商确认的 `provider -> model -> input_per_million_tokens/output_per_million_tokens` 后才估算成本；任一阶段价格未知时请求/汇总成本保持 `null`，并通过 `cost_complete=false` 暴露。

对运行中的 HTTP API 做有界并发基准：

```bash
make benchmark BENCHMARK_ARGS="--requests 100 --concurrency 8 --output evaluation/reports/benchmark.json"

uv run python scripts/benchmark.py \
  --input evaluation/datasets/kubernetes_eval.jsonl \
  --requests 100 \
  --concurrency 8 \
  --mode hybrid \
  --top-k 8
```

脚本从 `API_KEY` 环境变量读取凭据且不输出，统计成功延迟、P50/P90/P95/P99、吞吐量、状态码和错误率。它只调用目标 HTTP 服务，不自行调用收费模型；单元测试使用 HTTPX MockTransport。

## 20. 测试与代码质量

```bash
make test
make test-unit
make lint
make format-check
make typecheck
make precommit
```

集成测试需要独立 PostgreSQL 16 + pgvector，防止误用开发数据库：

```bash
export TEST_DATABASE_URL=postgresql+asyncpg://rag:rag@127.0.0.1:5432/rag_test
make test-integration
```

外部 GitHub/模型请求均可 Mock；测试不读取真实 API Key。CI 在 Python 3.12 上执行锁定依赖安装、Ruff、format check、mypy、Alembic、完整 pytest，并静态验证 Compose 与构建不含本地模型依赖的生产镜像。

## 21. 常见错误排查

### `/health` 是 200，但 `/ready` 是 503

查看 `checks` 中的失败项。`postgresql` 通常是 URL/网络问题；`pgvector` 表示尚未迁移、扩展权限不足，或 `chunks.embedding` 实际维度与 `DATABASE_VECTOR_DIMENSION` 不一致；模型项表示 base URL、model/deployment、认证或本地依赖/权重不可用。先执行 `make migrate`，再按 Provider 契约检查模型目录、`/embeddings`、Azure deployment 或 `/rerank`。

### 宿主机无法解析 `postgres`

`postgres` 是 Compose service DNS。宿主机运行时把 `DATABASE_URL` 主机改为 `127.0.0.1`；容器内保持 `postgres`。

### 本地模型报 `Unable to load`

执行 `make install-models`。确认可写模型缓存、磁盘/内存充足，且 `EMBEDDING_DIMENSION` 与模型真实维度一致。Docker 需 `INSTALL_LOCAL_MODELS=true docker compose build`；不想构建大镜像则改用远程 Provider。

### 迁移创建扩展失败

数据库角色需要 `CREATE EXTENSION vector` 和 `CREATE EXTENSION pg_trgm` 权限。托管 PostgreSQL 必须先确认支持 pgvector；不要删除 migration 中的扩展检查来绕过。

### GitHub 403/429 或同步很慢

配置 `GITHUB_TOKEN`，检查 GitHub rate-limit 响应和 worker 结构化日志。Loader 尊重 `Retry-After`/rate-limit reset 并执行有界指数退避；缩小 include glob、关闭 comments，或按 source 单独同步。

### 同步任务一直 pending

确认 `app.worker` 正在运行且连接同一数据库。检查任务 `heartbeat_at`；过期租约可由另一个 worker 安全重领。相同 idempotency key 不会创建第二条任务。

### 搜索不到连字符或代码标识符

FTS 使用 English/Simple 配置，同时以 `pg_trgm` 补足 `apps/v1`、`kubectl.kubernetes.io/*`、连字符和点号。确认 `search_vector` 触发器存在、Chunk 未软删除，并检查 metadata/version filter 是否过窄。

### 成本显示 `null`

这是预期的未知价格语义。为配置的精确 provider/model 填写 `config/pricing.yaml`，不要使用近似模型名或猜测价格。

### SSE 看起来不是逐 token

这是安全设计：回答先完成引用/拒答校验，再按句段发送。客户端仍应处理 `metadata`、`delta`、`done` 三类事件和断线重试。

更多生产检查、备份和故障处理见 [`docs/operations.md`](docs/operations.md)。

## 22. 模型替换

### 本地模型

```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
RERANKER_PROVIDER=local
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

本地推理通过 `asyncio.to_thread` 移出事件循环，并有批量大小限制。CPU 可运行但延迟较高；GPU/独立模型服务应按部署资源选择。

### OpenAI-compatible、Ollama 或 vLLM

```env
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://model-gateway.example/v1
LLM_API_KEY=
LLM_MODEL=deployed-chat-model
LLM_TOKENIZER_MODEL=matching-huggingface-tokenizer

EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_BASE_URL=http://embedding-gateway.example/v1
EMBEDDING_MODEL=deployed-embedding-model
EMBEDDING_DIMENSION=1024

RERANKER_PROVIDER=remote
RERANKER_BASE_URL=http://reranker.example/v1
RERANKER_MODEL=deployed-reranker-model
```

LLM 适配 Chat Completions，Embedding 适配 `/embeddings`，Reranker 适配常见 `/rerank` 的 `results[index,relevance_score]`。Ollama/vLLM 只要暴露兼容契约即可使用相同 Provider；其他特殊协议通过独立 Adapter 接入。

Azure OpenAI 使用独立 Adapter，不会误发 Bearer 鉴权或普通 `model` 字段。LLM、Embedding 与 Judge 可以使用不同资源和 deployment；每一项都必须配置资源根 URL、API Key、API version 和 deployment：

```env
LLM_PROVIDER=azure_openai
LLM_BASE_URL=https://your-resource.openai.azure.com
LLM_API_KEY=replace-me
LLM_API_VERSION=2024-10-21
LLM_DEPLOYMENT=chat-production
LLM_TOKENIZER_MODEL=matching-huggingface-tokenizer

EMBEDDING_PROVIDER=azure_openai
EMBEDDING_BASE_URL=https://your-resource.openai.azure.com
EMBEDDING_API_KEY=replace-me
EMBEDDING_API_VERSION=2024-10-21
EMBEDDING_DEPLOYMENT=embedding-production
EMBEDDING_DIMENSION=1024
```

换 Embedding 时先校验维度，生成 Alembic migration，清空/重建不兼容向量并完整重索引；报告必须记录新模型和配置快照。Judge 应尽量与被测 LLM 使用不同部署。

## 23. 已知限制

- 首版迁移固定 `vector(1024)`，多 Embedding 维度并存需要独立向量表或多列设计。
- 采集默认使用 Embedding 模型对应的 Hugging Face tokenizer；远程私有模型必须提供可加载的 `CHUNK_TOKENIZER_MODEL`。Regex 仅能通过显式配置启用或作为显式应急降级，不能视为模型精确计数。
- citation-safe SSE 在校验后发送，不降低首个 token 时间。
- 本地 BGE 模型权重大、冷启动慢，生产更适合独立 GPU 服务；Compose 默认安装本地模型依赖，直接构建基础 Dockerfile 时可通过 build arg 选择是否安装。
- GitHub GraphQL、分布式任务系统、跨区域容灾和租户级 RBAC 不在当前实现范围。
- 无答案分数依赖具体 Reranker/语料，生产前必须用审核数据校准，不应照搬其他模型阈值。
- Reranker 阈值只应用于真实 Cross-Encoder 输出；禁用或故障降级时保留融合/向量分数原始量纲，不伪造成 rerank score，并使用主题、通道和可用向量信号做保守拒答。
- Kubernetes 文档会持续变化；示例数据集是带 source hash 的固定快照，不代表最新上游事实。
- Loader 目前按单个 source 物化一次快照；超大型 Issues 首次全量同步会占用较多内存和 API 配额。默认 Kubernetes source 将分页上限提高到 5000，生产仍应配置 `GITHUB_TOKEN`，必要时按 state/source 拆分；流式页级 checkpoint 属于后续演进。

## 24. 后续优化方向

- 扩展现有无答案 calibration CLI，加入 Reranker 阈值联合搜索和可靠性图。
- 增加评测任务取消、优先级、显式重试和租户并发配额 API。
- 支持多向量模型版本、在线重建索引和零停机切换。
- 引入 KEP/博客 Loader、GitHub GraphQL 增量同步和 webhook 驱动更新。
- 增加 OpenTelemetry trace、Prometheus 业务阶段指标及 Grafana dashboard。
- 支持租户隔离、细粒度 RBAC、审计保留策略和对象存储原始文档。
- 增加 learned sparse retrieval、query expansion、父文档检索和多路 Reranker 对比。
- 基于实际失败案例持续维护人工审核评测集，而不是用自动生成数据替代人工标注。

## 25. Makefile 命令

| 命令 | 作用 |
| --- | --- |
| `make install` / `make install-models` / `make install-telemetry` | 安装开发、本地模型或 OpenTelemetry 可选依赖 |
| `make dev` | 开发 Uvicorn |
| `make test`, `test-unit`, `test-integration` | 测试集合 |
| `make lint`, `format`, `format-check`, `typecheck`, `precommit` | 质量检查 |
| `make migrate`, `make migration name=...` | 升级/生成迁移 |
| `make seed`, `make seed-eval`, `make ingest` | 写入 source / 写入固定评测 fixture 索引 / 同步知识库 |
| `make eval-dataset`, `make evaluate` | 生成数据集 / 运行评测 |
| `make benchmark` | 对运行中的 API 做 HTTP 基准 |
| `make docker-up`, `make docker-down` | 启停 Compose，不删除卷 |
| `make clean` | 仅删除仓库内工具缓存和构建产物 |

许可证声明：Apache-2.0（见 `pyproject.toml`）。
