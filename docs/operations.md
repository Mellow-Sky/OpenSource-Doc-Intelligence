# Operations runbook

## 启动前检查

1. 以 secret manager 或环境变量提供密钥，不把 `.env` 提交到版本库。
2. 备份 PostgreSQL，先在影子库执行 `alembic upgrade head`。
3. 检查 Embedding 维度与数据库 `vector(N)` 一致；`/ready` 会实际读取 `chunks.embedding` typmod 再比对 `DATABASE_VECTOR_DIMENSION`。
4. 运行 `docker compose config`，确认没有意外的空 endpoint 或明文生产密码。
5. 启动后同时检查 `/health` 和 `/ready`；只有后者表示依赖可用。

Compose 优先使用完整的 `DATABASE_URL`，因此可直接指向托管 PostgreSQL。使用内置 `postgres` 服务时，`DATABASE_URL` 中的密码必须与 `POSTGRES_PASSWORD` 一致；`@`、`:`、`/` 等 URL 保留字符必须百分号编码。

`/ready` 的 OpenAI-compatible 探针默认校验模型目录，Azure 默认用最小推理校验 deployment，Remote Reranker 默认调用单文档 `/rerank`。后两者可能产生很少的计费使用；将 `/health` 用于高频 liveness，并用 `*_HEALTHCHECK_MODE` 选择部署适用的 readiness 契约。仅返回 JSON 健康状态的 Reranker 服务可设 `RERANKER_HEALTHCHECK_MODE=endpoint` 和安全的相对 `RERANKER_HEALTHCHECK_RESOURCE`。

## 备份与恢复

权威状态位于 PostgreSQL，包括文档版本、Chunk、任务、会话、引用、usage、评测样本和带样本外键的评测结果。使用组织批准的 `pg_dump`/物理备份策略，并对恢复演练计时。Git checkout、模型缓存和 FTS/向量索引都可重建，但重建成本较高；评测报告若需长期审计，应从容器卷复制到受版本控制或不可变对象存储。

恢复后执行：

```bash
uv run alembic current
curl -i http://localhost:8000/ready
```

再对一个已知 source 做 dry-run，同步前确认不会异常软删除大量文档。

## 发布与回滚

- 每个 schema 变化必须有向前 Alembic migration；先迁移后滚动发布 API/worker。
- migration 执行期间只运行一个 migrate job。
- Prompt、模型和检索参数变化应先运行固定指纹评测集，保存对比报告。
- 应用回滚前确认旧代码兼容新 schema；不默认执行 destructive downgrade。

## 任务运维

采集任务状态为 pending/running/succeeded/failed。worker 用 heartbeat 表示租约；进程死亡后，过期 running job 可由其他 worker 重领。评测任务也由独立 evaluation worker 领取，过期 running 运行按 `EVALUATION_STALE_SECONDS` 恢复。`*_MAX_OUTSTANDING_*` 控制数据库待处理容量，`*_MAX_CONCURRENT_*` 通过 PostgreSQL 门锁限制跨副本运行数；不要让 Worker 副本数成为隐式并发配置。队列满时 API 返回 429/`Retry-After`，调用方应复用 idempotency key 退避重试。不要直接把任务改成 succeeded。失败错误经过密钥/Authorization/URL credential 脱敏并截断，但仍应按内部日志保留策略处理。

升级到 `0002` 时会从历史 `evaluation_results.metrics._evaluation.case` 快照回填
`evaluation_cases` 并补外键。迁移后可用以下 SQL 审计；结果应为 0：

```sql
SELECT count(*)
FROM evaluation_results
WHERE evaluation_case_id IS NULL;
```

`0003` 为 `usage_records` 增加非负的 `input_text_count` 和
`input_character_count`。采集 Worker 的 `ingestion_embedding.request_id` 等于 job ID，
可同时核对任务统计和实际模型输入：

```sql
SELECT operation, model, provider,
       sum(input_text_count) AS texts,
       sum(input_character_count) AS characters,
       sum(prompt_tokens) AS input_tokens,
       sum(latency_ms) AS embedding_latency_ms,
       CASE WHEN count(estimated_cost) = count(*) THEN sum(estimated_cost) END AS cost_usd
FROM usage_records
WHERE request_id = :job_id
GROUP BY operation, model, provider;
```

成本列为 `NULL` 表示至少一个批次没有配置可信价格，不应解释为零成本。

完整快照允许软删除缺失文档；不确定上游结果是否完整时使用 `--no-delete` 或 API 的 `allow_delete_missing=false`。先检查统计中的 scanned/deleted 比例，再决定正式同步。

## 容量与性能

- 监控数据库连接池、HNSW 索引大小、Chunk 数、软删除比例和 autovacuum。
- 本地模型按进程复制内存；默认一个 API worker。远程 Provider 可横向扩展 API。
- Embedding/Reranker 使用批量和并发限制；先测 Provider 限额再提高。
- `/metrics` 使用 route template 作为 label，不要新增 query、user ID 或 UUID 等高基数标签。
- 使用 `scripts/benchmark.py` 保存每次发布的真实 HTTP 基准，不把单元测试耗时当生产性能。

## 安全响应

若怀疑密钥泄露：立即在供应商侧轮换，更新 secret，重启 API/worker，并审计 access log；不要只从 `.env` 删除旧值。错误响应不会返回堆栈，详细异常只进入结构化服务日志。管理接口必须配置 `ADMIN_API_KEY` 并在网关层增加网络策略、速率限制和 TLS。

## 告警建议

- `/ready` 连续失败；
- HTTP 5xx 或 Provider/RateLimit 错误率上升；
- ingestion heartbeat 过期或 failed 增长；
- P95/P99 总耗时和各检索阶段耗时异常；
- no-answer 比例、引用覆盖率或 reranker degradation 突变；
- usage 中 unpriced operation 增长；
- PostgreSQL 存储、连接、复制延迟和备份失败。
