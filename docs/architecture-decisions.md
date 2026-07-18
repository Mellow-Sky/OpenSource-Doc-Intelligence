# Architecture decisions

本文记录会影响数据兼容性、可靠性和生产运维的首版决策。实现变更若破坏这些约束，应新增 Alembic migration、测试和决策记录，而不是静默修改。

## ADR-001: PostgreSQL 同时承载权威数据与检索索引

状态：Accepted

文档版本、Chunk、FTS、向量、会话和审计使用同一个 PostgreSQL 事务域。这样一次回答可以把消息、检索排名、引用和 usage 原子提交，metadata filter 也无需跨存储拼接。

代价是向量规模最终会受单集群容量约束。达到容量边界后可把召回索引投影到专用搜索系统，但 PostgreSQL 中的 Chunk 和审计记录仍是权威来源，投影必须支持重建。

## ADR-002: FTS 与向量并发召回，融合后再重排

状态：Accepted

英文技术文档的专有名词、版本和代码标识符适合关键词搜索，语义改写适合向量搜索。两路分别保留原始 rank/score，RRF 默认 `k=60`，也允许归一化加权融合。Cross-Encoder 只处理融合后的有限候选；不可用时降级到融合结果并写入 degradation metadata。

## ADR-003: 结构优先 Chunk 与稳定增量对齐

状态：Accepted

标题、代码块和表格边界优先于固定字符窗口。超长章节才按段落/Token 拆分，所有 Chunk 保存 heading path、字符 offset、行号、父 Chunk 和内容哈希。更新时用稳定序列对齐复用未变化 Chunk/Embedding，删除使用软删除。

固定迁移使用 `vector(1024)`。维度属于数据库 schema，不是可以无迁移切换的普通运行参数。

## ADR-004: 文档内容是数据，不是指令

状态：Accepted

上下文使用编号 SOURCE 边界，系统 Prompt 明确忽略文档中的角色修改、密钥泄露和执行指令。模型输出的引用编号必须映射到本次构建的 source table，不能信任模型给出的 URL 或 Chunk ID。

## ADR-005: 无答案决定不能只交给生成模型

状态：Accepted

在生成前组合召回通道、Top-1/平均分、margin、主题重叠和可选 evidence judge；生成后再用引用覆盖率否决。分数阈值与具体模型绑定，未校准的阈值保持空值，不能用魔法数字假装通用。

## ADR-006: Provider 和价格使用精确配置

状态：Accepted

LLM、Embedding、Reranker、Judge 通过端口隔离，业务服务只依赖协议。模型名、URL、密钥和并发边界全部来自 Settings。价格表按精确 provider/model 查找，缺失即 `null`，禁止回退到相似模型价格。

## ADR-007: 长任务使用数据库租约队列

状态：Accepted

采集 API 只写 pending job；worker 在短事务中 claim，慢 I/O 在事务外执行，并通过 heartbeat 和 compare-and-set 完成。任务用 idempotency key 去重。入队 count+insert 与 worker count+claim 都使用按队列命名的 PostgreSQL transaction advisory lock，因此容量背压和运行并发上限在多 API、多 Worker 副本之间保持原子。此规模不引入额外分布式队列；当吞吐、优先级或跨区域需求出现时再替换任务基础设施。

## ADR-008: SSE 在引用校验后发送

状态：Accepted

当前流式接口先完成生成、无答案和引用校验，再发 metadata/delta/done。它牺牲首 token 延迟，换取不向客户端发送随后会被否决的无来源文本。若未来实现真正 token streaming，需要增加可撤回协议或流式 claim/citation 验证，不能直接透传不受控 token。

## ADR-009: 评测样本使用数据集内稳定身份

状态：Accepted

`evaluation_cases` 以 `(dataset_name, external_id)` 唯一标识一个可复现样本。worker 在完成
租约的同一事务内批量 upsert 完整样本、批量插入结果并填写外键；这样不会出现运行已成功
但结果无法追溯到样本，或逐条提交造成半成品。结果 JSON 仍嵌入 case 快照，使报告可以脱离
数据库审计，也让 migration 能从旧版本结果恢复缺失的 case 行与关联。
