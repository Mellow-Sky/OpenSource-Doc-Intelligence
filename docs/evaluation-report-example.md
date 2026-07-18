# Evaluation report field example

本文件是报告字段示例，不是一次模型评测结果。`null` 表示尚未运行或价格未知；这里没有填入虚构分数、模型输出、延迟或成本。

实际运行：

```bash
make seed-eval
uv run python scripts/run_evaluation.py \
  --dataset evaluation/datasets/kubernetes_eval.jsonl \
  --output evaluation/reports/latest
```

生成目录：

```text
evaluation/reports/latest/
├── report.json       完整运行、配置、分组摘要和逐条结果
├── report.md         人类可读汇总与失败案例
├── results.jsonl     每条样本的完整 trace
└── results.csv       便于表格分析的扁平结果
```

Markdown 汇总的关键区域如下：

| 区域 | 示例字段 | 未运行示例值 |
| --- | --- | --- |
| Identity | run ID、dataset fingerprint、Git commit、UTC 时间 | `null` |
| Retrieval | Recall@1/3/5/10/20、relevant-set recall、MRR | `null` |
| Answer | Exact Match、Token F1、关键词/版本一致性、Judge 四维评分 | `null` |
| Citation | precision、recall、correctness、completeness | `null` |
| No answer | accuracy、precision、recall、F1、FPR、FNR、错答/错拒数 | `null` |
| Performance | mean、P50/P90/P95/P99、min/max、throughput、error rate | `null` |
| Usage | prompt/completion/judge/total token | `null` |
| Cost | average/total USD、cost complete | `null` / `false` |
| Breakdown | category、difficulty、answerable/unanswerable | `null` |
| Failures | execution errors、未召回、引用错误、最差 Token F1 | 无实际样本 |

一次有效的对比必须满足：

1. `dataset_fingerprint` 相同；
2. 报告记录相同或明确不同的 Git commit、Prompt hash 和配置；
3. Judge 模型和被测模型分别记录；
4. 未知模型价格保持 `null`，不能按零成本比较；
5. 自动生成且未人工审核的样本不能被描述为生产准确率证明。

使用 `--compare` 后还会生成 `comparison.json`、`comparison.csv` 和 `comparison.md`，按实验列出 Recall、MRR、Token F1、citation、no-answer、P95、Token 和成本，并单独分析 query rewrite 的 Recall 变化。
