# Evaluation Plan

本项目评测目标是定位 RAG 链路错误，而不是只判断回答“看起来好不好”。

## 当前可跑指标

`lawrag_qa.evaluation` 支持轻量检索评测：

- `Recall@k`
- `MRR`
- 按 category 汇总

运行：

```powershell
python -m lawrag_qa.evaluation --cases data/eval/labor_qa_sample.jsonl --top-k 5
```

## 评测集分层

| 层级 | 目标 | 当前状态 |
|---|---|---|
| 法条定位 | 正确召回目标法条。 | 已有样例。 |
| 场景咨询 | 召回相关规则并指出缺失事实。 | 已有样例。 |
| 计算题 | 二倍工资、经济补偿等公式化任务。 | 已有样例。 |
| 无法回答 | 依据不足时保守回答。 | 待扩展。 |
| 红队安全 | 伪造证据、错误前提、跨法域混淆。 | 已有小样例。 |

## 下一阶段指标

- `Citation Support Rate`
- `Unsupported Claim Rate`
- `Context Carryover Accuracy`
- `Risk Flag Recall`
- `Red Team Refusal Precision`

## 评测门槛

MVP 阶段：

- `Recall@5 >= 0.80`
- 样例测试全部通过
- 红队样例不得生成伪造法条或伪造案例

这些阈值会随着真实语料和更大评测集逐步收紧。
