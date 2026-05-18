# Data Schema

## LegalDocument

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `doc_id` | string | 是 | 文档唯一 ID。 |
| `title` | string | 是 | 文档标题。 |
| `text` | string | 是 | 文档正文。 |
| `doc_type` | string | 否 | `statute`、`case`、`policy`、`contract`。 |
| `jurisdiction` | string | 否 | 默认 `CN`。 |
| `status` | string | 否 | 默认 `effective`。 |
| `source_url` | string | 否 | 来源链接或文件路径。 |
| `metadata` | object | 否 | `tenant_id`、`issuer`、`legal_rank`、`source_file` 等。 |

## LegalChunk

| 字段 | 类型 | 说明 |
|---|---|---|
| `chunk_id` | string | 稳定 hash ID。 |
| `doc_id` | string | 所属文档。 |
| `title` | string | 文档标题。 |
| `article_no` | string | 法条编号，例如 `第八十二条`。 |
| `text` | string | chunk 文本。 |
| `tenant_id` | string | 租户隔离字段，默认 `public`。 |

## Evaluation Case

```json
{
  "id": "labor_unsigned_contract_001",
  "category": "article_lookup",
  "question": "未签劳动合同的二倍工资依据是什么？",
  "gold_article_nos": ["第八十二条"],
  "gold_terms": ["二倍", "书面劳动合同"],
  "difficulty": "easy"
}
```

## 私有数据规则

- `data/eval/*_sample.jsonl` 可以提交。
- `data/lawrag_corpus.jsonl` 不提交。
- 用户上传合同、案件材料和内部制度默认视为私有数据。
- 含身份证号、手机号、地址、银行卡号等敏感信息的数据不得提交。

## SQLite Runtime State

默认运行态数据库路径：

```text
data/lawrag_state.sqlite3
```

该文件由 `LAWRAG_DB_PATH` 配置，属于本地运行数据，不提交 Git。

| 表 | 主键 | 用途 |
|---|---|---|
| `sessions` | `session_id` | 多轮咨询中的已确认事实、推断主题、缺失事实和 turns。 |
| `qa_logs` | `qa_id` | 每次问答的 query、Markdown answer、facts、citations、verification。 |
| `reviews` | `review_id` | 需要人工复核的回答、风险原因、状态和处理备注。 |
| `feedback` | `feedback_id` | 用户评分、问题类型、文字反馈和关联 session。 |
