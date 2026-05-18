# API Reference

Base URL:

```text
http://127.0.0.1:8011
```

## GET `/health`

返回服务状态、文档数和 chunk 数。

```json
{
  "status": "ok",
  "documents": 3,
  "chunks": 6
}
```

## POST `/api/v1/qa`

法律问答接口。

请求：

```json
{
  "query": "我工作8个月没签劳动合同，月薪8000，可以赔多少？",
  "tenant_id": "public",
  "session_id": "demo-session"
}
```

响应字段：

| 字段 | 说明 |
|---|---|
| `status` | `ok` 或 `needs_review`。 |
| `session` | 多轮会话状态。 |
| `answer` | Markdown 风格答案。 |
| `structured_answer` | 结构化答案。 |
| `citations` | 引用来源列表。 |
| `facts` | 识别事实、缺失事实、推断和风险。 |
| `verification` | 引用校验结果。 |
| `review` | 若回答需要人工复核，返回复核编号和风险原因。 |

`verification` 现在包含：

| 字段 | 说明 |
|---|---|
| `claim_checks` | 每条法律 claim 的引用支持情况。 |
| `citation_spans` | 支持 claim 的原文 span，含 `start_char`、`end_char` 和 `support_score`。 |
| `risk_reasons` | 触发复核的原因。 |

## POST `/api/v1/search`

检索接口。

请求：

```json
{
  "query": "未签劳动合同 二倍工资",
  "top_k": 3,
  "tenant_id": "public"
}
```

## POST `/api/v1/documents/import`

文档导入接口。

请求：

```json
{
  "persist": true,
  "documents": [
    {
      "title": "样例员工手册",
      "text": "第一条 用人单位应当依法与员工订立书面劳动合同。",
      "doc_type": "policy",
      "metadata": { "tenant_id": "public" }
    }
  ]
}
```

## GET `/api/v1/sessions/{session_id}`

查看多轮会话状态。

## POST `/api/v1/feedback`

提交用户反馈。

```json
{
  "query": "我工作8个月没签劳动合同，月薪8000，可以赔多少？",
  "issue_type": "answer_issue",
  "comment": "引用不够清楚",
  "rating": 1,
  "session_id": "demo-session"
}
```

## GET `/api/v1/feedback`

查看当前内存中的反馈列表。

## GET `/api/v1/review-queue`

查看人工复核队列。可选 query 参数：

```text
status=open
```

## POST `/api/v1/review-queue/{review_id}/resolve`

标记复核项已处理。

```json
{
  "note": "已确认该回答可接受"
}
```

## GET `/`

打开无构建步骤的 Web demo。
