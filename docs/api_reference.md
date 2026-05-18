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

## GET `/`

打开无构建步骤的 Web demo。
