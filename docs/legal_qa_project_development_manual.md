# 法律问答 RAG 项目开发手册（完整版）

版本：v1.1  
日期：2026-05-18  
适用对象：准备做法律问答、法律检索、法律咨询辅助、合同/法规问答项目的算法工程师、后端工程师和实习求职者。  
默认场景：中文法律问答系统，优先选择劳动法、合同法、消费者权益、婚姻家事、企业合规中的一个垂直领域做 MVP。

> 重要声明：本项目是法律信息检索与问答辅助系统，不提供正式法律意见，不替代律师、法务或司法机关判断。所有回答必须展示依据、适用范围、风险提示和人工复核建议。

---

## 1. 项目定位

### 1.1 项目名称

推荐名称：`LawRAG-QA`、`LexiRAG-CN`、`LegalCopilot-RAG`

### 1.2 项目目标

构建一个基于权威法律语料的法律问答系统，支持用户输入自然语言法律问题，系统通过法规、案例、合同条款或企业制度检索，生成带引用依据、风险提示和可追溯来源的回答。

### 1.3 核心原则

1. 检索优先：法律问答的可靠性主要来自正确检索，而不是让模型凭记忆回答。
2. 引用强约束：答案必须绑定来源，引用必须来自检索上下文。
3. 时效敏感：法规有施行日期、修订日期、失效状态，不能忽略。
4. 辖区敏感：不同国家、地区、法院层级、部门规章的适用范围不同。
5. 风险透明：对不确定、事实不足、需要律师介入的情况明确提示。
6. 评测闭环：项目必须有测试集、检索指标、引用准确率、答案忠实度和 bad case 分析。

### 1.4 不做什么

- 不直接承诺诉讼结果。
- 不生成伪造案例、伪造法条、伪造判决编号。
- 不处理未经脱敏的个人敏感信息。
- 不绕过法律数据库、裁判文书网站或商业数据源的使用条款。
- 不把模型回答包装成“律师正式意见”。

---

## 2. 用户与场景

### 2.1 目标用户

| 用户 | 典型问题 | 项目价值 |
|---|---|---|
| 普通用户 | 被辞退怎么赔偿、合同违约怎么办 | 用通俗语言解释法律依据 |
| 学生/研究者 | 某类法条如何适用、案例有哪些 | 快速检索和归纳材料 |
| 企业法务 | 员工制度、合同条款、合规要求 | 内部知识库问答 |
| 律师助理 | 案例检索、法规定位、初步分析 | 减少资料搜集时间 |
| 产品/运营 | 平台规则、消费者争议 | 标准化答复与风险提醒 |

### 2.2 推荐 MVP 场景

优先选一个垂直领域，不建议第一版覆盖所有法律：

- 劳动法问答：辞退、赔偿、试用期、社保、加班、竞业限制。
- 合同法问答：违约责任、解除合同、定金、格式条款、管辖条款。
- 消费者权益问答：退款、质量问题、虚假宣传、平台责任。
- 企业制度问答：员工手册、报销制度、数据安全制度、合同模板。

### 2.3 用户故事

```text
作为普通用户，
我希望输入“公司没签劳动合同，工作 8 个月能赔多少”，
系统能告诉我需要补充哪些事实、相关法律依据、可能的赔偿方向和风险点，
并给出法条引用，而不是只给一个绝对结论。
```

```text
作为企业法务，
我希望上传公司制度和合同模板，
系统能基于内部文档回答问题，
并标出答案来自哪一条制度或合同条款。
```

---

## 3. 功能需求

### 3.1 MVP 功能

| 模块 | 功能 | 验收标准 |
|---|---|---|
| 文档导入 | 支持法规、案例、制度、合同文本导入 | 能解析 txt/md/pdf/docx 中至少 2 种格式 |
| 文本结构化 | 按法条/条款/段落切分 | chunk 保留标题、条号、来源、日期 |
| 检索 | BM25 + 向量检索 | Top-k 能召回目标法条或条款 |
| 重排序 | reranker 对候选片段排序 | 相关片段排到前 5 |
| 问答生成 | 按固定格式回答 | 包含结论、依据、分析、风险提示 |
| 引用溯源 | 每段核心结论带引用 | 引用能点回原文片段 |
| 评测 | 构建测试集和指标 | 至少 50 个问题，有结果表 |
| Web/API | 提供问答接口和简单页面 | 用户可以交互提问 |

### 3.2 V1 增强功能

- 多轮追问：记住案件事实，但不把历史错误当事实。
- 澄清问题：事实不足时先问用户，例如地区、时间、合同类型、金额、工作年限。
- 法律时效过滤：按施行日期、修订日期、失效状态过滤。
- 案例相似检索：按案由、法院、事实争点、裁判观点检索。
- 对比回答：不同地区、不同司法实践的差异。
- 反馈系统：用户标记“有用/无用/引用错误/答非所问”。
- 管理后台：文档导入、索引状态、评测结果、bad case 管理。

### 3.3 V2 可探索功能

- 法律 Agent：自动拆解“查法条、查案例、算赔偿、生成文书草稿”。
- 合同审查：识别高风险条款，给出修改建议和依据。
- 类案检索：根据事实摘要找类似裁判文书。
- SFT/LoRA：用项目积累的高质量问答数据微调模型。
- 自动评测平台：持续监控检索召回、引用准确率和答案质量。

---

## 4. 非功能需求

| 类别 | 要求 |
|---|---|
| 准确性 | 回答不得脱离检索依据；不知道时必须说明不确定 |
| 可追溯 | 每条引用都能定位到原始文档、条款、段落 |
| 可维护 | 数据导入、索引构建、模型配置可独立替换 |
| 性能 | MVP 单问答响应建议小于 10 秒；检索阶段小于 2 秒 |
| 安全 | 用户输入和上传文档要脱敏、鉴权、审计 |
| 合规 | 遵守数据源使用条款、隐私保护和版权要求 |
| 可评测 | 每次改动后能复跑固定评测集 |

---

## 5. 总体架构

```mermaid
flowchart TD
    U["用户问题"] --> API["FastAPI / Web API"]
    API --> QP["Query Processor<br/>意图识别/事实抽取/澄清判断"]
    QP --> R1["BM25 检索<br/>Elasticsearch/OpenSearch"]
    QP --> R2["向量检索<br/>pgvector/Qdrant/Milvus/FAISS"]
    R1 --> MERGE["候选合并与去重"]
    R2 --> MERGE
    MERGE --> RR["Reranker 重排序"]
    RR --> CTX["上下文组装<br/>引用编号/元数据/时效过滤"]
    CTX --> LLM["LLM 生成<br/>API 或 vLLM 本地服务"]
    LLM --> V["答案校验<br/>引用检查/事实一致性/风险规则"]
    V --> RESP["答案 + 法律依据 + 风险提示"]
    RESP --> U

    D["法规/案例/合同/制度"] --> PARSE["文档解析"]
    PARSE --> CLEAN["清洗与结构化"]
    CLEAN --> SPLIT["法条/条款级切分"]
    SPLIT --> EMB["Embedding"]
    SPLIT --> IDX1["关键词索引"]
    EMB --> IDX2["向量索引"]
    IDX1 --> R1
    IDX2 --> R2
```

---

## 6. 推荐技术栈

### 6.1 标准方案

| 层 | 推荐技术 | 说明 |
|---|---|---|
| 后端 | FastAPI | Python 生态好，便于算法集成 |
| 前端 | React / Next.js / Streamlit | MVP 可用 Streamlit，作品展示可用 React |
| 数据库 | PostgreSQL | 存文档、chunk、用户、反馈、评测记录 |
| 向量库 | pgvector / Qdrant / Milvus / FAISS | MVP 用 pgvector 或 FAISS，生产用 Qdrant/Milvus |
| 关键词检索 | Elasticsearch / OpenSearch | BM25、字段过滤、中文分词 |
| 文档解析 | MinerU / Docling / unstructured / PyMuPDF | PDF、表格、标题结构解析 |
| Embedding | BGE-M3、bge-large-zh、text-embedding 系列 | 中文法律场景建议先试 BGE-M3 |
| Reranker | BGE reranker、cross-encoder reranker | 法律 QA 强烈建议加 rerank |
| LLM | Qwen、DeepSeek、GLM、GPT、Claude 等 | MVP 可用 API，本地部署可用 vLLM |
| 推理服务 | vLLM | 本地模型 OpenAI-compatible API |
| 评测 | Ragas、自建脚本、DeepEval、OpenCompass 思路 | 法律项目必须加人工评测 |
| 部署 | Docker Compose | 本地和服务器部署一致 |

### 6.2 MVP 简化方案

如果只想快速做作品：

```text
FastAPI + Streamlit
PostgreSQL + pgvector
Elasticsearch 或直接 BM25 库
BGE-M3 embedding
BGE reranker
通用 LLM API
自建 50-100 条评测集
```

### 6.3 不建议的第一版方案

- 第一版就训练法律大模型。
- 第一版就做复杂多 Agent。
- 只用向量检索，不做 BM25。
- 只做 demo 页面，不做评测集。
- 只用模型内置知识回答，不接权威法律库。

---

## 7. 仓库结构

推荐结构：

```text
lawrag-qa/
  README.md
  docs/
    product_requirements.md
    data_schema.md
    evaluation_report.md
    deployment.md
  configs/
    app.yaml
    retrieval.yaml
    prompts.yaml
  data/
    raw/
    processed/
    eval/
  src/
    app/
      main.py
      routers/
      schemas/
      services/
    ingestion/
      parse_documents.py
      normalize_legal_text.py
      split_legal_chunks.py
      build_index.py
    retrieval/
      bm25_search.py
      vector_search.py
      hybrid_search.py
      rerank.py
    generation/
      prompt_builder.py
      llm_client.py
      answer_generator.py
      citation_verifier.py
    evaluation/
      eval_retrieval.py
      eval_answer.py
      eval_citation.py
      make_eval_set.py
    common/
      logging.py
      config.py
      text_utils.py
  tests/
    test_splitter.py
    test_retrieval.py
    test_citation.py
  docker-compose.yml
  pyproject.toml
  .env.example
```

---

## 8. 数据建设

### 8.1 数据类型

| 类型 | 示例 | 关键字段 |
|---|---|---|
| 法律法规 | 法律、行政法规、司法解释、地方性法规 | 名称、条号、效力层级、施行日期、修订日期、失效状态 |
| 裁判文书 | 判决书、裁定书 | 案号、法院、案由、裁判日期、事实、理由、结果 |
| 合同模板 | 劳动合同、采购合同、租赁合同 | 条款编号、主题、风险标签 |
| 企业制度 | 员工手册、报销制度、合规制度 | 部门、版本、生效日期、适用范围 |
| QA 评测集 | 人工问题、标准依据、标准答案 | 问题、目标法条、答案要点、难度 |

### 8.2 数据源建议

公开数据源需要遵守使用条款：

- 国家法律法规数据库：https://flk.npc.gov.cn/
- 中国裁判文书网：https://wenshu.court.gov.cn/
- CAIL、LawBench、LegalBench、LegalBench-RAG、LexRAG 等公开研究数据。
- 企业内部文档：必须确认授权、脱敏和访问控制。

### 8.3 数据处理流程

```mermaid
flowchart LR
    RAW["raw 文档"] --> PARSE["解析"]
    PARSE --> CLEAN["清洗"]
    CLEAN --> NORM["字段标准化"]
    NORM --> SPLIT["法律结构切分"]
    SPLIT --> QC["质量检查"]
    QC --> STORE["入库"]
    STORE --> INDEX["构建索引"]
```

### 8.4 清洗规则

- 去除页眉页脚、重复页码、水印、目录噪声。
- 保留条号、款号、项号、标题层级。
- 标准化全角半角、空格、换行、标点。
- 对裁判文书保留案号、法院、裁判日期、案由。
- 对内部文档记录版本号、生效日期和适用部门。
- 对含个人信息的文本做脱敏：姓名、手机号、身份证号、地址、银行卡号。

### 8.5 法律文本切分策略

普通 chunk 切分不适合法律项目。推荐按结构切：

#### 法规切分

```text
法律名称
  编/章/节
    第 N 条
      第 1 款
      第 2 款
        第 1 项
```

chunk 粒度建议：

- 最小粒度：条、款、项。
- 检索粒度：条级为主，必要时合并上下文。
- 展示粒度：返回完整条文和来源。

#### 裁判文书切分

按以下字段切分：

- 基本信息：案号、法院、日期、案由。
- 当事人诉辩。
- 法院查明事实。
- 法院认为。
- 裁判结果。

优先检索“法院认为”和“裁判结果”，但回答时要提示案例不是普遍规则。

#### 合同/制度切分

按条款编号、标题和主题切分：

```text
第八条 违约责任
8.1 延迟交付
8.2 违约金
8.3 损失赔偿
```

### 8.6 元数据设计

每个 chunk 至少包含：

```json
{
  "chunk_id": "law_劳动合同法_第八十二条_1",
  "doc_id": "law_劳动合同法",
  "doc_type": "statute",
  "title": "中华人民共和国劳动合同法",
  "article_no": "第八十二条",
  "paragraph_no": "第一款",
  "text": "用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的...",
  "jurisdiction": "CN",
  "legal_level": "法律",
  "effective_date": "2008-01-01",
  "amended_date": "2012-12-28",
  "status": "effective",
  "source_url": "https://...",
  "checksum": "..."
}
```

---

## 9. 数据库设计

### 9.1 核心表

#### legal_documents

| 字段 | 类型 | 说明 |
|---|---|---|
| id | string | 文档 ID |
| title | string | 文档标题 |
| doc_type | string | statute/case/contract/policy |
| jurisdiction | string | 辖区 |
| legal_level | string | 效力层级 |
| source_url | string | 来源 |
| effective_date | date | 生效日期 |
| amended_date | date | 修订日期 |
| status | string | effective/expired/draft |
| raw_path | string | 原始文件路径 |
| created_at | datetime | 创建时间 |

#### legal_chunks

| 字段 | 类型 | 说明 |
|---|---|---|
| id | string | chunk ID |
| doc_id | string | 所属文档 |
| parent_id | string | 上级结构 |
| section_path | string | 章/节/条路径 |
| article_no | string | 条号 |
| text | text | chunk 文本 |
| metadata | jsonb | 结构化元数据 |
| embedding | vector | 向量 |
| token_count | int | token 数 |

#### qa_logs

| 字段 | 类型 | 说明 |
|---|---|---|
| id | string | 问答 ID |
| user_query | text | 用户问题 |
| normalized_query | text | 改写后问题 |
| retrieved_chunk_ids | jsonb | 检索结果 |
| answer | text | 模型答案 |
| citations | jsonb | 引用 |
| latency_ms | int | 延迟 |
| created_at | datetime | 时间 |

#### user_feedback

| 字段 | 类型 | 说明 |
|---|---|---|
| id | string | 反馈 ID |
| qa_id | string | 对应问答 |
| rating | int | 评分 |
| issue_type | string | 引用错误/答非所问/过度推断/有帮助 |
| comment | text | 用户备注 |

#### eval_cases

| 字段 | 类型 | 说明 |
|---|---|---|
| id | string | 测试用例 ID |
| question | text | 问题 |
| gold_chunk_ids | jsonb | 标准依据 |
| gold_answer_points | jsonb | 答案要点 |
| difficulty | string | easy/medium/hard |
| domain | string | 劳动法/合同法等 |

---

## 10. 检索系统设计

### 10.1 为什么必须混合检索

法律问题常包含精确术语、法条编号、案号、机构名称，也包含自然语言描述。单一向量检索容易漏掉精确词；单一 BM25 又不擅长语义改写。因此推荐：

```text
BM25 召回 50 条
+ 向量召回 50 条
+ 元数据过滤
+ 去重合并
+ reranker 排序
+ top 5-10 进入生成
```

### 10.2 Query Processor

输入用户问题后，先抽取：

```json
{
  "domain": "劳动法",
  "intent": "咨询赔偿",
  "jurisdiction": "中国大陆",
  "time": "2026",
  "facts": {
    "employment_duration_months": 8,
    "has_written_contract": false
  },
  "missing_facts": ["所在城市", "是否已离职", "工资标准"]
}
```

### 10.3 澄清策略

如果缺少关键事实，不要强行给结论。示例：

```text
为了更准确判断，需要补充 3 个信息：
1. 你所在城市或劳动合同履行地；
2. 月工资标准；
3. 是否已经离职，以及离职原因。

在未补充前，只能给出一般规则。
```

### 10.4 检索伪代码

```python
def retrieve(query: str, filters: dict, top_k: int = 8):
    parsed = parse_query(query)
    bm25_hits = bm25_search(parsed.keyword_query, filters, top_n=50)
    vector_hits = vector_search(parsed.semantic_query, filters, top_n=50)
    merged = merge_and_deduplicate(bm25_hits, vector_hits)
    reranked = rerank(query, merged)
    return reranked[:top_k]
```

### 10.5 Rerank 特征

排序时考虑：

- 语义相关性。
- 关键词命中。
- 法律效力层级。
- 时效状态。
- 条文是否完整。
- 来源可信度。
- 与用户辖区是否匹配。

---

## 11. 生成系统设计

### 11.1 回答格式

推荐统一模板：

```text
【初步结论】
...

【法律依据】
1. 《xxx》第 x 条：...
2. 《xxx》第 x 条：...

【适用分析】
结合你提供的事实，...

【需要补充的信息】
...

【风险提示】
...

【免责声明】
本回答仅为基于检索材料的信息参考，不构成正式法律意见。
```

### 11.2 System Prompt

```text
你是一个法律信息检索与问答辅助系统。你必须遵守以下规则：

1. 只能基于提供的检索材料回答，不得编造法条、案例、案号、法院名称或来源链接。
2. 如果检索材料不足以回答，必须说明“不足以判断”，并提出需要补充的事实。
3. 回答必须区分“一般法律规则”“结合事实的初步分析”“需要律师复核的风险点”。
4. 每个关键结论必须引用材料编号，例如 [S1]、[S2]。
5. 不得承诺诉讼结果，不得声称自己是律师。
6. 对时间、地区、主体身份、合同类型、金额等关键事实保持谨慎。
```

### 11.3 User Prompt 模板

```text
用户问题：
{query}

已识别事实：
{facts}

检索材料：
{contexts}

请按以下结构回答：
1. 初步结论
2. 法律依据
3. 适用分析
4. 仍需补充的信息
5. 风险提示
6. 免责声明
```

### 11.4 上下文格式

```text
[S1]
标题：中华人民共和国劳动合同法
条号：第八十二条
效力：现行有效
施行/修订：2008-01-01 / 2012-12-28
来源：https://...
正文：用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的...

[S2]
标题：中华人民共和国劳动合同法实施条例
条号：第六条
正文：...
```

### 11.5 Citation Verifier

生成后检查：

- 答案中出现的 `[Sx]` 是否存在。
- 引用编号对应的文本是否真的支持该结论。
- 是否出现“根据某某法第几条”，但该条不在上下文。
- 是否出现未检索到的案例名、案号、法院。
- 是否出现绝对化承诺，如“必胜”“一定赔偿”“肯定违法”。

如果校验失败：

1. 降级为保守回答。
2. 要求模型重写。
3. 或返回“依据不足，建议人工复核”。

---

## 12. Agent 设计

MVP 不建议复杂 Agent，但可以设计轻量工具调用。

### 12.1 推荐工具

| 工具 | 作用 |
|---|---|
| `search_law` | 检索法规 |
| `search_case` | 检索案例 |
| `search_policy` | 检索企业制度 |
| `calculate_compensation` | 粗略计算劳动赔偿 |
| `ask_clarification` | 生成澄清问题 |
| `verify_citation` | 校验引用 |

### 12.2 工具调用流程

```mermaid
flowchart TD
    Q["用户问题"] --> PLAN["判断任务类型"]
    PLAN --> LAW["search_law"]
    PLAN --> CASE["search_case 可选"]
    PLAN --> CALC["calculate_compensation 可选"]
    LAW --> DRAFT["生成草稿"]
    CASE --> DRAFT
    CALC --> DRAFT
    DRAFT --> VERIFY["verify_citation"]
    VERIFY --> FINAL["最终回答"]
```

### 12.3 何时不使用 Agent

- 简单法条解释。
- 用户事实不足，需要先澄清。
- 检索结果为空。
- 问题涉及高风险具体法律意见，如刑事案件、重大诉讼策略。

---

## 13. 评测体系

### 13.1 评测集构建

至少准备 50-200 条问题：

| 类型 | 数量建议 | 示例 |
|---|---:|---|
| 法条定位 | 20% | “未签劳动合同赔偿依据是哪条？” |
| 场景咨询 | 40% | “工作 8 个月没签合同，能主张双倍工资吗？” |
| 多事实推理 | 20% | “试用期被辞退且没缴社保怎么办？” |
| 反事实/陷阱 | 10% | “合同写了放弃社保就有效吗？” |
| 无法回答 | 10% | “某地最新内部口径是什么？” |

### 13.2 检索指标

| 指标 | 含义 |
|---|---|
| Recall@k | 标准依据是否在 top-k |
| MRR | 标准依据排名是否靠前 |
| nDCG | 排序质量 |
| Metadata Accuracy | 辖区、时效、效力层级过滤是否正确 |

### 13.3 生成指标

| 指标 | 含义 |
|---|---|
| Faithfulness | 答案是否忠于检索材料 |
| Citation Accuracy | 引用是否真实支持结论 |
| Legal Correctness | 法律结论是否正确 |
| Completeness | 是否覆盖关键要点 |
| Helpfulness | 是否对用户有帮助 |
| Risk Awareness | 是否提示限制和风险 |

### 13.4 人工评分表

| 分数 | 标准 |
|---:|---|
| 5 | 结论准确，依据完整，引用正确，风险提示充分 |
| 4 | 基本正确，有轻微遗漏 |
| 3 | 有帮助但不完整，需人工补充 |
| 2 | 存在明显误导或引用不足 |
| 1 | 法律结论错误或编造依据 |

### 13.5 Bad Case 分类

每个 bad case 记录：

```json
{
  "question": "...",
  "expected": "...",
  "actual": "...",
  "error_type": "retrieval_miss / wrong_citation / over_reasoning / missing_facts / outdated_law",
  "root_cause": "...",
  "fix_plan": "..."
}
```

常见错误：

- 没召回目标法条。
- 召回了过期法规。
- 将案例裁判观点当成普遍法律规则。
- 用户事实不足却给了确定结论。
- 引用了正确法条，但解释方向错误。
- 模型把上下文之外的知识混入答案。

---

## 14. API 设计

### 14.1 问答接口

```http
POST /api/v1/qa
Content-Type: application/json
```

请求：

```json
{
  "query": "公司工作 8 个月没签劳动合同，能赔多少？",
  "domain": "labor",
  "jurisdiction": "CN",
  "session_id": "optional-session-id"
}
```

响应：

```json
{
  "answer": "基于你提供的信息，初步看...",
  "citations": [
    {
      "source_id": "S1",
      "doc_title": "中华人民共和国劳动合同法",
      "article_no": "第八十二条",
      "text": "用人单位自用工之日起超过一个月不满一年...",
      "source_url": "https://..."
    }
  ],
  "missing_facts": ["月工资标准", "是否已经离职"],
  "risk_level": "medium",
  "latency_ms": 6230
}
```

### 14.2 检索接口

```http
POST /api/v1/search
```

请求：

```json
{
  "query": "未签劳动合同 双倍工资",
  "filters": {
    "domain": "labor",
    "status": "effective"
  },
  "top_k": 10
}
```

### 14.3 文档导入接口

```http
POST /api/v1/documents/import
```

支持：

- 文件上传。
- URL 导入。
- 批量目录导入。
- 指定文档类型和辖区。

### 14.4 反馈接口

```http
POST /api/v1/feedback
```

请求：

```json
{
  "qa_id": "qa_123",
  "rating": 2,
  "issue_type": "wrong_citation",
  "comment": "引用的法条没有支持赔偿计算。"
}
```

---

## 15. 前端设计

### 15.1 页面结构

MVP 页面：

1. 左侧：领域选择、地区选择、是否只查法规/案例/内部制度。
2. 中间：问答对话。
3. 右侧：引用来源、检索片段、风险提示。
4. 底部：反馈按钮。

### 15.2 必备交互

- 问题输入框。
- 领域/地区筛选。
- “显示依据”展开按钮。
- 引用片段高亮。
- 复制回答。
- 标记引用错误。
- 继续追问。

### 15.3 回答展示规范

回答中不要只显示大段文字，应结构化展示：

```text
初步结论
法律依据
适用分析
还缺哪些事实
风险提示
来源
```

---

## 16. 部署方案

### 16.1 本地开发

```text
Python 3.10+
PostgreSQL 15+
pgvector
Elasticsearch/OpenSearch
Redis 可选
LLM API Key 或本地 vLLM
```

### 16.2 Docker Compose 服务

建议服务：

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
  postgres:
    image: pgvector/pgvector:pg16
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.x
  redis:
    image: redis:7
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
```

### 16.3 环境变量

```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/lawrag
ELASTICSEARCH_URL=http://localhost:9200
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://localhost:8001/v1
LLM_API_KEY=...
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### 16.4 性能优化

- 检索、rerank、生成分阶段记录耗时。
- embedding 批处理。
- 热门问题缓存。
- 固定法规库预构建索引。
- 上下文 token 控制在模型窗口的 30%-60%。
- 对长文档先检索章节，再检索条款。

---

## 17. 安全与合规

### 17.1 隐私保护

- 上传文档默认私有。
- 个人信息入库前脱敏。
- 问答日志可配置关闭。
- 管理员查看日志需要权限。
- 删除用户数据要有接口。

### 17.2 数据源合规

- 记录每条数据来源。
- 遵守公开网站 robots、版权和使用条款。
- 商业数据库内容不得擅自再分发。
- 内部文档必须经过授权。

### 17.3 法律风险控制

回答必须包含：

- “基于已提供事实”的限定。
- “仅供信息参考”的声明。
- “事实不足时需补充”的提示。
- “重大事项建议咨询专业律师”的提示。

对高风险问题触发保守模式：

- 刑事辩护策略。
- 重大金额诉讼。
- 涉及人身安全。
- 涉及规避监管或违法行为。
- 用户要求伪造证据、规避法律。

---

## 18. 开发里程碑

### 18.1 两周 MVP

| 天数 | 任务 | 产出 |
|---:|---|---|
| 1 | 确定领域和数据源 | PRD、数据清单 |
| 2-3 | 文档解析与清洗 | processed 数据 |
| 4-5 | 法条/条款切分 | chunk 表 |
| 6 | embedding 和向量索引 | vector search 可用 |
| 7 | BM25 检索 | keyword search 可用 |
| 8 | hybrid + rerank | top-k 结果 |
| 9 | prompt 和生成 | 带引用答案 |
| 10 | FastAPI 接口 | `/qa` 可用 |
| 11 | 简单前端 | 可交互页面 |
| 12 | 评测集 50 条 | eval_cases |
| 13 | 跑评测和 bad case | evaluation_report |
| 14 | README 和演示视频 | 可投简历版本 |

### 18.2 四周增强版

第 1 周：数据管道、索引、基础检索。  
第 2 周：rerank、prompt、引用校验、API。  
第 3 周：前端、反馈、评测平台。  
第 4 周：性能优化、bad case 修复、部署、项目文档。

---

## 19. 实验设计

### 19.1 Baseline

必须有 baseline，方便证明改进有效：

| 实验 | 配置 |
|---|---|
| Baseline A | LLM 直接回答 |
| Baseline B | 仅向量检索 + LLM |
| Baseline C | 仅 BM25 + LLM |
| Full | BM25 + 向量 + rerank + citation verifier |

### 19.2 消融实验

建议做：

- 无 rerank vs 有 rerank。
- 普通 chunk vs 法条结构化 chunk。
- 无时效过滤 vs 有时效过滤。
- top-3 vs top-5 vs top-10。
- 不同 embedding 模型对比。
- 不同 prompt 对 citation accuracy 的影响。

### 19.3 结果表模板

| 方法 | Recall@5 | MRR | Citation Accuracy | Faithfulness | Avg Latency |
|---|---:|---:|---:|---:|---:|
| LLM only | - | - | - | 0.62 | 2.1s |
| Vector RAG | 0.71 | 0.48 | 0.68 | 0.76 | 4.8s |
| BM25 RAG | 0.75 | 0.52 | 0.72 | 0.79 | 4.1s |
| Hybrid + Rerank | 0.86 | 0.67 | 0.84 | 0.88 | 6.3s |

---

## 20. Prompt 与回答质量规范

### 20.1 好回答标准

一个合格法律问答回答应该：

- 先给谨慎结论。
- 明确适用前提。
- 给出法律依据。
- 结合用户事实分析。
- 指出还缺哪些事实。
- 提示风险和下一步行动。
- 所有关键结论都有引用。

### 20.2 差回答特征

- 没有引用。
- 引用不存在。
- 直接说“你一定能赢”。
- 不问地区、时间、合同、金额等关键事实。
- 混用不同法律体系。
- 把模型经验当法律依据。
- 把个案判决当普遍规则。

### 20.3 回答示例

用户：

```text
我在公司工作 8 个月了，公司一直没签劳动合同，可以要求赔偿吗？
```

理想回答：

```text
【初步结论】
如果你与公司存在劳动关系，且公司自用工之日起超过一个月仍未与你订立书面劳动合同，通常可以主张未签书面劳动合同的二倍工资差额。但具体金额还需要结合入职时间、月工资、是否已离职、仲裁时效等事实判断。[S1]

【法律依据】
《中华人民共和国劳动合同法》第八十二条规定，用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。[S1]

【适用分析】
你提到已工作 8 个月且未签劳动合同。一般情况下，二倍工资差额计算区间不是从入职当天开始，而是从用工满一个月后的次日起计算，最多计算至满一年之前。具体还要核实工资标准、入职日期、是否有劳动关系证明等。[S1]

【仍需补充的信息】
1. 入职日期；
2. 月工资标准；
3. 是否有考勤、工资流水、工作聊天记录；
4. 所在城市或劳动合同履行地；
5. 是否已经离职。

【风险提示】
劳动争议通常存在仲裁时效问题，建议尽快整理证据并咨询当地劳动仲裁机构或专业律师。

【免责声明】
本回答仅基于检索材料和你提供的信息作一般说明，不构成正式法律意见。
```

---

## 21. 代码实现要点

### 21.1 文档切分函数

```python
ARTICLE_PATTERN = r"(第[一二三四五六七八九十百千万零〇0-9]+条)"

def split_statute(text: str) -> list[dict]:
    """Split legal statute text into article-level chunks."""
    # 实现时保留标题、章/节路径、条号和正文。
    raise NotImplementedError
```

### 21.2 引用检查函数

```python
import re

def extract_citation_ids(answer: str) -> set[str]:
    return set(re.findall(r"\[S(\d+)\]", answer))

def verify_citations(answer: str, contexts: list[dict]) -> dict:
    valid_ids = {str(i + 1) for i in range(len(contexts))}
    used_ids = extract_citation_ids(answer)
    missing = used_ids - valid_ids
    return {
        "ok": len(missing) == 0,
        "missing_citations": sorted(missing),
        "used_citations": sorted(used_ids)
    }
```

### 21.3 混合检索合并

```python
def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores
```

---

## 22. 测试策略

### 22.1 单元测试

- 法条切分是否保留条号。
- 元数据字段是否完整。
- 失效法规是否被过滤。
- citation verifier 是否能发现不存在引用。
- 检索合并是否去重。

### 22.2 集成测试

- 导入一部法规后能检索指定条文。
- 输入问题后返回 answer 和 citations。
- 引用 source_id 能定位到原始 chunk。
- 关闭向量库或 LLM API 时有合理错误提示。

### 22.3 回归测试

每次改动后跑固定评测集：

```bash
python -m src.evaluation.eval_retrieval --cases data/eval/labor_qa.jsonl
python -m src.evaluation.eval_answer --cases data/eval/labor_qa.jsonl
python -m src.evaluation.eval_citation --logs outputs/eval_run.jsonl
```

---

## 23. 日志与可观测性

每次问答记录：

- query 原文。
- query 解析结果。
- 检索 filters。
- BM25 top-k。
- vector top-k。
- rerank 分数。
- 最终上下文。
- LLM 输入 token / 输出 token。
- 延迟。
- citation verifier 结果。
- 用户反馈。

不要记录：

- 未脱敏身份证号、手机号、住址、银行卡号。
- 未授权内部文件全文。
- 用户明确要求不保存的内容。

---

## 24. 项目验收清单

### 24.1 技术验收

- [ ] 能导入至少 1000 个法律 chunk。
- [ ] 能对法规进行条级切分。
- [ ] 支持 BM25 + 向量混合检索。
- [ ] 支持 reranker。
- [ ] 问答返回引用来源。
- [ ] 有引用校验。
- [ ] 有 50-200 条评测集。
- [ ] 有检索和生成评测报告。
- [ ] 有 bad case 分析。
- [ ] 有部署说明。

### 24.2 产品验收

- [ ] 用户知道回答不是正式法律意见。
- [ ] 用户能看到依据。
- [ ] 用户能展开原文。
- [ ] 用户能继续追问。
- [ ] 用户能反馈错误。
- [ ] 高风险问题会提示人工复核。

### 24.3 求职作品验收

- [ ] README 说明问题、方案、架构和效果。
- [ ] 有系统架构图。
- [ ] 有在线 demo 或录屏。
- [ ] 有评测表格。
- [ ] 有失败案例和改进记录。
- [ ] 有技术亮点：结构化 chunk、hybrid retrieval、rerank、citation verifier、evaluation。

---

## 25. 简历写法

### 25.1 一句话项目描述

```text
构建面向劳动法咨询的法律 RAG 问答系统，完成法规结构化解析、混合检索、rerank、引用校验和自动评测闭环，支持带法条依据的可追溯问答。
```

### 25.2 简历 bullet 示例

```text
- 设计法律法规条级切分与元数据建模方案，保留法条编号、效力层级、施行日期、来源链接等字段，支持按时效和辖区过滤。
- 实现 BM25 + BGE-M3 向量检索 + reranker 的混合检索链路，相比仅向量检索提升 Recall@5 和引用准确率。
- 构建引用强约束生成流程，要求模型仅基于检索材料回答，并实现 citation verifier 检查伪造引用和无依据结论。
- 自建 100 条劳动法问答评测集，覆盖法条定位、场景咨询、多事实推理和无法回答问题，输出 bad case 分析和迭代报告。
- 使用 FastAPI 提供问答、检索、文档导入和反馈接口，并用 Docker Compose 完成本地部署。
```

### 25.3 面试讲解顺序

1. 为什么法律问答不能只靠模型参数知识。
2. 数据如何结构化，为什么不用普通 chunk。
3. 为什么 BM25 + 向量 + rerank。
4. 如何防止模型编造法条。
5. 如何构建评测集和分析 bad case。
6. 项目还有哪些风险和改进方向。

---

## 26. 常见问题

### Q1：要不要微调法律大模型？

第一版不建议。先做 RAG 和评测。只有当你已经有高质量标注数据，并且发现模型在固定回答格式、法律语言表达或特定任务上稳定不足时，再考虑 LoRA/SFT。

### Q2：只用 LangChain 可以吗？

可以做 MVP，但不要只展示链式调用。求职项目更看重你是否理解数据结构、检索、rerank、引用校验和评测。

### Q3：法律法规数据可以直接爬吗？

必须先确认数据源使用条款。公开可访问不等于可以无限抓取、商用或再分发。建议开发阶段使用少量公开数据和开源数据集，生产阶段使用授权数据。

### Q4：回答里一定要引用案例吗？

不一定。普通法律规则问题优先引用法规。案例可用于解释司法实践，但要提示个案不等于普遍规则。

### Q5：怎么判断项目做得比普通 RAG 更专业？

看四点：法条结构化、时效/辖区过滤、引用校验、法律评测集。没有这四点，通常只是普通文档问答换了法律语料。

---

## 27. 参考资料

- 国家法律法规数据库：https://flk.npc.gov.cn/
- 中国裁判文书网：https://wenshu.court.gov.cn/
- ChatLaw：https://github.com/PKU-YuanGroup/ChatLaw
- DISC-LawLLM：https://github.com/FudanDISC/DISC-LawLLM
- LawBench：https://arxiv.org/abs/2309.16289
- LegalBench-RAG：https://arxiv.org/abs/2408.10343
- Legal RAG Bench：https://arxiv.org/abs/2603.01710
- LexRAG：https://arxiv.org/abs/2502.20640
- Stanford HAI 法律 AI 幻觉研究：https://hai.stanford.edu/news/ai-trial-legal-models-hallucinate-1-out-6-or-more-benchmarking-queries
- Ragas：https://docs.ragas.io/
- vLLM：https://docs.vllm.ai/
- Hugging Face PEFT：https://huggingface.co/docs/peft/

---

## 28. 推荐执行顺序

如果从零开始，按这个顺序做：

1. 选定一个垂直领域，例如劳动法。
2. 收集 1-3 部核心法规和 20-50 篇案例或制度文档。
3. 实现结构化切分和元数据入库。
4. 做 BM25 和向量检索。
5. 加 reranker。
6. 写固定回答 prompt。
7. 加 citation verifier。
8. 做前端 demo。
9. 构建 50 条评测集。
10. 写评测报告和 bad case 分析。
11. 做 Docker 部署。
12. 打磨 README、架构图、演示视频和简历描述。

这条路线足够支撑一个算法实习项目，也能自然覆盖 RAG、LLM 应用、检索、评测、工程部署这些岗位高频关键词。

---

## 29. 20 个缺点与优化方案

本节用于把手册从“方案说明”推进到“可执行交付”。下面 20 项按优先级整理，可直接作为下一版迭代清单。

| 序号 | 当前缺点 | 优化方案 | 验收信号 |
|---:|---|---|---|
| 1 | MVP 边界仍偏宽，容易同时做劳动法、合同法、案例、制度，导致两周内无法闭环。 | 第一版只锁定一个垂直领域，例如“劳动法未签合同与辞退赔偿”，并明确不覆盖的场景。 | PRD 中有 `in_scope`、`out_of_scope`、P0/P1/P2 功能表。 |
| 2 | 用户画像有说明，但没有转成需求优先级。 | 为普通用户、企业法务、律师助理分别列出核心任务、痛点、成功标准，并映射到功能优先级。 | 每个 P0 功能都能追溯到至少一个用户故事。 |
| 3 | 数据源合规只停留在原则层面，没有授权台账。 | 新增数据源登记表，记录来源、用途、授权状态、抓取限制、可否商用、更新时间和负责人。 | 每个入库文档都有 `source_license_id` 或授权说明。 |
| 4 | 法律文本更新机制不足，无法处理法规修订、废止、替换。 | 增加版本化法律文档表，保存生效区间、废止状态、替代关系和增量更新日志。 | 查询同一法规时能返回当前有效版本，并保留历史版本追溯。 |
| 5 | 元数据缺少发布机关、效力位阶排序、适用地域粒度等关键字段。 | 扩展 chunk 元数据：`issuer`、`legal_rank`、`region_code`、`effective_from`、`effective_to`、`superseded_by`。 | 检索可按效力位阶、地区和生效时间过滤。 |
| 6 | 数据库设计没有索引、迁移和约束说明。 | 补充 Alembic 迁移、唯一约束、外键关系、常用查询索引和向量索引策略。 | 新环境能一键迁移建表，核心查询有索引命中。 |
| 7 | 文本切分策略合理，但缺少切分质量检查。 | 增加 split QA 报告：空 chunk、超长 chunk、条号缺失、标题丢失、重复 chunk、token 分布。 | 每次导入后生成 `split_report.json`，异常比例低于设定阈值。 |
| 8 | 重复、冲突、过期材料的处理规则不够明确。 | 建立去重和冲突优先级：现行有效优先，高效力位阶优先，官方来源优先，更新时间新者优先。 | 同一问题命中冲突材料时，答案能说明依据取舍。 |
| 9 | Query Processor 示例清晰，但缺少意图分类枚举和置信度阈值。 | 定义 `intent` 枚举、`confidence` 阈值、低置信度回退策略和无法分类场景。 | 低置信度问题不会直接进入确定性回答。 |
| 10 | 澄清策略有示例，但没有“何时澄清、何时先给一般规则”的判定。 | 给 missing facts 分级：阻断型事实必须追问，非阻断型事实可先给一般规则并标注限制。 | 缺少地区、时间、金额等关键事实时，系统行为稳定一致。 |
| 11 | 检索参数是经验值，没有调参方法。 | 将 BM25 top_n、vector top_n、rerank top_k、过滤条件写入配置，并用评测集做网格对比。 | 评测报告能解释当前参数为什么被选中。 |
| 12 | Reranker 没有降级和性能预算。 | 设定 rerank 超时、批处理大小、缓存策略；超时则回退到融合排序并标记降级。 | reranker 不可用时接口仍能返回保守结果。 |
| 13 | Citation Verifier 主要检查编号存在，不能证明引用支持结论。 | 增加引用片段定位、关键句匹配、NLI 或 LLM judge 校验，并采用 fail-closed 策略。 | 无依据结论会被拦截、重写或降级为“依据不足”。 |
| 14 | Prompt 缺少拒答和安全回答示例。 | 补充伪造证据、规避监管、刑事辩护策略、重大金额诉讼等场景的拒答模板。 | 高风险问题触发保守模式，有固定响应格式。 |
| 15 | API 只给正常响应，缺少错误码、澄清响应和追踪字段。 | 统一响应结构，增加 `status`、`error_code`、`clarifying_questions`、`trace_id`、`degraded`。 | 前端能区分正常回答、需要澄清、拒答、系统降级和服务错误。 |
| 16 | 前端设计偏布局说明，缺少引用核验体验。 | 右侧来源面板展示原文高亮、效力状态、发布时间、来源链接、引用支持关系和风险标识。 | 用户点击每个引用都能看到支持该结论的原文句子。 |
| 17 | 安全章节缺少 RBAC、限流和审计模型。 | 增加角色权限：普通用户、企业用户、管理员、审核员；同时加入 rate limit、操作审计和敏感操作二次确认。 | 管理后台和文档导入接口不能匿名访问。 |
| 18 | 隐私生命周期不完整，只提到脱敏和删除接口。 | 明确保留期限、加密存储、日志脱敏、用户删除请求流程、备份数据删除策略。 | 问答日志和上传文档均有 TTL 或保留策略配置。 |
| 19 | 可观测性列了日志字段，但缺少 SLO、指标名和告警阈值。 | 定义核心指标：`qa_latency_p95`、`retrieval_recall_eval`、`citation_fail_rate`、`llm_error_rate`，并设置告警。 | 指标异常时能定位到检索、rerank、生成或引用校验阶段。 |
| 20 | 测试策略没有 CI 门禁，容易只在本地手动跑。 | 增加 CI 流程：lint、unit tests、integration tests、eval smoke、schema migration check。 | PR 合并前自动阻断格式错误、核心测试失败和评测明显回退。 |

建议先落地 P0 优化：1、3、4、8、13、14、17、18。它们直接影响项目边界、法律合规、引用可信度和安全底线。随后再处理 6、7、11、15、19、20，把工程交付质量补齐。

---

## 30. 第二轮 20 个新增缺点与优化动作

第 29 节解决的是“手册从方案到交付”的第一批缺口。本节继续从真实研发、上线、运维和面试展示角度做第二轮审视，重点补足容易被忽略但会影响项目可信度的细节。

| 序号 | 新增缺点 | 优化动作 | 可交付产物 |
|---:|---|---|---|
| 1 | 缺少法律领域术语表，用户说法和法条术语之间无法稳定映射。 | 建立领域词典和同义词表，例如“赔偿金/经济补偿/二倍工资差额/违法解除赔偿金”，在 query rewrite 和 BM25 检索前统一扩展。 | `legal_terms.yaml`、同义词扩展规则、术语命中评测。 |
| 2 | 没有定义领域本体，法规、案由、争议焦点、赔偿项目之间关系不清。 | 为选定垂直领域建立轻量 ontology，描述主体、行为、争议焦点、请求权基础、证据、赔偿项。 | `domain_ontology.md`、实体关系图、样例标注数据。 |
| 3 | 多跳法律问题处理不足，例如“未签合同 + 被辞退 + 没缴社保”需要拆成多个子问题。 | 增加 query decomposition，将复杂问题拆成事实确认、法条定位、赔偿计算、证据建议等子任务，再汇总回答。 | `query_decomposer.py`、多跳问题评测集。 |
| 4 | 赔偿金额、期限、时效等计算逻辑完全依赖模型，容易算错。 | 把可公式化的部分做成确定性 calculator，例如二倍工资差额、经济补偿年限、仲裁时效提醒；LLM 只解释结果。 | `calculators/labor_compensation.py`、计算单元测试。 |
| 5 | 回答结构是文本约定，没有机器可校验的结构化 schema。 | 定义 Answer JSON Schema，包括 conclusion、basis、analysis、missing_facts、risks、citations、disclaimer。 | `schemas/answer.schema.json`、schema validation 测试。 |
| 6 | 引用只到 chunk 级，无法证明具体句子支持具体结论。 | 引入 citation span，记录 `source_id`、`start_char`、`end_char`、`supported_claim_id`，前端高亮到句子级。 | citation span 数据结构、引用高亮 demo。 |
| 7 | 缺少 claim-level 校验，答案中每个法律结论没有单独验真。 | 先抽取答案 claim，再逐条判断是否被检索材料支持；不支持的 claim 删除、降级或要求重写。 | `claim_checker.py`、unsupported claim 报告。 |
| 8 | 对话记忆策略不清晰，容易把用户早先误说的事实长期保留。 | 区分 confirmed facts、user assertions、system inferences；追问确认后才能把关键事实写入 confirmed facts。 | session memory schema、多轮对话测试。 |
| 9 | 多租户隔离没有展开，企业内部文档可能被其他用户检索到。 | 所有文档、chunk、索引、日志加 `tenant_id`；检索层强制 tenant filter，管理接口做权限校验。 | 多租户权限测试、越权检索测试。 |
| 10 | 文档导入流程没有幂等和失败重试，重复导入会污染索引。 | 用文件 hash、source_url、version 生成导入任务 ID；导入状态支持 pending/running/failed/done 和重试。 | `ingestion_jobs` 表、导入重试脚本。 |
| 11 | 没有增量索引和索引回滚方案，更新法规时容易造成线上短暂不可用。 | 建立 blue-green index：新索引构建完成并通过 smoke eval 后再切 alias；失败则保留旧索引。 | index alias 方案、索引切换 runbook。 |
| 12 | 模型供应商和本地模型耦合在一起，切换成本高。 | 抽象 LLM provider interface，统一 chat、embedding、rerank、timeout、retry、fallback 行为。 | `llm_provider.py`、provider 配置样例。 |
| 13 | Prompt 没有版本管理，优化后难以复现评测结果。 | 为 prompt 增加版本号、变更说明、适用模型、评测结果绑定；每次评测记录 prompt version。 | `prompts.yaml`、prompt changelog、eval run metadata。 |
| 14 | 没有成本预算，LLM、embedding、rerank 和存储成本不可控。 | 记录 token、请求次数、embedding 批量大小、rerank 调用量；设置日预算和超限降级策略。 | cost dashboard、token usage log、预算告警。 |
| 15 | 缺少缓存策略，重复问题会浪费检索和生成成本。 | 设计 semantic cache 和 exact cache，但缓存命中必须校验法规版本、tenant、权限和时效。 | cache key 设计、缓存命中率指标。 |
| 16 | 人工复核流程缺失，法律高风险答案没有审核闭环。 | 对高风险、低置信度、引用冲突、用户投诉问题进入 review queue，由人工标注正确答案和错误类型。 | review queue 页面、审核状态流转表。 |
| 17 | 评测集缺少红队和对抗样例，无法验证安全边界。 | 增加诱导编造法条、要求伪造证据、跨法域混淆、过期法规、无依据确定结论等 adversarial cases。 | `data/eval/red_team.jsonl`、安全评测报告。 |
| 18 | 前端没有展示“不确定性”，用户可能把一般信息当正式建议。 | 在回答顶部显示 confidence、依据充分度、缺失事实数量、风险等级；低置信度使用保守视觉状态。 | 不确定性 UI、风险等级说明。 |
| 19 | 缺少上线前发布检查，容易把未评测的索引、prompt 或模型直接上线。 | 增加 release checklist：迁移、索引 smoke、评测阈值、回滚点、配置 diff、隐私检查、负责人确认。 | `docs/release_checklist.md`、发布记录模板。 |
| 20 | 项目展示只强调功能，没有沉淀“失败案例如何驱动迭代”的故事线。 | 在 README 和答辩材料中加入 3-5 个 bad case：初始错误、根因、修复方案、修复后指标变化。 | `docs/bad_case_report.md`、面试讲解稿。 |

### 30.1 第二轮优先级

P0 先做：4、5、6、7、9、16、17、19。它们分别对应法律计算正确性、结构化可校验、引用可信、租户隔离、人工复核、安全边界和上线门禁。

P1 再做：1、3、8、10、11、13、14、20。它们会显著提高检索质量、复杂问题处理、工程稳定性和项目展示质量。

P2 后做：2、12、15、18。它们适合在 MVP 跑通后逐步产品化，不必阻塞第一版上线。

### 30.2 优化后的交付顺序

如果继续迭代这份项目，建议把下一阶段拆成三条线并行推进：

1. 可信问答线：Answer Schema、citation span、claim checker、红队评测。
2. 工程稳定线：导入幂等、blue-green index、prompt version、release checklist。
3. 产品闭环线：人工复核、bad case report、不确定性展示、成本监控。

做到这一步后，项目就不只是“法律 RAG demo”，而更像一个能被面试官认真追问架构细节、评测细节和上线风险的完整工程项目。

---

## 31. 基于网上项目调研的项目书增强版

本节根据公开开源项目、法律 RAG benchmark、商业法律 AI 产品和近期论文，对本项目书做一次“外部对标后的升级”。目标不是照搬别人的系统，而是吸收成熟项目反复强调的能力：权威来源、条款级检索、多轮咨询、可验证引用、专家评测、工作流集成和风险控制。

### 31.1 对标对象与可借鉴点

| 对标对象 | 类型 | 观察到的重点 | 对本项目的启发 |
|---|---|---|---|
| [ChatLaw](https://github.com/PKU-YuanGroup/ChatLaw) | 中文法律大模型/多 Agent | 结合 MoE、多 Agent、知识图谱和人工筛选数据来降低幻觉风险。 | 我们不必第一版训练 MoE，但应加入“法律任务路由 + 知识图谱/关系索引 + 人工复核数据闭环”。 |
| [DISC-LawLLM](https://github.com/FudanDISC/DISC-LawLLM) | 中文法律大模型系统 | 公开法律问答 SFT 数据，并强调 legal reasoning 与 verifiable retrieval。 | 我们应把 SFT 放到第二阶段，第一阶段先沉淀可验证检索日志、人工标注 QA 和 bad case。 |
| [LawBench](https://github.com/open-compass/LawBench) | 中文法律能力评测 | 面向中国法律体系评估大模型法律知识与任务能力。 | 我们的评测集应分层：法律知识、法条定位、场景适用、拒答与安全边界。 |
| [LegalBench-RAG](https://arxiv.org/abs/2408.10343) | 法律 RAG 检索基准 | 强调 minimal relevant snippets，过长上下文会增加成本、延迟和幻觉。 | 检索目标从“找相关文档”升级为“找最小充分条款片段”。 |
| [LexRAG](https://arxiv.org/abs/2502.20640) | 多轮法律咨询 RAG 基准 | 包含多轮渐进式提问、法律专家标注和 LLM-as-judge 评测。 | 我们要把项目从单轮 QA 升级为多轮咨询：历史事实管理、追问、上下文检索和多轮评测。 |
| [Legal-DC](https://arxiv.org/abs/2603.11772) | 中文法律文档 RAG 基准 | 强调 clause-boundary segmentation、条款级引用、检索和生成联合评测。 | 中文法律场景必须把“条款边界完整性”作为数据切分和评测指标。 |
| [Legal RAG Bench](https://arxiv.org/abs/2603.01710) | 端到端法律 RAG 基准 | 结论指出信息检索通常是法律 RAG 性能上限的主要驱动因素。 | 项目书应把主要创新写成“检索与引用可信度优化”，而不是“换更大模型”。 |
| [CoCounsel Legal](https://legal.thomsonreuters.com/en/products/cocounsel-legal) | 商业法律 AI | 强调接入 Westlaw、Practical Law、Microsoft 365、DMS，并输出可验证结果。 | 产品路线应增加“外部知识源 + 内部 DMS/文档库 + 工作流集成”的长期规划。 |
| [Harvey Assistant](https://www.harvey.ai/platform/assistant) | 商业法律 AI | 同时检索上传文件、DMS、高质量法律库和公开来源，并迭代生成 targeted searches。 | 我们可以加入 agentic search：先规划检索策略，再多轮改写搜索，直到证据充分。 |
| [Lexis+ AI](https://www.lexisnexis.com/community/pressroom/b/news/posts/lexisnexis-launches-lexis-ai-a-generative-ai-solution-with-hallucination-free-linked-legal-citations) | 商业法律 AI | 主打 linked legal citations、Shepard's citation validation 和权威内容。 | 引用不只要能点开，还要有“引用有效性/被废止/被引用关系”的校验思路。 |
| [Stanford 法律 AI 幻觉研究](https://arxiv.org/abs/2405.20362) | 风险研究 | 即使是 RAG 法律研究工具，仍可能出现 17%-33% 的错误或误导回答。 | 项目书不能写“消除幻觉”，应写“降低、检测、标注和人工复核幻觉风险”。 |
| [Docling](https://github.com/docling-project/docling) / [MinerU](https://mineru.net/doc/docs/index_en/) | 文档解析工具 | 面向 RAG 的结构化解析、表格、版面、OCR、多格式输出。 | 数据管道应从“读文本”升级为“版面感知解析 + 表格保真 + 解析质量报告”。 |

### 31.2 项目定位升级

原定位：

```text
面向法律咨询的 RAG 问答系统。
```

升级后定位：

```text
面向中文劳动法场景的可验证法律咨询辅助系统，通过条款级结构化索引、混合检索、句子级引用、claim-level 校验、多轮事实管理和人工复核闭环，为用户提供有依据、可追溯、可审计的法律信息参考。
```

这一定义比“法律问答 demo”更适合写进项目书，因为它明确了：

- 领域：中文劳动法，第一版聚焦未签合同、违法解除、经济补偿、社保争议。
- 方法：结构化 RAG，而不是泛化聊天。
- 可信机制：引用、校验、人工复核。
- 边界：信息参考，不替代律师意见。
- 可评测：有检索、引用、生成、安全四类指标。

### 31.3 新增项目亮点

项目书中建议把技术亮点改成以下 8 个：

| 亮点 | 说明 | 对标来源 |
|---|---|---|
| 条款级结构化索引 | 按法、章、节、条、款、项切分，保留效力层级、地区和时间。 | Legal-DC、LegalBench-RAG |
| 最小充分片段检索 | 检索目标是能支撑答案的最小条款/句子，而不是整篇文档。 | LegalBench-RAG |
| 多轮事实管理 | 区分用户陈述、已确认事实和系统推断，避免错误事实污染后续回答。 | LexRAG |
| Agentic Search | 对复杂问题先规划检索路径，多次改写关键词并比较证据充分度。 | Harvey、CoCounsel Deep Research |
| Claim-level Citation | 对答案中的每个法律结论绑定支持片段，引用能定位到句子。 | Lexis+ AI、CoCounsel |
| 法律计算器 | 将二倍工资、经济补偿、仲裁时效等可公式化内容交给确定性模块。 | 法律产品工程实践 |
| 红队评测 | 覆盖伪造证据、错误前提、过期法规、跨法域混淆和诱导确定结论。 | Stanford 幻觉研究 |
| 人工复核闭环 | 高风险、低置信度、引用冲突问题进入 review queue，反哺数据集。 | 商业法律 AI 审核实践 |

### 31.4 架构升级

建议将第 5 节总体架构升级为“证据优先”的两阶段架构：

```mermaid
flowchart TD
    Q["用户问题"] --> F["事实抽取与缺失事实判断"]
    F --> P["检索计划生成<br/>法规/案例/制度/计算器"]
    P --> KR["关键词检索 BM25"]
    P --> VR["向量检索"]
    P --> GR["结构/关系检索<br/>条款引用/修订关系/定义条款"]
    KR --> M["候选合并"]
    VR --> M
    GR --> M
    M --> RR["Rerank"]
    RR --> E["证据包 Evidence Pack"]
    E --> C["Claim Plan<br/>先列结论草案和所需依据"]
    C --> L["LLM 生成答案"]
    L --> V["Claim-level Verifier"]
    V --> H{"高风险或低置信度?"}
    H -->|是| RQ["人工复核队列"]
    H -->|否| A["返回答案<br/>引用/风险/缺失事实"]
```

其中 `Evidence Pack` 是新增核心概念，建议结构如下：

```json
{
  "query_id": "qa_20260518_001",
  "facts": {
    "confirmed": ["工作 8 个月", "未签书面劳动合同"],
    "missing": ["月工资", "入职日期", "所在城市"],
    "inferred": ["可能涉及二倍工资差额"]
  },
  "evidence": [
    {
      "source_id": "S1",
      "doc_title": "中华人民共和国劳动合同法",
      "article_no": "第八十二条",
      "span": "用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的...",
      "supports": ["claim_1"],
      "validity": "effective",
      "rank_reason": "法条直接规定未签书面劳动合同二倍工资"
    }
  ],
  "risk_flags": ["missing_salary", "arbitration_limitation_possible"]
}
```

### 31.5 数据管道升级

线上项目和文档解析工具的共同提醒是：法律 RAG 质量首先卡在数据解析和结构保持。项目书应新增如下数据管道要求：

| 环节 | 升级要求 | 验收方式 |
|---|---|---|
| 文档解析 | PDF/DOCX/HTML 分别走不同 parser，复杂 PDF 使用 Docling 或 MinerU 做版面感知解析。 | 抽样 30 页，检查标题、条号、表格、脚注、页眉页脚。 |
| 条款边界 | 切分时不得切断“第 N 条”内部逻辑，定义条款和引用条款需要建立关系。 | `clause_integrity_score >= 0.95`。 |
| 版本管理 | 法规、司法解释、地方规定要保存生效和失效时间。 | 同名法规能查询历史版本和当前版本。 |
| 关系索引 | 记录“引用、修订、废止、替代、定义、例外”关系。 | 能回答“本条被哪条修订/引用”。 |
| 解析报告 | 每次导入输出结构化质量报告。 | 空 chunk、超长 chunk、条号丢失率低于阈值。 |

新增解析质量指标：

```text
条号保留率 = 成功识别条号的条款数 / 应识别条款数
条款完整率 = 未被错误截断的条款数 / 抽样条款数
表格保真率 = 正确保留表头、行列关系的表格数 / 抽样表格数
来源可追溯率 = 有 source_url 或 source_file 的 chunk 数 / 总 chunk 数
```

### 31.6 检索策略升级

根据 Legal RAG Bench 和 LegalBench-RAG 的结论，项目书中应把检索写成主战场：

```text
第一优先级：目标法条/条款是否召回。
第二优先级：召回片段是否足够短、足够完整。
第三优先级：片段是否能直接支撑答案中的 claim。
第四优先级：LLM 表达是否自然。
```

推荐新增 `retrieval policy`：

| 问题类型 | 检索策略 |
|---|---|
| 法条定位 | BM25 优先，向量补充，必须返回条号。 |
| 场景咨询 | Query rewrite + hybrid retrieval + rerank。 |
| 多事实问题 | 先分解子问题，再分别检索，最后合并证据。 |
| 赔偿计算 | 检索法条 + 调用 calculator + 返回计算假设。 |
| 内部制度问答 | tenant filter 强制隔离，只查用户有权限的文档。 |
| 过期法规问题 | point-in-time retrieval，按问题时间查当时有效版本。 |

新增核心指标：

```text
Evidence Recall@5：标准依据是否在前 5 个证据片段中。
Minimality：返回证据是否足够短，是否避免整篇文档塞入上下文。
Clause Integrity：返回证据是否保持完整条款边界。
Citation Support Rate：每个结论是否至少有一个支持引用。
Conflict Detection Rate：冲突或过期依据是否被识别。
```

### 31.7 多轮咨询升级

LexRAG 的启发是：真实法律咨询往往不是单轮，而是用户逐步补充事实。因此项目书应新增多轮能力：

```mermaid
sequenceDiagram
    participant U as 用户
    participant S as 系统
    U->>S: 公司没签合同能赔吗？
    S->>U: 需要确认入职时间、工资、是否离职；先给一般规则。
    U->>S: 入职 8 个月，月薪 8000，还没离职。
    S->>S: 更新 confirmed facts，重新检索法条，调用计算器。
    S->>U: 给出二倍工资差额的大致区间、依据和风险。
    U->>S: 公司说试用期不用签，合法吗？
    S->>S: 带历史事实检索试用期和书面合同规则。
    S->>U: 回答并指出试用期也应签书面劳动合同。
```

多轮记忆必须遵守：

- 用户说法默认是 `user_assertion`，不是事实真相。
- 关键事实需要追问确认后进入 `confirmed_facts`。
- 每轮回答都要说明“基于目前提供的信息”。
- 如果后续事实推翻前文，系统必须更新结论并解释原因。

### 31.8 产品功能升级

商业法律 AI 的共同方向是“嵌入工作流”，因此项目书可以增加三个产品模式：

| 模式 | 面向用户 | 功能 |
|---|---|---|
| 咨询模式 | 普通用户/学生 | 多轮问答、澄清事实、法条依据、风险提示。 |
| 研究模式 | 律师助理/法务 | 多源检索、证据包、引用导出、类案摘要、研究备忘录。 |
| 审核模式 | 管理员/专家 | 查看低置信度问题、修正答案、标注错误类型、沉淀训练数据。 |

前端新增三个关键面板：

1. 证据包面板：展示命中的法规、案例、制度和每条证据支持的 claim。
2. 事实面板：展示已确认事实、缺失事实、系统推断和用户补充。
3. 风险面板：展示时效、地区、证据不足、过度确定、引用冲突等风险。

### 31.9 评测升级

项目书中的评测从“回答好不好”升级为“RAG 链路哪里错了”：

| 层级 | 评测问题 | 指标 |
|---|---|---|
| 数据层 | 文档是否解析正确，条款是否完整。 | 条号保留率、条款完整率、来源可追溯率 |
| 检索层 | 是否找到正确法条/条款。 | Recall@k、MRR、Minimality、Clause Integrity |
| 生成层 | 是否忠于证据，是否遗漏要点。 | Faithfulness、Completeness、Legal Correctness |
| 引用层 | 引用是否支持具体 claim。 | Citation Support Rate、Unsupported Claim Rate |
| 多轮层 | 是否正确利用历史事实和追问。 | Context Carryover Accuracy、Fact Update Accuracy |
| 安全层 | 是否拒绝伪造证据、错误前提和高风险建议。 | Refusal Precision、Risk Flag Recall |

新增评测集构成：

```text
50 条法条定位题
50 条单轮场景咨询题
50 条多事实组合题
30 条多轮追问题
30 条赔偿/期限计算题
30 条红队安全题
20 条无法回答/依据不足题
```

第一版可以只做 100 条，但项目书里要说明最终会扩到 260 条以上。

### 31.10 商业与合规表达升级

参考商业法律 AI 产品的表达，项目书不要写：

```text
本系统消除幻觉，提供准确法律意见。
```

应改成：

```text
本系统通过权威来源检索、条款级引用、claim-level 校验、红队评测和人工复核机制降低幻觉与误导风险。系统输出仅作为法律信息参考，不构成正式法律意见；高风险、低置信度或依据冲突场景会触发保守回答或人工复核。
```

项目书中应新增风险登记表：

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| 编造法条/案例 | 严重误导用户 | 只允许引用证据包来源，claim-level 校验，失败则拒答。 |
| 过期法规 | 输出错误依据 | 法规版本管理，按问题时间 point-in-time 检索。 |
| 用户错误前提 | 系统顺着错误前提回答 | false premise 检测，先纠正再回答。 |
| 跨法域混淆 | 不同地区规则混用 | jurisdiction filter 和回答中的适用范围提示。 |
| 隐私泄露 | 上传合同或案件材料泄露 | tenant 隔离、权限控制、脱敏、日志 TTL。 |
| 过度依赖 | 用户把系统当律师 | 显示不确定性、风险等级和人工咨询建议。 |

### 31.11 修改后的项目路线图

| 阶段 | 时间 | 目标 | 关键交付 |
|---|---:|---|---|
| P0 数据闭环 | 第 1 周 | 完成劳动法核心法规结构化和条款级索引。 | 解析报告、chunk 表、metadata schema、source ledger。 |
| P1 检索闭环 | 第 2 周 | 实现 hybrid retrieval + rerank + evidence pack。 | Recall@5 评测、证据包 JSON、检索参数报告。 |
| P2 可信生成 | 第 3 周 | 实现结构化答案、claim-level citation 和 verifier。 | Answer schema、citation span、unsupported claim report。 |
| P3 多轮咨询 | 第 4 周 | 实现事实面板、澄清问题、多轮记忆和计算器。 | session memory、labor calculator、多轮评测。 |
| P4 安全评测 | 第 5 周 | 完成红队测试、人工复核队列和风险提示。 | red_team.jsonl、review queue、风险登记表。 |
| P5 展示交付 | 第 6 周 | 完成 demo、README、评测报告和答辩材料。 | 演示视频、bad case report、项目总结。 |

### 31.12 项目书最终卖点

如果把本项目写进简历或立项书，推荐最终卖点是：

```text
本项目不是普通文档问答，而是面向中文劳动法咨询的可信法律 RAG 系统。项目围绕“证据是否找对、引用是否支撑、事实是否充分、风险是否透明”四个问题设计，采用条款级结构化索引、混合检索、最小充分片段召回、多轮事实管理、claim-level 引用校验和人工复核闭环，构建覆盖检索、生成、引用、多轮和安全的评测体系。
```

面试时可以用下面这句话收束：

```text
我没有把重点放在训练一个会背法律的大模型，而是把法律问答拆成可审计的证据工程问题：先把正确、有效、最小充分的法律依据找出来，再让模型在受约束的证据包内生成，并用 claim-level verifier 和人工复核闭环检查它有没有越界。
```
