# LawRAG-QA MVP 实现说明

本仓库已新增 `lawrag_qa/`，用于落地法律问答项目书中的第一版 MVP。

## 当前能力

- 法律法规条级切分：保留文档标题、条号、来源、效力状态和租户字段。
- 本地混合检索：BM25 + 轻量字符/词项向量相似度 + RRF 融合。
- 证据包：返回 `EvidencePack`，包含识别事实、缺失事实、风险标记和检索依据。
- 结构化回答：结论、法律依据、适用分析、缺失事实、风险提示、免责声明。
- 引用校验：检查引用编号是否存在，并对法律依据做轻量支持性检查。
- 确定性计算器：对未签劳动合同二倍工资差额做演示级估算。
- FastAPI 接口：`/health`、`/api/v1/search`、`/api/v1/qa`。

## LLM 接口

默认使用 OpenAI-compatible 接口：

```text
http://127.0.0.1:49328/v1
```

API key 不写入仓库，请在本地环境变量设置：

```powershell
$env:LAWRAG_LLM_BASE_URL = "http://127.0.0.1:49328/v1"
$env:LAWRAG_LLM_API_KEY = "<your-api-key>"
$env:LAWRAG_LLM_MODEL = "gpt-5.4-mini"
```

如果接口不可用，系统会降级为规则生成，方便本地测试和开发。

## 本地命令

```powershell
python -m lawrag_qa.cli "我工作8个月没签劳动合同，月薪8000，可以赔多少？"
```

运行 API：

```powershell
python -m uvicorn lawrag_qa.app:app --host 127.0.0.1 --port 8000
```

测试：

```powershell
python -m unittest discover -s tests -p "test_lawrag_qa.py"
```

## 下一步

1. 替换样例法规为授权法律数据源。
2. 接入真实 embedding/reranker。
3. 增加 Answer JSON Schema、citation span 和 claim-level verifier。
4. 增加红队评测集和人工复核队列。
