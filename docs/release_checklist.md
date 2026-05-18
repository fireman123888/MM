# Release Checklist

发布前按顺序检查。

## 代码

- [ ] `python -m unittest discover -s tests -p "test_lawrag_qa.py"` 通过。
- [ ] `python -m compileall -q lawrag_qa` 通过。
- [ ] `python -m lawrag_qa.evaluation --cases data/eval/labor_qa_sample.jsonl --red-team-cases data/eval/red_team_sample.jsonl --top-k 5` 通过。
- [ ] `powershell -ExecutionPolicy Bypass -File scripts\run_lawrag_checks.ps1` 通过。
- [ ] 没有提交 `.env`、私有语料、API key、用户材料。

## API

- [ ] `GET /health` 返回 `status=ok`。
- [ ] `POST /api/v1/qa` 返回 `answer`、`citations`、`facts`、`verification`。
- [ ] `POST /api/v1/search` 能召回第八十二条。
- [ ] `POST /api/v1/documents/import` 可导入样例文档。
- [ ] `POST /api/v1/feedback` 可记录用户反馈。
- [ ] `GET /api/v1/qa-logs` 可查看最近问答日志。
- [ ] `GET /api/v1/review-queue` 可查看复核队列。
- [ ] `GET /` Web demo 可打开。

## 数据

- [ ] 新增语料有来源说明。
- [ ] 私有语料没有进入 Git。
- [ ] 文档导入后 chunk 数符合预期。
- [ ] 过期、冲突或低可信来源有风险标记。

## 法律安全

- [ ] 回答包含免责声明。
- [ ] 事实不足时提示需要补充信息。
- [ ] 不承诺诉讼结果。
- [ ] 不编造法条、案例、法院或案号。
- [ ] `verification.claim_checks` 和 `verification.citation_spans` 正常返回。

## 运行态存储

- [ ] `LAWRAG_DB_PATH` 指向本地 SQLite 文件。
- [ ] 多轮 session 重启后仍可读取。
- [ ] feedback、review queue、qa logs 重启后仍可读取。
- [ ] `data/*.sqlite3` 未进入 Git。

## 回滚

- [ ] 记录发布提交 hash。
- [ ] 保留上一版分支或 tag。
- [ ] 如新版本失败，回退到上一提交并重启服务。
