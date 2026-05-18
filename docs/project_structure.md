# 项目结构规范

本项目按“可运行 MVP + 可扩展工程骨架”组织。

## 目录职责

| 路径 | 职责 |
|---|---|
| `lawrag_qa/` | 法律 RAG 核心代码，包含 API、检索、生成、导入、会话和校验。 |
| `lawrag_qa/static/` | 无构建步骤的 Web demo 静态文件。 |
| `configs/` | 应用、检索、prompt 的配置模板。当前为规范模板，运行时仍优先使用环境变量。 |
| `schemas/` | 结构化回答、未来证据包和评测结果的 JSON Schema。 |
| `data/eval/` | 可提交的小型评测集。真实或敏感语料不得提交。 |
| `docs/` | 项目书、API、数据、评测、发布和实现说明。 |
| `scripts/` | 本地质量检查脚本。 |
| `tests/` | 单元测试与轻量回归测试。 |

## 命名约定

- Python 模块使用 `snake_case`。
- API JSON 字段使用 `snake_case`。
- 文档使用小写英文文件名，例如 `api_reference.md`。
- 可提交样例数据使用 `_sample` 后缀。
- 私有语料写入 `data/lawrag_corpus.jsonl`，该文件被 `.gitignore` 忽略。

## 提交流程

1. 运行 `scripts/run_lawrag_checks.ps1`。
2. 确认没有 API key 或私有语料进入暂存区。
3. 只提交与法律 RAG 相关的文件。
4. 推送到 `codex/legal-rag-mvp` 或后续功能分支。
