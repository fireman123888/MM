# LawRAG-QA

LawRAG-QA is a Chinese labor-law RAG MVP focused on verifiable legal information assistance. It is not a formal legal-opinion system. The first version emphasizes article-level legal text splitting, hybrid retrieval, evidence packaging, structured answers, citation checks, multi-turn fact memory, document ingestion, and a simple browser demo.

## Quick Start

Install optional API dependencies:

```powershell
pip install -r requirements-lawrag.txt
```

Run tests:

```powershell
python -m unittest discover -s tests -p "test_lawrag_qa.py"
python -m compileall -q lawrag_qa
```

Run CLI demo:

```powershell
python -m lawrag_qa.cli "我工作8个月没签劳动合同，月薪8000，可以赔多少？"
```

Run API and Web demo:

```powershell
$env:LAWRAG_LLM_BASE_URL = "http://127.0.0.1:49328/v1"
$env:LAWRAG_LLM_API_KEY = "<your-local-key>"
$env:LAWRAG_LLM_MODEL = "gpt-5.4-mini"
python -m uvicorn lawrag_qa.app:app --host 127.0.0.1 --port 8011
```

Open:

```text
http://127.0.0.1:8011/
```

## Project Layout

```text
lawrag_qa/                 Core package
  app.py                   FastAPI app and Web demo route
  pipeline.py              End-to-end RAG pipeline
  retrieval.py             In-memory hybrid retrieval
  ingestion.py             Document loaders and JSONL persistence
  sessions.py              Multi-turn session memory
  generator.py             LLM/rule-based answer generation
  verifier.py              Citation checks
configs/                   Project configuration templates
schemas/                   Machine-readable output schemas
data/eval/                 Small evaluation fixtures
docs/                      Product, API, data, evaluation, and release docs
tests/                     Unit tests
scripts/                   Local quality gates
```

## Safety Boundary

The system only provides legal information based on retrieved materials and user-provided facts. It must not fabricate statutes, cases, case numbers, courts, or source links. High-risk, low-confidence, or insufficient-evidence questions should be answered conservatively or routed to human review in future versions.

## Implemented MVP Capabilities

- Article-level legal text splitting
- In-memory hybrid retrieval
- Multi-turn session memory
- OpenAI-compatible LLM client with rule-based fallback
- Claim-level verification
- Sentence-level citation spans
- Feedback API
- In-memory review queue
- Red-team sample evaluation
- FastAPI + no-build Web demo

## Current Branch

Active GitHub branch:

```text
codex/legal-rag-mvp
```
