"""Lightweight evaluation runner for retrieval quality."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .config import Settings
from .pipeline import LawRagPipeline
from .sample_data import sample_documents


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    gold_article_nos: list[str]
    gold_terms: list[str]
    category: str = "general"
    difficulty: str = "medium"


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        cases.append(
            EvalCase(
                case_id=str(payload.get("id") or payload.get("case_id")),
                question=str(payload["question"]),
                gold_article_nos=[str(item) for item in payload.get("gold_article_nos", [])],
                gold_terms=[str(item) for item in payload.get("gold_terms", [])],
                category=str(payload.get("category") or "general"),
                difficulty=str(payload.get("difficulty") or "medium"),
            )
        )
    return cases


def evaluate_retrieval(pipeline: LawRagPipeline, cases: list[EvalCase], top_k: int = 5) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for case in cases:
        evidence = pipeline.retriever.search(case.question, top_k=top_k, filters={"tenant_id": "public", "status": "effective"})
        article_nos = [item.chunk.article_no for item in evidence]
        texts = [item.chunk.text for item in evidence]
        hit_rank = first_hit_rank(case, article_nos, texts)
        rows.append(
            {
                "id": case.case_id,
                "category": case.category,
                "question": case.question,
                "hit": hit_rank is not None,
                "rank": hit_rank,
                "top_articles": article_nos,
            }
        )

    recall = sum(1 for row in rows if row["hit"]) / max(len(rows), 1)
    reciprocal_ranks = [1 / row["rank"] for row in rows if isinstance(row["rank"], int)]
    mrr = mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    by_category: dict[str, dict[str, float | int]] = {}
    for category in sorted({str(row["category"]) for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "count": len(subset),
            "recall": sum(1 for row in subset if row["hit"]) / max(len(subset), 1),
        }

    return {
        "total": len(rows),
        "top_k": top_k,
        "recall_at_k": round(recall, 4),
        "mrr": round(mrr, 4),
        "by_category": by_category,
        "cases": rows,
    }


def first_hit_rank(case: EvalCase, article_nos: list[str], texts: list[str]) -> int | None:
    for index, (article_no, text) in enumerate(zip(article_nos, texts), start=1):
        article_hit = bool(case.gold_article_nos) and article_no in case.gold_article_nos
        term_hit = bool(case.gold_terms) and all(term in text for term in case.gold_terms)
        if article_hit or term_hit:
            return index
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LawRAG retrieval on JSONL cases.")
    parser.add_argument("--cases", default="data/eval/labor_qa_sample.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    settings = Settings(use_llm=False, retrieval_top_k=args.top_k)
    pipeline = LawRagPipeline(sample_documents(), settings=settings)
    report = evaluate_retrieval(pipeline, load_eval_cases(args.cases), top_k=args.top_k)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
