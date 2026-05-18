"""Command-line demo for the legal RAG MVP."""

from __future__ import annotations

import argparse

from .pipeline import LawRagPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the legal RAG MVP a question.")
    parser.add_argument("query", nargs="?", default="我工作8个月没签劳动合同，月薪8000，可以赔多少？")
    parser.add_argument("--corpus", help="Optional .json/.jsonl/.txt/.md file or directory to import before asking.")
    parser.add_argument("--session-id", help="Optional session id for multi-turn demos.")
    args = parser.parse_args()

    pipeline = LawRagPipeline.from_sample()
    if args.corpus:
        pipeline.add_documents_from_path(args.corpus)
    answer, pack, verification, session = pipeline.ask(args.query, session_id=args.session_id)

    print(answer.to_markdown())
    print("\n---")
    print("检索依据：")
    for item in pack.evidence:
        print(f"- [{item.source_id}] {item.chunk.title}{item.chunk.article_no} score={item.score:.4f}")
    print(f"校验：{verification}")
    if session:
        print(f"会话：{session.session_id} turns={len(session.turns)}")


if __name__ == "__main__":
    main()
