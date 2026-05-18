"""Command-line demo for the legal RAG MVP."""

from __future__ import annotations

import argparse

from .pipeline import LawRagPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the legal RAG MVP a question.")
    parser.add_argument("query", nargs="?", default="我工作8个月没签劳动合同，月薪8000，可以赔多少？")
    args = parser.parse_args()

    pipeline = LawRagPipeline.from_sample()
    answer, pack, verification = pipeline.ask(args.query)

    print(answer.to_markdown())
    print("\n---")
    print("检索依据：")
    for item in pack.evidence:
        print(f"- [{item.source_id}] {item.chunk.title}{item.chunk.article_no} score={item.score:.4f}")
    print(f"校验：{verification}")


if __name__ == "__main__":
    main()
