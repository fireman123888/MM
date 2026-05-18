"""In-memory hybrid retrieval for the legal RAG MVP."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

from .models import LegalChunk, RetrievedEvidence
from .text_utils import term_counts, tokenize


class InMemoryHybridRetriever:
    def __init__(self, chunks: Iterable[LegalChunk]):
        self.chunks = list(chunks)
        self.doc_tokens = [term_counts(chunk.text + " " + chunk.title + " " + chunk.article_no) for chunk in self.chunks]
        self.doc_lengths = [sum(tokens.values()) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            for token in tokens:
                self.doc_freq[token] += 1

    def search(self, query: str, top_k: int = 6, filters: dict[str, str] | None = None) -> list[RetrievedEvidence]:
        if not self.chunks:
            return []
        filters = filters or {}
        query_tokens = tokenize(query)
        bm25_scores = self._bm25(query_tokens, filters)
        cosine_scores = self._cosine(query_tokens, filters)
        fused = self._reciprocal_rank_fusion([bm25_scores, cosine_scores])
        for index, chunk in enumerate(self.chunks):
            if index in fused and self._passes_filters(chunk, filters):
                fused[index] += self._phrase_boost(query, chunk)

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
        evidence: list[RetrievedEvidence] = []
        for index, (chunk_index, score) in enumerate(ranked, start=1):
            chunk = self.chunks[chunk_index]
            reason = self._rank_reason(query_tokens, chunk)
            evidence.append(
                RetrievedEvidence(
                    source_id=f"S{index}",
                    chunk=chunk,
                    score=score,
                    rank_reason=reason,
                )
            )
        return evidence

    def _passes_filters(self, chunk: LegalChunk, filters: dict[str, str]) -> bool:
        for key, value in filters.items():
            if value in {"", None}:
                continue
            if key == "tenant_id" and chunk.tenant_id != value:
                return False
            if key == "doc_type" and chunk.doc_type != value:
                return False
            if key == "jurisdiction" and chunk.jurisdiction != value:
                return False
            if key == "status" and chunk.status != value:
                return False
        return True

    def _bm25(self, query_tokens: list[str], filters: dict[str, str]) -> dict[int, float]:
        k1 = 1.5
        b = 0.75
        scores: dict[int, float] = {}
        total_docs = max(len(self.chunks), 1)
        for index, (chunk, tokens) in enumerate(zip(self.chunks, self.doc_tokens)):
            if not self._passes_filters(chunk, filters):
                continue
            score = 0.0
            doc_len = max(self.doc_lengths[index], 1)
            for token in query_tokens:
                tf = tokens.get(token, 0)
                if tf == 0:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1 - b + b * doc_len / max(self.avg_doc_len, 1))
                score += idf * tf * (k1 + 1) / denom
            if score > 0:
                scores[index] = score
        return scores

    def _cosine(self, query_tokens: list[str], filters: dict[str, str]) -> dict[int, float]:
        query_counts = Counter(query_tokens)
        query_norm = math.sqrt(sum(value * value for value in query_counts.values())) or 1.0
        scores: dict[int, float] = {}
        for index, (chunk, tokens) in enumerate(zip(self.chunks, self.doc_tokens)):
            if not self._passes_filters(chunk, filters):
                continue
            dot = sum(query_counts[token] * tokens.get(token, 0) for token in query_counts)
            if dot == 0:
                continue
            doc_norm = math.sqrt(sum(value * value for value in tokens.values())) or 1.0
            scores[index] = dot / (query_norm * doc_norm)
        return scores

    def _reciprocal_rank_fusion(self, score_maps: list[dict[int, float]], k: int = 60) -> dict[int, float]:
        fused: dict[int, float] = defaultdict(float)
        for scores in score_maps:
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            for rank, (chunk_index, _) in enumerate(ranked, start=1):
                fused[chunk_index] += 1.0 / (k + rank)
        return dict(fused)

    def _rank_reason(self, query_tokens: list[str], chunk: LegalChunk) -> str:
        text = chunk.text + chunk.title + chunk.article_no
        overlaps = [token for token in query_tokens if token and token in text]
        shown = "、".join(sorted(set(overlaps), key=len, reverse=True)[:5])
        if shown:
            return f"命中关键词或语义片段：{shown}"
        return "混合检索排序命中"

    def _phrase_boost(self, query: str, chunk: LegalChunk) -> float:
        text = chunk.text
        boost = 0.0
        if any(term in query for term in ["未签", "没签", "未订立"]) and (
            "未与劳动者订立书面劳动合同" in text or "未同时订立书面劳动合同" in text
        ):
            boost += 0.04
        if any(term in query for term in ["赔", "二倍工资", "双倍工资"]) and "二倍的工资" in text:
            boost += 0.04
        if "试用期" in query and "试用期" in text:
            boost += 0.03
        if "社保" in query and ("社会保险" in text or "社保" in text):
            boost += 0.03
        return boost
