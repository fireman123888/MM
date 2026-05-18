import unittest

from lawrag_qa.ingestion import document_from_dict
from lawrag_qa.calculators import estimate_double_wage_gap
from lawrag_qa.config import Settings
from lawrag_qa.pipeline import LawRagPipeline
from lawrag_qa.sample_data import sample_documents
from lawrag_qa.splitter import split_documents


class LawRagQaTests(unittest.TestCase):
    def setUp(self):
        settings = Settings(use_llm=False, retrieval_top_k=4)
        self.pipeline = LawRagPipeline(sample_documents(), settings=settings)

    def test_splitter_preserves_article_numbers(self):
        chunks = split_documents(sample_documents())
        article_numbers = {chunk.article_no for chunk in chunks}
        self.assertIn("第八十二条", article_numbers)
        self.assertIn("第四十七条", article_numbers)
        self.assertTrue(all(chunk.title for chunk in chunks))

    def test_retrieval_finds_double_wage_article(self):
        evidence = self.pipeline.retriever.search("工作8个月没签劳动合同，可以要二倍工资吗", top_k=3)
        found = [item.chunk.article_no for item in evidence]
        self.assertIn("第八十二条", found)

    def test_pipeline_returns_cited_answer(self):
        answer, pack, verification, session, review = self.pipeline.ask("我工作8个月没签劳动合同，月薪8000，可以赔多少？")
        self.assertTrue(answer.citations)
        self.assertIn("第八十二条", [citation.article_no for citation in answer.citations])
        self.assertIn("劳动合同履行地或所在城市", answer.missing_facts)
        self.assertTrue(verification["ok"])
        self.assertIn("claim_checks", verification)
        self.assertIn("citation_spans", verification)
        self.assertTrue(verification["citation_spans"])
        self.assertEqual(pack.facts.intent, "compensation_consultation")
        self.assertIsNone(session)
        self.assertIsNone(review)

    def test_double_wage_calculator(self):
        estimate = estimate_double_wage_gap("工作8个月，月薪8000，没签劳动合同")
        self.assertIsNotNone(estimate)
        self.assertIn("56000", estimate)

    def test_add_documents_from_payload(self):
        document = document_from_dict(
            {
                "title": "样例员工手册",
                "text": "第九条 公司应当依法与员工签订书面劳动合同。",
                "doc_type": "policy",
                "metadata": {"tenant_id": "public"},
            }
        )
        summary = self.pipeline.add_documents([document])
        self.assertEqual(summary["documents_added"], 1)
        self.assertGreaterEqual(summary["chunks_added"], 1)
        evidence = self.pipeline.retriever.search("员工手册 书面劳动合同", top_k=3)
        self.assertTrue(any(item.chunk.title == "样例员工手册" for item in evidence))

    def test_session_tracks_multi_turn_facts(self):
        _, _, _, session, _ = self.pipeline.ask("我工作8个月没签劳动合同", session_id="case-1")
        self.assertIsNotNone(session)
        self.assertEqual(session.session_id, "case-1")
        self.assertIn("工作时长约 8 个月", session.confirmed_facts)

        _, pack, _, session, _ = self.pipeline.ask("月薪8000，能赔多少？", session_id="case-1")
        self.assertIsNotNone(session)
        self.assertIn("工作时长约 8 个月", pack.facts.confirmed)
        self.assertEqual(len(session.turns), 2)

    def test_review_queue_captures_unsupported_answer(self):
        answer, _, verification, _, _ = self.pipeline.ask("我工作8个月没签劳动合同", session_id="review-1")
        verification = {
            **verification,
            "ok": False,
            "needs_review": True,
            "risk_reasons": ["unit_test"],
        }
        review = self.pipeline.reviews.enqueue(
            query="我工作8个月没签劳动合同",
            answer_markdown=answer.to_markdown(),
            verification=verification,
            session_id="review-1",
        )
        self.assertIsNotNone(review)
        items = self.pipeline.reviews.list_reviews(status="open")
        self.assertTrue(any(item["review_id"] == review.review_id for item in items))
        resolved = self.pipeline.reviews.resolve(review.review_id, note="checked")
        self.assertEqual(resolved["status"], "resolved")


if __name__ == "__main__":
    unittest.main()
