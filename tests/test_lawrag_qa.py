import unittest

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
        answer, pack, verification = self.pipeline.ask("我工作8个月没签劳动合同，月薪8000，可以赔多少？")
        self.assertTrue(answer.citations)
        self.assertIn("第八十二条", [citation.article_no for citation in answer.citations])
        self.assertIn("劳动合同履行地或所在城市", answer.missing_facts)
        self.assertTrue(verification["ok"])
        self.assertEqual(pack.facts.intent, "compensation_consultation")

    def test_double_wage_calculator(self):
        estimate = estimate_double_wage_gap("工作8个月，月薪8000，没签劳动合同")
        self.assertIsNotNone(estimate)
        self.assertIn("56000", estimate)


if __name__ == "__main__":
    unittest.main()
