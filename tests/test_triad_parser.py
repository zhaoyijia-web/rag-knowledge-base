import unittest

from eval.run_rag_triad_eval import parse_judge_output


class TriadParserTest(unittest.TestCase):
    def test_six_line_protocol(self):
        output = """CONTEXT_RELEVANCE_SCORE=0.75
CONTEXT_RELEVANCE_REASON=包含证据，但有重复内容
FAITHFULNESS_SCORE=1.00
FAITHFULNESS_REASON=所有结论均有依据
ANSWER_RELEVANCE_SCORE=0.5
ANSWER_RELEVANCE_REASON=遗漏了一个要点"""
        result = parse_judge_output(output)
        self.assertEqual(result["context_relevance"]["score"], 0.75)
        self.assertEqual(result["faithfulness"]["score"], 1.0)
        self.assertEqual(result["answer_relevance"]["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
