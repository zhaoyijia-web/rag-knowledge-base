import unittest

from retrieval.search import HybridRetriever


class RetrievalFusionTest(unittest.TestCase):
    def test_weighted_rrf_and_deduplication(self):
        vector = [
            {"chunk_id": "both", "rank": 1, "score": 0.9},
            {"chunk_id": "vector_only", "rank": 2, "score": 0.8},
        ]
        bm25 = [
            {"chunk_id": "both", "rank": 2, "score": 8.0},
            {"chunk_id": "bm25_only", "rank": 1, "score": 9.0},
        ]
        fused = HybridRetriever._rrf(vector, bm25)
        self.assertEqual(fused[0]["chunk_id"], "both")
        self.assertEqual(len({item["chunk_id"] for item in fused}), 3)
        self.assertEqual(fused[0]["vector_rank"], 1)
        self.assertEqual(fused[0]["bm25_rank"], 2)


if __name__ == "__main__":
    unittest.main()
