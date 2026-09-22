import unittest

from fastapi.testclient import TestClient

from backend.app import create_app


class FakeService:
    chunk_count = 375

    def health(self):
        return {
            "status": "ok",
            "chunks": 375,
            "collection": "aozhi_knowledge_base",
            "embedding_model": "bge-m3",
            "reranker_model": "Qwen3-Reranker-0.6B",
            "generator_model": "deepseek-v4-flash",
        }

    def ask(self, question):
        return {
            "question": question,
            "answer": "奥智的使命是……[资料1]",
            "sources": [{
                "index": 1,
                "file": "员工手册.docx",
                "section": "第一章",
                "chunk_id": "employee_handbook_0005",
                "content": "第一条 奥智的使命是……",
                "reranker_score": 8.5,
            }],
            "timings": {
                "vector_ms": 10,
                "bm25_ms": 2,
                "rerank_ms": 100,
                "retrieval_ms": 112,
                "generation_ms": 300,
                "total_ms": 412,
            },
            "trace": {
                "vector": [],
                "bm25": [],
                "rrf": [],
                "reranked": [],
            },
        }


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.client_context = TestClient(create_app(FakeService))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chunks"], 375)

    def test_ask(self):
        response = self.client.post("/api/ask", json={"question": "  奥智的使命是什么？  "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["question"], "奥智的使命是什么？")
        self.assertEqual(response.json()["sources"][0]["chunk_id"], "employee_handbook_0005")
        self.assertIn("trace", response.json())

    def test_blank_question_is_rejected(self):
        response = self.client.post("/api/ask", json={"question": "   "})
        self.assertEqual(response.status_code, 422)

    def test_long_question_is_rejected(self):
        response = self.client.post("/api/ask", json={"question": "问" * 501})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
