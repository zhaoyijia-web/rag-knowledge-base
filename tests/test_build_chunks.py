import unittest
from collections import Counter

from ingestion.build_chunks import SOURCES, build_chunks


class BuildChunksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = build_chunks()

    def test_only_whitelisted_documents_are_included(self):
        self.assertEqual({c.document_id for c in self.chunks}, {s.document_id for s in SOURCES})

    def test_question_bank_is_one_question_per_chunk(self):
        questions = [c for c in self.chunks if c.document_id == "question_bank"]
        self.assertEqual(len(questions), 200)
        self.assertEqual(len({c.question_id for c in questions}), 200)
        counts = Counter(c.question_id.split("-")[1] for c in questions)
        self.assertEqual(counts, {"SINGLE": 50, "MULTIPLE": 50, "JUDGMENT": 50, "SHORT": 50})

    def test_equipment_appendix_pages_are_excluded(self):
        straight = "\n".join(c.text for c in self.chunks if c.document_id == "straightening_cutting_machine")
        rewinder = "\n".join(c.text for c in self.chunks if c.document_id == "pipe_rewinding_machine")
        self.assertNotIn("图1  矫直切割机", straight)
        self.assertNotIn("1.放料装置", straight)
        self.assertNotIn("图1  复绕机", rewinder)
        self.assertNotIn("图2  卷取筒", rewinder)

    def test_required_metadata_is_present(self):
        for chunk in self.chunks:
            for value in (chunk.chunk_id, chunk.document_id, chunk.title, chunk.knowledge_domain,
                          chunk.document_type, chunk.source_path, chunk.section_path, chunk.text):
                self.assertTrue(value)
            self.assertIsNone(chunk.page)


if __name__ == "__main__":
    unittest.main()
