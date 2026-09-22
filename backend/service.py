from __future__ import annotations

import threading
import time
from pathlib import Path

from rag.deepseek_client import DeepSeekClient
from rag.generate_answer import AnswerGenerator
from retrieval.config import COLLECTION_NAME, EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH
from retrieval.search import HybridRetriever


class RAGService:
    """进程内只加载一次本地模型，并串行保护 MPS 推理。"""

    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.generator = AnswerGenerator(DeepSeekClient())
        self._inference_lock = threading.Lock()

    @property
    def chunk_count(self) -> int:
        return len(self.retriever.chunks)

    def health(self) -> dict:
        return {
            "status": "ok",
            "chunks": self.chunk_count,
            "collection": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL_PATH.name,
            "reranker_model": RERANKER_MODEL_PATH.name,
            "generator_model": self.generator.model,
        }

    def ask(self, question: str) -> dict:
        total_started = time.perf_counter()
        # PyTorch MPS 与 CrossEncoder 共享同一实例，不并行进入推理区。
        with self._inference_lock:
            retrieval = self.retriever.search(question)
            generation_started = time.perf_counter()
            answer = self.generator.generate(question, retrieval["results"])
            generation_ms = (time.perf_counter() - generation_started) * 1000

        sources = []
        for index, item in enumerate(retrieval["results"], 1):
            sources.append({
                "index": index,
                "file": Path(item["source_path"]).name,
                "section": item["section_path"],
                "chunk_id": item["chunk_id"],
                "content": item["text"],
                "reranker_score": item["reranker_score"],
            })
        vector_trace = [self._candidate(item, rank, "vector") for rank, item in enumerate(retrieval["vector_results"], 1)]
        bm25_trace = [self._candidate(item, rank, "bm25") for rank, item in enumerate(retrieval["bm25_results"], 1)]
        rrf_trace = [self._candidate(item, rank, "rrf") for rank, item in enumerate(retrieval["rrf_candidates"], 1)]
        reranked = sorted(retrieval["rrf_candidates"], key=lambda item: item["reranker_score"], reverse=True)
        reranked_trace = [self._candidate(item, rank, "reranked") for rank, item in enumerate(reranked, 1)]
        retrieval_timings = retrieval["timings"]
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "timings": {
                "vector_ms": retrieval_timings["vector_ms"],
                "bm25_ms": retrieval_timings["bm25_ms"],
                "rerank_ms": retrieval_timings["rerank_ms"],
                "retrieval_ms": retrieval_timings["total_ms"],
                "generation_ms": generation_ms,
                "total_ms": (time.perf_counter() - total_started) * 1000,
            },
            "trace": {
                "vector": vector_trace,
                "bm25": bm25_trace,
                "rrf": rrf_trace,
                "reranked": reranked_trace,
            },
        }

    def _candidate(self, item: dict, rank: int, stage: str) -> dict:
        chunk = self.retriever.by_id[item["chunk_id"]]
        text = chunk["text"]
        candidate = {
            "rank": rank,
            "chunk_id": item["chunk_id"],
            "file": Path(chunk["source_path"]).name,
            "section": chunk["section_path"],
            "preview": text[:260] + ("…" if len(text) > 260 else ""),
            "vector_rank": item.get("vector_rank"),
            "vector_score": item.get("vector_score"),
            "bm25_rank": item.get("bm25_rank"),
            "bm25_score": item.get("bm25_score"),
            "rrf_score": item.get("rrf_score"),
            "reranker_score": item.get("reranker_score"),
        }
        if stage == "vector":
            candidate["vector_rank"] = item["rank"]
            candidate["vector_score"] = item["score"]
        elif stage == "bm25":
            candidate["bm25_rank"] = item["rank"]
            candidate["bm25_score"] = item["score"]
        return candidate
