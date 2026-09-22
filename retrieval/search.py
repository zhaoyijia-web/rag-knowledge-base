from __future__ import annotations

import json
import time
from dataclasses import dataclass

import chromadb
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from retrieval.config import (
    BM25_PATH, BM25_TOP_K, BM25_WEIGHT, CHROMA_PATH, CHUNKS_PATH, COLLECTION_NAME,
    EMBEDDING_MODEL_PATH, FINAL_TOP_K, RERANKER_MODEL_PATH, RRF_CANDIDATES, RRF_K,
    RERANK_INSTRUCTION, VECTOR_TOP_K, VECTOR_WEIGHT,
)
from retrieval.tokenizer import tokenize


@dataclass
class SearchTimings:
    vector_ms: float
    bm25_ms: float
    rerank_ms: float
    total_ms: float


class HybridRetriever:
    def __init__(self, load_reranker: bool = True):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8") if line.strip()]
        self.by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        bm25_data = json.loads(BM25_PATH.read_text(encoding="utf-8"))
        self.bm25_ids = bm25_data["chunk_ids"]
        self.bm25 = BM25Okapi(bm25_data["tokenized_corpus"])
        self.embedding_model = SentenceTransformer(str(EMBEDDING_MODEL_PATH), device=self.device)
        self.collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION_NAME)
        self.reranker = None
        if load_reranker:
            self.reranker = CrossEncoder(
                str(RERANKER_MODEL_PATH),
                device=self.device,
                max_length=1024,
                prompts={"enterprise_qa": RERANK_INSTRUCTION},
                default_prompt_name="enterprise_qa",
            )

    def _vector_search(self, query: str) -> list[dict]:
        vector = self.embedding_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        result = self.collection.query(query_embeddings=[vector.tolist()], n_results=VECTOR_TOP_K)
        return [
            {"chunk_id": chunk_id, "score": 1.0 - distance, "rank": rank}
            for rank, (chunk_id, distance) in enumerate(zip(result["ids"][0], result["distances"][0]), 1)
        ]

    def _bm25_search(self, query: str) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        indexes = np.argsort(scores)[::-1][:BM25_TOP_K]
        return [
            {"chunk_id": self.bm25_ids[index], "score": float(scores[index]), "rank": rank}
            for rank, index in enumerate(indexes, 1)
        ]

    @staticmethod
    def _rrf(vector_results: list[dict], bm25_results: list[dict]) -> list[dict]:
        fused: dict[str, dict] = {}
        for source, weight, results in (
            ("vector", VECTOR_WEIGHT, vector_results),
            ("bm25", BM25_WEIGHT, bm25_results),
        ):
            for result in results:
                item = fused.setdefault(result["chunk_id"], {"chunk_id": result["chunk_id"], "rrf_score": 0.0})
                item["rrf_score"] += weight / (RRF_K + result["rank"])
                item[f"{source}_rank"] = result["rank"]
                item[f"{source}_score"] = result["score"]
        return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)[:RRF_CANDIDATES]

    def search(self, query: str, final_top_k: int = FINAL_TOP_K) -> dict:
        total_started = time.perf_counter()
        started = time.perf_counter()
        vector_results = self._vector_search(query)
        vector_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        bm25_results = self._bm25_search(query)
        bm25_ms = (time.perf_counter() - started) * 1000
        fused = self._rrf(vector_results, bm25_results)
        if self.reranker is None:
            raise RuntimeError("当前检索器未加载 Reranker")
        started = time.perf_counter()
        scores = self.reranker.predict(
            [(query, self.by_id[item["chunk_id"]]["text"]) for item in fused],
            batch_size=2,
            show_progress_bar=False,
        )
        rerank_ms = (time.perf_counter() - started) * 1000
        for item, score in zip(fused, scores):
            item["reranker_score"] = float(score)
            item.update(self.by_id[item["chunk_id"]])
        final = sorted(fused, key=lambda item: item["reranker_score"], reverse=True)[:final_top_k]
        timings = SearchTimings(vector_ms, bm25_ms, rerank_ms, (time.perf_counter() - total_started) * 1000)
        return {
            "query": query,
            "vector_results": vector_results,
            "bm25_results": bm25_results,
            "rrf_candidates": fused,
            "results": final,
            "timings": timings.__dict__,
        }


def print_results(output: dict) -> None:
    print(f"\n问题：{output['query']}")
    for rank, item in enumerate(output["results"], 1):
        print(f"\n[{rank}] reranker={item['reranker_score']:.4f} rrf={item['rrf_score']:.6f}")
        print(f"来源：{item['source_path']}｜{item['section_path']}｜{item['chunk_id']}")
        print(item["text"][:500])
    timings = output["timings"]
    print(f"\n耗时：向量 {timings['vector_ms']:.0f}ms，BM25 {timings['bm25_ms']:.0f}ms，"
          f"重排 {timings['rerank_ms']:.0f}ms，总计 {timings['total_ms']:.0f}ms")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    args = parser.parse_args()
    print_results(HybridRetriever().search(args.query))
