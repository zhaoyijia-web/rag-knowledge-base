from __future__ import annotations

import json
import time

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from retrieval.config import (
    BM25_PATH,
    CHROMA_PATH,
    CHUNKS_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_PATH,
)
from retrieval.tokenizer import tokenize


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def choose_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def chroma_metadata(chunk: dict) -> dict:
    return {key: value for key, value in chunk.items() if key != "text" and value is not None}


def build_index(batch_size: int = 16) -> None:
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError(f"没有可入库文本块：{CHUNKS_PATH}")
    if not EMBEDDING_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到 Embedding 模型：{EMBEDDING_MODEL_PATH}")

    device = choose_device()
    print(f"加载 BGE-M3，device={device}")
    model = SentenceTransformer(str(EMBEDDING_MODEL_PATH), device=device)
    started = time.perf_counter()
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embed_seconds = time.perf_counter() - started

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": "BAAI/bge-m3"},
    )
    for start in range(0, len(chunks), 100):
        batch = chunks[start:start + 100]
        collection.add(
            ids=[chunk["chunk_id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[chroma_metadata(chunk) for chunk in batch],
            embeddings=embeddings[start:start + len(batch)].tolist(),
        )

    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    bm25_data = {
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "tokenized_corpus": [tokenize(chunk["text"]) for chunk in chunks],
    }
    BM25_PATH.write_text(json.dumps(bm25_data, ensure_ascii=False), encoding="utf-8")
    print(f"完成：{len(chunks)} 块、{embeddings.shape[1]} 维，Embedding 用时 {embed_seconds:.2f}s")
    print(f"Chroma：{CHROMA_PATH}")
    print(f"BM25：{BM25_PATH}")


if __name__ == "__main__":
    build_index()

