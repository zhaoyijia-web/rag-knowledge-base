import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# retrieval 模块可能早于 FastAPI 生命周期导入，因此在读取模型路径前先加载 .env。
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"
BM25_PATH = PROJECT_ROOT / "data" / "index" / "bm25_corpus.json"
EMBEDDING_MODEL_PATH = Path(
    os.getenv("EMBEDDING_MODEL_PATH") or PROJECT_ROOT / "models" / "embedding" / "bge-m3"
).expanduser()
RERANKER_MODEL_PATH = Path(
    os.getenv("RERANKER_MODEL_PATH") or PROJECT_ROOT / "models" / "reranker" / "Qwen3-Reranker-0.6B"
).expanduser()
COLLECTION_NAME = "aozhi_knowledge_base"

VECTOR_TOP_K = 10
BM25_TOP_K = 10
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
RRF_K = 60
RRF_CANDIDATES = 10
FINAL_TOP_K = 5
RERANK_INSTRUCTION = (
    "Given a Chinese enterprise knowledge-base query, judge whether the document directly "
    "contains evidence that answers the query. Prefer exact policy clauses, parameters and answers."
)
