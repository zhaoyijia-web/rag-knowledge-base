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
# 网页与命令行共用已审核的六文件实验索引。旧 data/chroma 已删除，不能再作为运行时数据源。
ACTIVE_INDEX_DIR = PROJECT_ROOT / "data" / "experiments" / "langchain_pdf_pilot"
CHUNKS_PATH = ACTIVE_INDEX_DIR / "chunks.jsonl"
CHROMA_PATH = ACTIVE_INDEX_DIR / "chroma"
BM25_PATH = ACTIVE_INDEX_DIR / "bm25.json"
EMBEDDING_MODEL_PATH = Path(
    os.getenv("EMBEDDING_MODEL_PATH") or PROJECT_ROOT / "models" / "embedding" / "bge-m3"
).expanduser()
RERANKER_MODEL_PATH = Path(
    os.getenv("RERANKER_MODEL_PATH") or PROJECT_ROOT / "models" / "reranker" / "Qwen3-Reranker-0.6B"
).expanduser()
COLLECTION_NAME = "enterprise_rag_pdf_pilot"

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
