from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="用户问题")

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("问题不能为空")
        return value


class SourceItem(BaseModel):
    index: int
    file: str
    section: str
    chunk_id: str
    content: str
    reranker_score: float


class TimingInfo(BaseModel):
    vector_ms: float
    bm25_ms: float
    rerank_ms: float
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class RetrievalCandidate(BaseModel):
    rank: int
    chunk_id: str
    file: str
    section: str
    preview: str
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None


class RetrievalTrace(BaseModel):
    vector: list[RetrievalCandidate]
    bm25: list[RetrievalCandidate]
    rrf: list[RetrievalCandidate]
    reranked: list[RetrievalCandidate]


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    timings: TimingInfo
    trace: RetrievalTrace


class HealthResponse(BaseModel):
    status: str
    chunks: int
    collection: str
    embedding_model: str
    reranker_model: str
    generator_model: str
