from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import AskRequest, AskResponse, HealthResponse
from backend.service import RAGService
from rag.deepseek_client import load_dotenv

logger = logging.getLogger(__name__)


def create_app(service_factory: Callable[[], RAGService] = RAGService) -> FastAPI:
    load_dotenv()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("正在加载 BGE-M3、Qwen3 Reranker、Chroma 和 BM25")
        app.state.rag_service = await run_in_threadpool(service_factory)
        logger.info("RAG 服务加载完成，共 %s 个文本块", app.state.rag_service.chunk_count)
        yield

    application = FastAPI(
        title="奥智 RAG 知识库问答 API",
        version="0.1.0",
        description="基于 BGE-M3、BM25、Qwen3 Reranker 和 DeepSeek 的企业知识库问答服务。",
        lifespan=lifespan,
    )
    origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @application.get("/", include_in_schema=False)
    async def root() -> dict:
        return {"name": "奥智 RAG 知识库问答 API", "docs": "/docs"}

    @application.get("/api/health", response_model=HealthResponse)
    async def health(request: Request) -> dict:
        service: RAGService = request.app.state.rag_service
        return service.health()

    @application.post("/api/ask", response_model=AskResponse)
    async def ask(payload: AskRequest, request: Request) -> dict:
        service: RAGService = request.app.state.rag_service
        try:
            return await run_in_threadpool(service.ask, payload.question)
        except Exception as error:
            logger.exception("问答请求失败")
            raise HTTPException(status_code=502, detail="问答服务暂时不可用，请稍后重试") from error

    return application


app = create_app()

