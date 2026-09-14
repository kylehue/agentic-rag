from fastapi import FastAPI
from app.api.auth import router as auth_router
from app.api.chats import router as chats_router
from app.api.health import router as health_router
from app.api.rag import router as rag_router
from app.api.rag_agent import router as rag_agent_router
from app.container import lifespan
from app.core.exceptions import register_exception_handlers

app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)

register_exception_handlers(app)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chats_router)
app.include_router(rag_router)
app.include_router(rag_agent_router)
