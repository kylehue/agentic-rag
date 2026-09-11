from fastapi import FastAPI
from app.api.agent import router as agent_router
from app.api.health import router as health_router
from app.api.rag import router as rag_router
from app.container import lifespan
from app.core.exceptions import register_exception_handlers

app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)

register_exception_handlers(app)

app.include_router(health_router)
app.include_router(rag_router)
app.include_router(agent_router)
