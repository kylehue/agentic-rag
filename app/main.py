from fastapi import FastAPI
from app.api.health import router as health_router
from app.api.documents import router as documents_router
from app.core.exceptions import register_exception_handlers

app = FastAPI(
    title="RAG API",
    version="1.0.0",
)

register_exception_handlers(app)

app.include_router(health_router)
app.include_router(documents_router)
