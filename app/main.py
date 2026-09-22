import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.auth import router as auth_router
from app.api.chats import router as chats_router
from app.api.health import router as health_router
from app.api.rag import router as rag_router
from app.api.rag_agent import router as rag_agent_router
from app.container import lifespan
from app.core.config import settings
from app.core.exceptions import register_exception_handlers

app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)

register_exception_handlers(app)

# Sessions are cookie based, so CORS is credentialed: the request origin is
# echoed back instead of "*". The SameSite=Lax session cookie keeps
# cross-site requests from carrying it; override CORS_ORIGINS to lock it
# down further.
_cors_origins = [
    origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chats_router)
app.include_router(rag_router)
app.include_router(rag_agent_router)

logging.basicConfig(level=logging.INFO)
