from fastapi import APIRouter, Depends

from app.api.auth import require_user
from app.api_schemas.chat import ChatCreated, ChatInfo, ChatMessage
from app.container import chat_service, rag_agent_service
from app.errors.chat import ChatForbiddenError, ChatNotFoundError

router = APIRouter(prefix="/chats", tags=["Chats"])


@router.get("")
async def list_chats(user: str = Depends(require_user)):
    """The caller's chats with their ingested sources. Pick one and send its
    `chat_id` to the RAG endpoints to continue that conversation."""
    chats = await chat_service.list_chats(user)
    return [ChatInfo(**chat) for chat in chats]


@router.post("", status_code=201)
async def create_chat(user: str = Depends(require_user)):
    """Start a new chat (the RAG endpoints also auto-create one)."""
    chat_id = await chat_service.create(user)
    return ChatCreated(chat_id=chat_id)


@router.get("/{chat_id}/messages", response_model=list[ChatMessage])
async def chat_messages(chat_id: str, user: str = Depends(require_user)):
    """The chat's conversation history, read from the agent's checkpointer
    (user and assistant turns only)."""
    owner = await chat_service.username_of(chat_id)
    if owner is None:
        raise ChatNotFoundError(chat_id)
    if owner != user:
        raise ChatForbiddenError(chat_id)
    return [
        ChatMessage(**message)
        for message in await rag_agent_service.chat_history(chat_id)
    ]
