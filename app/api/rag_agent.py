from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.auth import require_user
from app.api.sse import sse_frame
from app.api_schemas.rag import (
    ChatAnswerSchema,
    RagAnswerRequestSchema,
    RagAnswerSchema,
)
from app.container import chat_service, rag_agent_service
from app.models.rag import RagAnswer

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/answer", response_model=ChatAnswerSchema)
async def answer(
    request: RagAnswerRequestSchema,
    user: str = Depends(require_user),
):
    # The chat (created if not given) is both the conversation thread and
    # the retrieval scope.
    chat_id = await chat_service.get_or_create(user, request.chat_id)
    result = await rag_agent_service.ask(request.query, chat_id=chat_id)
    return ChatAnswerSchema(
        chat_id=chat_id,
        query=result.query,
        answer=result.answer,
        chunk_refs=result.chunk_refs,
    )


@router.post("/answer/stream")
async def answer_stream(
    request: RagAnswerRequestSchema,
    user: str = Depends(require_user),
):
    """The agent run as a server-sent event stream.

    Takes the same JSON body as /rag/answer. The first frame is a `chat`
    frame naming the chat (created if the request had none), followed by an
    `answer_delta` frame per streamed token, a `tool_call` frame per tool
    the agent requests, a `tool_result` frame per result it reads, and one
    final `answer` frame carrying the full answer.
    """
    chat_id = await chat_service.get_or_create(user, request.chat_id)

    async def generate():
        yield sse_frame("chat", {"chat_id": chat_id})
        async for event in rag_agent_service.ask_stream(
            request.query, chat_id=chat_id
        ):
            if event.name == "answer" and isinstance(event.payload, RagAnswer):
                payload = RagAnswerSchema.model_validate(event.payload).model_dump(
                    mode="json"
                )
                payload["chat_id"] = chat_id
            else:
                payload = event.payload
            yield sse_frame(event.name, payload)

    return StreamingResponse(generate(), media_type="text/event-stream")
