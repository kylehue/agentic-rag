import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api_schemas.rag import RagAnswerRequestSchema, RagAnswerSchema
from app.container import agent_service
from app.models.rag import RagAnswer

router = APIRouter(prefix="/rag", tags=["RAG"])


def _history(request: RagAnswerRequestSchema) -> list[tuple[str, str]]:
    """The request's conversation as the (role, content) pairs the service
    and agent take."""
    return [(message.role, message.content) for message in request.history]


def _sse(event: str, data: dict) -> str:
    """One server-sent event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/answer", response_model=RagAnswerSchema)
async def answer(request: RagAnswerRequestSchema):
    result = await agent_service.answer(request.query, history=_history(request))
    return RagAnswerSchema.model_validate(result)


@router.post("/answer/stream")
async def answer_stream(request: RagAnswerRequestSchema):
    """The agent run as a server-sent event stream.

    Takes the same JSON body as /rag/answer: the question plus the prior
    conversation. Emits an `answer_delta` frame per streamed token, a
    `tool_call` frame per tool the agent requests and a `tool_result` frame
    per result it reads, then one final `answer` frame carrying the full
    RagAnswer (text plus the evidence chunks).
    """

    async def generate():
        async for event in agent_service.answer_stream(
            request.query, history=_history(request)
        ):
            if event.name == "answer" and isinstance(event.payload, RagAnswer):
                payload = RagAnswerSchema.model_validate(event.payload).model_dump(
                    mode="json"
                )
            else:
                payload = event.payload
            yield _sse(event.name, payload)

    return StreamingResponse(generate(), media_type="text/event-stream")
