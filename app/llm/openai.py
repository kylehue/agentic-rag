import asyncio
import base64
import json
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageFunctionToolCall

from app.core.config import Settings
from app.llm.base import (
    ChatMessage,
    ContentPart,
    LLMCapabilities,
    LLMProvider,
    RawResult,
    ToolCall,
    ToolSpec,
)

FULL_CAPABILITIES = LLMCapabilities()


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible implementation of the LLM interface (text only)."""

    def __init__(self, settings: Settings | None = None):
        """Read settings and defer client creation until it is needed."""
        settings = settings or Settings()
        self._api_key = settings.OPENAI_API_KEY
        self._model = settings.OPENAI_MODEL
        self._base_url = settings.OPENAI_BASE_URL
        self._client = None

    def _ensure_client(self) -> OpenAI:
        """Create and reuse the OpenAI client, or explain when the API key is missing."""
        if self._client is None:
            if not self._api_key:
                raise ValueError("OPENAI_API_KEY is not configured")
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    @property
    def capabilities(self) -> LLMCapabilities:
        return FULL_CAPABILITIES

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        json_schema: dict | None = None,
    ) -> RawResult:
        client = self._ensure_client()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [self._to_wire(message) for message in messages],
        }
        if tools:
            kwargs["tools"] = [self._to_wire_tool(spec) for spec in tools]
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }

        response = await asyncio.to_thread(client.chat.completions.create, **kwargs)
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        for call in message.tool_calls or ():
            # tool_calls is a union that also includes the custom tool-call
            # shape, which has no .function member.
            if not isinstance(call, ChatCompletionMessageFunctionToolCall):
                continue
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )

        return RawResult(
            content=message.content or "",
            tool_calls=tuple(tool_calls),
            finish_reason=choice.finish_reason,
        )

    @staticmethod
    def _to_wire_content(
        content: str | list[ContentPart],
    ) -> str | list[dict[str, Any]]:
        """Map message content to the OpenAI wire format (text or text+image parts)."""
        if isinstance(content, str):
            return content

        parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                parts.append({"type": "text", "text": part})
            else:
                encoded = base64.b64encode(part.data).decode("ascii")
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{part.mime_type};base64,{encoded}"},
                    }
                )
        return parts

    @classmethod
    def _to_wire(cls, message: ChatMessage) -> dict[str, Any]:
        if message.role == "assistant":
            wire: dict[str, Any] = {
                "role": "assistant",
                "content": cls._to_wire_content(message.content) or None,
            }
            if message.tool_calls:
                wire["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
            return wire
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id or "",
                "content": cls._to_wire_content(message.content),
            }
        return {"role": message.role, "content": cls._to_wire_content(message.content)}

    @staticmethod
    def _to_wire_tool(spec: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
