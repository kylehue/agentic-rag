import base64
import json
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from app.llm.base import (
    ChatMessage,
    CompletionOptions,
    ContentPart,
    LLMProvider,
    RawDelta,
    ToolCall,
    ToolSpec,
)


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible implementation of the LLM interface."""

    def __init__(self, *, api_key: str, model: str, base_url: str):
        """Take the config this provider needs and defer client creation until
        it is needed."""
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client: AsyncOpenAI | None = None

    def _ensure_client(self) -> AsyncOpenAI:
        """Create and reuse the OpenAI client, or explain when the API key is missing."""
        if self._client is None:
            if not self._api_key:
                raise ValueError("OPENAI_API_KEY is not configured")
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    @staticmethod
    def _apply_options(kwargs: dict[str, Any], options: CompletionOptions | None) -> None:
        """Fold per-call options into the create() kwargs, skipping unset ones.

        `max_output_tokens` maps to the modern `max_completion_tokens` (works
        for reasoning and non-reasoning models alike). `reasoning` maps to
        `reasoning_effort`; "none" is omitted (there is no effort below low).
        """
        if options is None:
            return
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        if options.max_output_tokens is not None:
            kwargs["max_completion_tokens"] = options.max_output_tokens
        if options.reasoning is not None and options.reasoning != "none":
            kwargs["reasoning_effort"] = options.reasoning

    async def stream_complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        json_schema: dict | None = None,
        options: CompletionOptions | None = None,
    ) -> AsyncIterator[RawDelta]:
        client = self._ensure_client()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [self._to_wire(message) for message in messages],
            "stream": True,
        }
        if tools:
            kwargs["tools"] = [self._to_wire_tool(spec) for spec in tools]
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        self._apply_options(kwargs, options)

        # Tool calls stream in fragments: the id and name arrive once, and
        # the JSON arguments arrive in pieces, each keyed by its index.
        # Buffer the fragments and yield each call whole, at stream end.
        pending: dict[int, dict[str, str]] = {}
        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield RawDelta(text=delta.content)
            for fragment in delta.tool_calls or ():
                state = pending.setdefault(
                    fragment.index, {"id": "", "name": "", "arguments": ""}
                )
                if fragment.id:
                    state["id"] = fragment.id
                if fragment.function is not None:
                    if fragment.function.name:
                        state["name"] += fragment.function.name
                    if fragment.function.arguments:
                        state["arguments"] += fragment.function.arguments

        for state in pending.values():
            try:
                arguments = json.loads(state["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            yield RawDelta(
                tool_call=ToolCall(
                    id=state["id"],
                    name=state["name"],
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
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
