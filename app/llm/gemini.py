from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from google import genai
from google.genai import types

from app.core.config import Settings
from app.llm.base import (
    ChatMessage,
    ContentPart,
    LLMProvider,
    RawDelta,
    ToolCall,
    ToolSpec,
)

_JSON_TYPE_MAP = {
    "string": types.Type.STRING,
    "number": types.Type.NUMBER,
    "integer": types.Type.INTEGER,
    "boolean": types.Type.BOOLEAN,
    "array": types.Type.ARRAY,
    "object": types.Type.OBJECT,
}


def _text_parts(content: str | list[ContentPart]) -> list[str]:
    """The text of a message content (system instructions are text only)."""
    if isinstance(content, str):
        return [content]
    return [part for part in content if isinstance(part, str)]


def _to_genai_schema(node: dict[str, Any]) -> types.Schema:
    """Convert a JSON schema node into the genai SDK's Schema type."""
    node_type = node.get("type", "string")
    kwargs: dict[str, Any] = {"type": _JSON_TYPE_MAP.get(node_type, types.Type.STRING)}

    if node.get("description"):
        kwargs["description"] = node["description"]

    if node_type == "object":
        properties = node.get("properties") or {}
        kwargs["properties"] = {
            name: _to_genai_schema(prop) for name, prop in properties.items()
        }
        if node.get("required"):
            kwargs["required"] = node["required"]

    if node_type == "array" and "items" in node:
        kwargs["items"] = _to_genai_schema(node["items"])

    return types.Schema(**kwargs)


class GeminiProvider(LLMProvider):
    """Gemini implementation of the LLM interface."""

    def __init__(self, settings: Settings | None = None):
        """Read settings and defer client creation until it is needed."""
        settings = settings or Settings()
        self._api_key = settings.GOOGLE_API_KEY
        self._model = settings.GEMINI_MODEL
        self._client: genai.Client | None = None

    def _ensure_client(self) -> genai.Client:
        """Create and reuse the Gemini client, or explain when the API key is missing."""
        if self._client is None:
            if not self._api_key:
                raise ValueError("GOOGLE_API_KEY is not configured")
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @staticmethod
    def _to_parts(content: str | list[ContentPart]) -> list[types.Part]:
        """Map message content to Gemini parts (text and inline images)."""
        if isinstance(content, str):
            content = [content]

        parts: list[types.Part] = []
        for part in content:
            if isinstance(part, str):
                parts.append(types.Part.from_text(text=part))
            else:
                parts.append(
                    types.Part.from_bytes(data=part.data, mime_type=part.mime_type)
                )
        return parts

    def _prepare(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None,
        json_schema: dict | None,
    ) -> tuple[list[types.ContentUnionDict], types.GenerateContentConfig]:
        """The Gemini contents and config for one completion."""
        # Gemini pairs function responses by name, not call id: recover the
        # names of any tool calls we are responding to.
        call_names = {
            call.id: call.name for message in messages for call in message.tool_calls
        }

        config = types.GenerateContentConfig()
        system = "\n".join(
            text
            for message in messages
            if message.role == "system"
            for text in _text_parts(message.content)
        )
        if system:
            config.system_instruction = system
        if tools:
            config.tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=spec.name,
                            description=spec.description,
                            parameters=_to_genai_schema(spec.parameters),
                        )
                        for spec in tools
                    ]
                )
            ]
        if json_schema is not None:
            config.response_schema = _to_genai_schema(json_schema)

        # The SDK's contents parameter is invariant in its element type;
        # ContentUnionDict is the exact element union it declares.
        contents: list[types.ContentUnionDict] = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=self._to_parts(message.content),
                    )
                )
            elif message.role == "assistant":
                assistant_parts: list[types.Part] = []
                if message.content:
                    assistant_parts.extend(self._to_parts(message.content))
                for call in message.tool_calls:
                    assistant_parts.append(
                        types.Part.from_function_call(
                            name=call.name, args=call.arguments
                        )
                    )
                contents.append(types.Content(role="model", parts=assistant_parts))
            elif message.role == "tool":
                name = call_names.get(message.tool_call_id or "", "tool")
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=name,
                                response={"result": message.content},
                            )
                        ],
                    )
                )

        return contents, config

    async def stream_complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        json_schema: dict | None = None,
    ) -> AsyncIterator[RawDelta]:
        client = self._ensure_client()
        contents, config = self._prepare(messages, tools, json_schema)

        saw_candidate = False
        # The aio method is a coroutine that returns an async iterator.
        stream = await client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        )
        async for chunk in stream:
            candidates = chunk.candidates
            if not candidates:
                continue
            saw_candidate = True
            candidate = candidates[0]
            # Both `candidate.content` and `content.parts` are optional in
            # the SDK.
            if candidate.content is None:
                continue
            for part in candidate.content.parts or ():
                if part.text:
                    yield RawDelta(text=part.text)
                elif part.function_call is not None and part.function_call.name:
                    yield RawDelta(
                        tool_call=ToolCall(
                            id=f"gemini_{uuid4().hex}",
                            name=part.function_call.name,
                            arguments=dict(part.function_call.args or {}),
                        )
                    )
        if not saw_candidate:
            raise RuntimeError("Gemini returned no candidates")
