import asyncio
from typing import Any
from uuid import uuid4

from google import genai
from google.genai import types

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
    """Gemini implementation of the LLM interface (text only)."""

    def __init__(self, settings: Settings | None = None):
        """Read settings and defer client creation until it is needed."""
        settings = settings or Settings()
        self._api_key = settings.GOOGLE_API_KEY
        self._model = settings.GEMINI_MODEL
        self._client = None

    def _ensure_client(self) -> genai.Client:
        """Create and reuse the Gemini client, or explain when the API key is missing."""
        if self._client is None:
            if not self._api_key:
                raise ValueError("GOOGLE_API_KEY is not configured")
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @property
    def capabilities(self) -> LLMCapabilities:
        return FULL_CAPABILITIES

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

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        json_schema: dict | None = None,
    ) -> RawResult:
        client = self._ensure_client()

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

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=self._model,
            contents=contents,
            config=config,
        )

        candidates = response.candidates
        if not candidates:
            raise RuntimeError("Gemini returned no candidates")
        candidate = candidates[0]

        content = ""
        tool_calls: list[ToolCall] = []
        # Both `candidate.content` and `content.parts` are optional in the SDK.
        parts: list[types.Part] | None = (
            candidate.content.parts if candidate.content is not None else None
        )
        for part in parts or ():
            if part.text:
                content += part.text
            elif part.function_call is not None and part.function_call.name:
                tool_calls.append(
                    ToolCall(
                        id=f"gemini_{uuid4().hex}",
                        name=part.function_call.name,
                        arguments=dict(part.function_call.args or {}),
                    )
                )

        finish_reason = (
            candidate.finish_reason.name if candidate.finish_reason else None
        )
        return RawResult(
            content=content,
            tool_calls=tuple(tool_calls),
            finish_reason=finish_reason,
        )
