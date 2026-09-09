from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ValidationError

Role = Literal["system", "user", "assistant", "tool"]


class StructuredOutputError(Exception):
    """The model did not produce output matching the requested schema."""


# --- wire types ---


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the model."""

    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ToolSpec:
    """A tool offered to the model: name, description, JSON-schema arguments."""

    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class ImageContent:
    """An image part of a message content."""

    data: bytes
    mime_type: str


ContentPart = str | ImageContent


@dataclass(frozen=True)
class ChatMessage:
    """One chat message in the wire format.

    ``content`` is a plain string or a list of text and image parts
    (an image-only message is ``[ImageContent(...)]``, never a bare
    ``ImageContent``).
    """

    role: Role
    content: str | list[ContentPart] = ""
    tool_calls: tuple[ToolCall, ...] = ()  # assistant messages only
    tool_call_id: str | None = None  # tool result messages only


@dataclass(frozen=True)
class RawResult:
    """The raw outcome of one completion call."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None


@dataclass(frozen=True)
class ChatResult:
    """What a chat turn produced: text, a tool call, or both."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class LLMCapabilities:
    """What a provider's model supports.

    This app assumes full native support -- there is no silent fallback; if
    the model cannot comply, the error propagates to the caller.

    - ``tool_calling``: native function/tool calling.
    - ``structured_outputs``: native JSON-schema response format.
    - ``vision``: the model accepts image content in messages.
    """

    tool_calling: bool = True
    structured_outputs: bool = True
    vision: bool = True


def _extract_json(content: str):
    """Parse JSON from a model reply, tolerating a wrapping code fence."""
    content = content.strip()
    if content.startswith("```"):
        content = content.removeprefix("```")
        if "\n" in content:
            content = content.split("\n", 1)[1]
        if content.rstrip().endswith("```"):
            content = content.rstrip()[:-3]
    return json.loads(content)


# --- the provider ---


class LLMProvider(abc.ABC):
    """Provider-neutral LLM interface.

    Providers implement one raw primitive, `complete`, and declare
    `capabilities` (assumed fully supported). The public methods are
    concrete here: `answer` (plain text), `chat` (native tool calling),
    and `structured` (native JSON-schema output, validated with pydantic).
    Callers that need a combination none of those provide (e.g. tools and
    a JSON schema in the same call) use `complete` directly.

    Messages carry text and image parts (vision); no other binary
    attachments are ever sent to the LLM.
    """

    @property
    @abc.abstractmethod
    def capabilities(self) -> LLMCapabilities:
        """What the configured model supports natively."""

    # --- public API ---

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        json_schema: dict | None = None,
    ) -> RawResult:
        """One raw completion against the provider's API.

        The lowest-level public entry point: `answer`, `chat`, and
        `structured` are thin compositions of it, and it accepts any
        combination of `tools` and `json_schema` they do not cover.
        """

    async def answer(self, query: str) -> str:
        """Generate an answer from a text prompt."""
        result = await self.complete([ChatMessage(role="user", content=query)])
        return result.content

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> ChatResult:
        """One chat turn, using native tool calling when tools are offered."""
        if not tools:
            result = await self.complete(messages)
            return ChatResult(content=result.content)

        result = await self.complete(messages, tools=tools)
        return ChatResult(content=result.content, tool_calls=list(result.tool_calls))

    async def structured(
        self, messages: list[ChatMessage], schema: type[BaseModel]
    ) -> BaseModel:
        """One completion that must return an instance of `schema`.

        Uses the model's native JSON-schema response format and validates
        the reply with pydantic. Raises StructuredOutputError when the
        reply does not match.
        """
        result = await self.complete(messages, json_schema=schema.model_json_schema())
        try:
            return schema.model_validate(_extract_json(result.content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise StructuredOutputError(
                f"Model did not produce valid {schema.__name__}: {exc}"
            ) from exc
