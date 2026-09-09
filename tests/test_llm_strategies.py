import asyncio

import pytest
from pydantic import BaseModel

from app.llm.base import (
    ChatMessage,
    ImageContent,
    RawResult,
    StructuredOutputError,
    ToolCall,
    ToolSpec,
)
from app.llm.gemini import GeminiProvider
from app.llm.openai import OpenAIProvider

from fakes import FakeLLM

SEARCH = ToolSpec(
    name="search",
    description="Search the documents.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)


class Note(BaseModel):
    text: str
    count: int


def user(content: str) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=content)]


# --- answer (unchanged surface) ---


def test_answer_returns_the_raw_content():
    llm = FakeLLM("hello there")

    assert asyncio.run(llm.answer("what?")) == "hello there"
    assert llm.prompts == ["what?"]


# --- chat: native tool calling ---


def test_chat_without_tools_is_a_plain_completion():
    llm = FakeLLM("just text")

    result = asyncio.run(llm.chat(user("hi")))

    assert result.content == "just text"
    assert result.tool_calls == []
    assert llm.tools[0] == []


def test_chat_passes_tools_to_the_api_and_returns_the_calls():
    llm = FakeLLM(
        RawResult(
            content="",
            tool_calls=(ToolCall(id="c1", name="search", arguments={"query": "garden"}),),
        )
    )

    result = asyncio.run(llm.chat(user("find it"), tools=[SEARCH]))

    assert result.tool_calls == [
        ToolCall(id="c1", name="search", arguments={"query": "garden"})
    ]
    # Native tool calling: the specs are handed to the API itself.
    assert llm.tools[0] == [SEARCH]


def test_chat_with_tools_and_text_returns_both():
    llm = FakeLLM(
        RawResult(
            content="checking now",
            tool_calls=(ToolCall(id="c1", name="search", arguments={}),),
        )
    )

    result = asyncio.run(llm.chat(user("find it"), tools=[SEARCH]))

    assert result.content == "checking now"
    assert len(result.tool_calls) == 1


# --- structured: native JSON schema ---


def test_structured_passes_the_json_schema_to_the_api():
    llm = FakeLLM('{"text": "t", "count": 1}')

    note = asyncio.run(llm.structured(user("describe"), Note))

    assert note == Note(text="t", count=1)
    assert llm.json_schemas[0] == Note.model_json_schema()


def test_structured_strips_code_fences():
    llm = FakeLLM('```json\n{"text": "t", "count": 4}\n```')

    note = asyncio.run(llm.structured(user("describe"), Note))

    assert note == Note(text="t", count=4)


def test_structured_raises_on_invalid_json():
    llm = FakeLLM("not json at all")

    with pytest.raises(StructuredOutputError):
        asyncio.run(llm.structured(user("describe"), Note))

    # No fallback or retry: a single attempt, error propagates.
    assert len(llm.calls) == 1


def test_structured_raises_on_schema_violation():
    llm = FakeLLM('{"text": "missing count"}')

    with pytest.raises(StructuredOutputError):
        asyncio.run(llm.structured(user("describe"), Note))

    assert len(llm.calls) == 1


# --- image content (vision) ---


def test_image_content_passes_through_to_the_provider():
    image = ImageContent(data=b"png-bytes", mime_type="image/png")
    llm = FakeLLM("a picture")

    result = asyncio.run(
        llm.chat([ChatMessage(role="user", content=["what is this?", image])])
    )

    assert result.content == "a picture"
    # The base layer does not rewrite content; the provider sees it verbatim.
    assert llm.calls[0][0].content == ["what is this?", image]


def test_openai_wire_maps_image_parts_to_data_urls():
    image = ImageContent(data=b"abc", mime_type="image/png")
    message = ChatMessage(role="user", content=["caption", image])

    wire = OpenAIProvider._to_wire(message)

    assert wire["role"] == "user"
    assert wire["content"] == [
        {"type": "text", "text": "caption"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,YWJj"},
        },
    ]


def test_openai_wire_keeps_plain_text_content():
    wire = OpenAIProvider._to_wire(ChatMessage(role="user", content="hello"))

    assert wire["content"] == "hello"


def test_gemini_parts_maps_text_and_images():
    image = ImageContent(data=b"abc", mime_type="image/png")

    parts = GeminiProvider._to_parts(["caption", image])

    assert len(parts) == 2
    assert parts[0].text == "caption"
    assert parts[1].inline_data is not None
    assert parts[1].inline_data.data == b"abc"
    assert parts[1].inline_data.mime_type == "image/png"
