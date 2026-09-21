import asyncio
import base64

import pytest
from google.genai import types
from pydantic import BaseModel

from app.llm.base import (
    ChatMessage,
    CompletionOptions,
    ImageContent,
    RawDelta,
    RawResult,
    StructuredOutputError,
    ToolCall,
    ToolSpec,
)
from app.llm.gemini import GeminiProvider, _thought_signature
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


# --- complete: the accumulated projection of stream_complete ---


def test_complete_accumulates_a_streamed_result():
    class SplitLLM(FakeLLM):
        async def stream_complete(
            self, messages, *, tools=None, json_schema=None, options=None
        ):
            self.calls.append(messages)
            self.tools.append(tools or [])
            for piece in ("he", "llo"):
                yield RawDelta(text=piece)
            yield RawDelta(
                tool_call=ToolCall(
                    id="c1", name="search", arguments={"query": "garden"}
                )
            )

    llm = SplitLLM()

    result = asyncio.run(llm.complete(user("hi"), tools=[SEARCH]))

    # complete is the stream, accumulated: text joined, tool calls kept.
    assert result.content == "hello"
    assert result.tool_calls == (
        ToolCall(id="c1", name="search", arguments={"query": "garden"}),
    )
    assert llm.tools[0] == [SEARCH]


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


# --- completion options (per-call generation parameters) ---


def test_options_flow_through_to_stream_complete():
    options = CompletionOptions(temperature=0.3, reasoning="low")
    llm = FakeLLM("hi")

    asyncio.run(llm.answer("q", options=options))

    assert llm.options[0] is options


def test_options_default_to_none():
    llm = FakeLLM("hi")

    asyncio.run(llm.answer("q"))

    assert llm.options[0] is None


def test_openai_options_map_to_native_params():
    kwargs: dict = {}
    options = CompletionOptions(
        temperature=0.2, max_output_tokens=128, reasoning="high"
    )

    OpenAIProvider._apply_options(kwargs, options)

    assert kwargs == {
        "temperature": 0.2,
        "max_completion_tokens": 128,
        "reasoning_effort": "high",
    }


def test_openai_options_none_is_a_noop():
    kwargs = {"model": "x"}

    OpenAIProvider._apply_options(kwargs, None)

    assert kwargs == {"model": "x"}


def test_openai_reasoning_none_omits_effort():
    kwargs: dict = {}

    OpenAIProvider._apply_options(kwargs, CompletionOptions(reasoning="none", temperature=0.5))

    assert kwargs == {"temperature": 0.5}


def test_gemini_options_map_to_native_params():
    config = types.GenerateContentConfig()
    options = CompletionOptions(
        temperature=0.7, max_output_tokens=256, reasoning="medium"
    )

    GeminiProvider._apply_options(config, options)

    assert config.temperature == 0.7
    assert config.max_output_tokens == 256
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM


def test_gemini_options_none_is_a_noop():
    config = types.GenerateContentConfig()

    GeminiProvider._apply_options(config, None)

    assert config.temperature is None
    assert config.max_output_tokens is None
    assert config.thinking_config is None


def test_gemini_reasoning_none_omits_thinking():
    config = types.GenerateContentConfig()

    GeminiProvider._apply_options(config, CompletionOptions(reasoning="none", temperature=0.1))

    assert config.temperature == 0.1
    assert config.thinking_config is None


# --- thought_signature round-trip (Gemini thinking + tools) ---


def test_gemini_thought_signature_normalizes_to_base64():
    encoded = base64.b64encode(b"sig-bytes").decode()
    # The SDK keeps the signature as bytes; the helper returns base64.
    part = types.Part(
        function_call=types.FunctionCall(name="n", args={}), thought_signature=encoded
    )

    assert _thought_signature(part) == encoded
    assert _thought_signature(types.Part.from_function_call(name="n", args={})) is None


def test_gemini_prepare_echoes_thought_signature():
    sig = base64.b64encode(b"thought").decode()
    messages = [
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=(
                ToolCall(
                    id="c1",
                    name="search",
                    arguments={"q": "x"},
                    provider_data={"thought_signature": sig},
                ),
            ),
        ),
    ]

    contents, _ = GeminiProvider(api_key="test", model="test")._prepare(
        messages, None, None, None
    )

    part = contents[0].parts[0]
    assert part.function_call.name == "search"
    # Echoed back on the same function-call part (the SDK stores it as bytes).
    assert part.thought_signature == b"thought"


def test_gemini_prepare_omits_signature_when_absent():
    messages = [
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=(ToolCall(id="c1", name="search", arguments={"q": "x"}),),
        ),
    ]

    contents, _ = GeminiProvider(api_key="test", model="test")._prepare(
        messages, None, None, None
    )

    part = contents[0].parts[0]
    assert part.function_call.name == "search"
    assert part.thought_signature is None
