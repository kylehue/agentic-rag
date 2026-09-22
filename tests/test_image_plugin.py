import asyncio

from app.models.chunk import IngestedImageChunk, IngestedTextChunk
from app.plugins.image import ImagePlugin

from fakes import FakeLLM, build_runtime, make_context

IMAGE_BYTES = b"\x89PNGfake-bytes"


def run_image(context, plugin, llm=None):
    parts = build_runtime(
        context,
        llm=llm if llm is not None else FakeLLM("A generated description."),
    )
    chunks = asyncio.run(plugin.on_ingestion_process(context, parts.runtime))
    return chunks, parts


def test_image_plugin_describes_with_the_runtime_llm_when_no_llm_is_given():
    context = make_context(
        filename="photo.png",
        content_type="image/png",
        source_bytes=IMAGE_BYTES,
        source_description="A cat.",
    )
    runtime_llm = FakeLLM("The runtime describes a cat on a mat.")

    chunks, _ = run_image(context, ImagePlugin(), llm=runtime_llm)

    assert len(chunks) == 2
    assert isinstance(chunks[0], IngestedImageChunk)
    assert chunks[0].plugin == "image"
    assert chunks[0].image.data == IMAGE_BYTES
    assert chunks[0].image.mime_type == "image/png"
    assert isinstance(chunks[1], IngestedTextChunk)
    assert chunks[1].text == "The runtime describes a cat on a mat."
    # Both chunks share the dedup key (the image's source id), so retrieval
    # keeps only one of them.
    assert chunks[0].key == chunks[1].key == "source-1"
    # The runtime LLM was the one called, with the source as context.
    assert runtime_llm.calls
    assert any(
        isinstance(part, str) and "A cat." in part for part in runtime_llm.prompts[0]
    )


def test_image_plugin_prefers_its_own_llm_over_the_runtime_llm():
    context = make_context(
        filename="photo.png",
        content_type="image/png",
        source_bytes=IMAGE_BYTES,
    )
    image_llm = FakeLLM("The image LLM describes it.")
    runtime_llm = FakeLLM("The runtime describes it.")

    chunks, _ = run_image(context, ImagePlugin(llm=image_llm), llm=runtime_llm)

    assert isinstance(chunks[1], IngestedTextChunk)
    assert chunks[1].text == "The image LLM describes it."
    assert image_llm.calls
    assert not runtime_llm.calls


def test_image_plugin_falls_back_to_source_when_the_llm_fails():
    context = make_context(
        filename="photo.png",
        content_type="image/png",
        source_bytes=IMAGE_BYTES,
        source_description="A cat on a mat.",
    )
    failing_llm = FakeLLM(RuntimeError("boom"))

    chunks, _ = run_image(context, ImagePlugin(llm=failing_llm))

    assert isinstance(chunks[1], IngestedTextChunk)
    assert chunks[1].text == "A cat on a mat."


def test_image_plugin_without_source_or_llm_text_produces_only_the_image_chunk():
    context = make_context(
        filename="photo.png",
        content_type="image/png",
        source_bytes=IMAGE_BYTES,
    )
    empty_llm = FakeLLM("")

    chunks, _ = run_image(context, ImagePlugin(llm=empty_llm))

    assert len(chunks) == 1
    assert isinstance(chunks[0], IngestedImageChunk)


def test_image_plugin_source_only_when_use_llm_description_is_false():
    context = make_context(
        filename="photo.png",
        content_type="image/png",
        source_bytes=IMAGE_BYTES,
        source_description="A cat on a mat.",
    )
    runtime_llm = FakeLLM("should not be used")

    chunks, _ = run_image(
        context, ImagePlugin(use_llm_description=False), llm=runtime_llm
    )

    assert isinstance(chunks[1], IngestedTextChunk)
    assert chunks[1].text == "A cat on a mat."
    # With LLM description disabled the LLM is never called.
    assert not runtime_llm.calls


def test_image_plugin_llm_disabled_without_source_produces_only_the_image_chunk():
    context = make_context(
        filename="photo.png",
        content_type="image/png",
        source_bytes=IMAGE_BYTES,
    )

    chunks, _ = run_image(context, ImagePlugin(use_llm_description=False))

    assert len(chunks) == 1
    assert isinstance(chunks[0], IngestedImageChunk)


def test_image_plugin_accepts_only_images():
    assert ImagePlugin().accepts(
        make_context(filename="a.png", content_type="image/png", source_bytes=b"x")
    )
    assert ImagePlugin().accepts(
        make_context(filename="a.JPEG", content_type="image/jpeg", source_bytes=b"x")
    )
    assert not ImagePlugin().accepts(
        make_context(filename="doc.txt", content_type="text/plain", source_bytes=b"x")
    )
