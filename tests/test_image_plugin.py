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


def test_image_plugin_produces_image_and_description_chunks():
    context = make_context(
        filename="photo.png",
        content_type="image/png",
        source_bytes=IMAGE_BYTES,
        source_description="A cat on a mat.",
    )

    chunks, _ = run_image(context, ImagePlugin(use_llm_description=False))

    assert len(chunks) == 2
    assert isinstance(chunks[0], IngestedImageChunk)
    assert chunks[0].plugin == "image"
    assert chunks[0].image.data == IMAGE_BYTES
    assert chunks[0].image.mime_type == "image/png"
    assert isinstance(chunks[1], IngestedTextChunk)
    assert chunks[1].text == "A cat on a mat."
    # Both chunks share the dedup key (the image's source id), so retrieval
    # keeps only one of them.
    assert chunks[0].key == chunks[1].key == "source-1"


def test_image_plugin_without_description_produces_only_the_image_chunk():
    context = make_context(
        filename="photo.png",
        content_type="image/png",
        source_bytes=IMAGE_BYTES,
    )

    chunks, _ = run_image(context, ImagePlugin(use_llm_description=False))

    assert len(chunks) == 1
    assert isinstance(chunks[0], IngestedImageChunk)


def test_image_plugin_llm_description_uses_the_llm_and_source_context():
    context = make_context(
        filename="photo.png",
        content_type="image/png",
        source_bytes=IMAGE_BYTES,
        source_description="A cat.",
    )
    llm = FakeLLM("The image shows a cat sitting on a mat.")

    chunks, parts = run_image(context, ImagePlugin(use_llm_description=True), llm=llm)

    assert len(chunks) == 2
    assert isinstance(chunks[1], IngestedTextChunk)
    assert chunks[1].text == "The image shows a cat sitting on a mat."
    # The LLM was called with the image and the source description as context.
    prompt_parts = parts.llm.prompts[0]
    assert any(isinstance(p, str) and "A cat." in p for p in prompt_parts)


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
