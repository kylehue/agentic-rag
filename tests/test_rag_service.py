import asyncio

from app.services.rag import RagService

from fakes import FakeLLM


class RecordingIngestionService:
    def __init__(self) -> None:
        self.ingested: list = []

    async def ingest(self, file) -> list:
        self.ingested.append(file)
        return [f"chunks-for-{file.filename}"]


class RecordingRetrievalService:
    async def retrieve(self, user_query) -> list:
        return []


def build_rag_service() -> tuple[RagService, RecordingIngestionService]:
    ingestion_service = RecordingIngestionService()
    service = RagService(
        llm=FakeLLM(),
        ingestion_service=ingestion_service,  # type: ignore
        retrieval_service=RecordingRetrievalService(),  # type: ignore
    )
    return service, ingestion_service


def test_ingest_builds_the_ingestion_file_from_raw_parts():
    service, ingestion_service = build_rag_service()

    chunks = asyncio.run(
        service.ingest(
            file_bytes=b"hello world",
            filename="notes.txt",
            content_type="text/plain",
            description="A test note.",
        )
    )

    assert chunks == ["chunks-for-notes.txt"]
    file = ingestion_service.ingested[0]
    assert file.filename == "notes.txt"
    assert file.content_type == "text/plain"
    assert file.file_bytes == b"hello world"
    assert file.description == "A test note."
    # Every IngestionFile gets its own auto-generated source_id.
    assert file.source_id


def test_ingest_without_description_defaults_to_none():
    service, ingestion_service = build_rag_service()

    asyncio.run(
        service.ingest(
            file_bytes=b"hello",
            filename="notes.txt",
            content_type="text/plain",
        )
    )

    assert ingestion_service.ingested[0].description is None
