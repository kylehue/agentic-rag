from collections.abc import Sequence

from app.embedders.base import Embedder
from app.llm.base import LLMProvider
from app.models.chunk import IngestedChunk, RetrievedChunk
from app.plugin.base import Plugin
from app.plugin.context import IngestionFile
from app.plugin.registry import PluginRegistry
from app.retrievers.base import Retriever
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage
from app.store_vector.base import VectorStorage


class RagService:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        embedder: Embedder,
        retriever: Retriever,
        vector_storage: VectorStorage,
        sql_storage: SqlStorage,
        file_storage: FileStorage,
        plugins: Sequence[Plugin] | None = None,
    ) -> None:
        registry = PluginRegistry()
        for plugin in (plugins if plugins is not None else []):
            registry.register(plugin)

        self._llm = llm
        self._embedder = embedder
        self._retriever = retriever
        self._vector_storage = vector_storage
        self._sql_storage = sql_storage
        self._file_storage = file_storage
        self._ingestion_service = IngestionService(
            registry=registry,
            llm=llm,
            embedder=embedder,
            vector_storage=vector_storage,
            sql_storage=sql_storage,
            file_storage=file_storage,
        )
        self._retrieval_service = RetrievalService(
            retriever=retriever,
            llm=llm,
            registry=registry,
        )

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def vector_storage(self) -> VectorStorage:
        return self._vector_storage

    @property
    def sql_storage(self) -> SqlStorage:
        return self._sql_storage

    @property
    def file_storage(self) -> FileStorage:
        return self._file_storage

    async def initialize(self) -> None:
        await self._ingestion_service.initialize()

    async def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        description: str | None = None,
        chat_id: str | None = None,
    ) -> tuple[str, list[IngestedChunk]]:
        """Store and index a file into `chat_id`; return its origin source
        id and chunks."""
        file = IngestionFile(
            filename=filename,
            content_type=content_type,
            file_bytes=file_bytes,
            description=description,
        )
        chunks = await self._ingestion_service.ingest(file, chat_id)
        return file.source_id, chunks

    async def retrieve(
        self,
        user_query: str,
        chat_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return evidence without asking the agent.

        `chat_id` bounds the corpus to that chat's chunks (mapped to the
        retrievers' `where` condition); None searches the whole store.
        """
        where = {"chat_id": chat_id} if chat_id is not None else None
        return await self._retrieval_service.retrieve(user_query, where)
