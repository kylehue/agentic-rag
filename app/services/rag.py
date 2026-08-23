import asyncio
from collections.abc import Sequence

from fastapi import UploadFile

from app.core.config import settings
from app.models.llm import LLMAttachment
from app.llm.base import LLMProvider
from app.models.chunk import RetrievedChunk
from app.models.rag import RagAnswer
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage
from app.utils.string import render_template

ANSWER_PROMPT_TEMPLATE = """Answer the question using only the retrieved evidence.
If the evidence is insufficient, say so. Cite factual claims using the source
filename and evidence id in square brackets, for example [report.pdf:abc:0].

Question:
{query}

Evidence:
{evidence}
"""


class RagService:
    def __init__(
        self,
        llm: LLMProvider,
        file_storage: FileStorage,
        sql_storage: SqlStorage,
        ingestion_service: IngestionService,
        retrieval_service: RetrievalService,
    ):
        self._llm = llm
        self._file_storage = file_storage
        self._sql_storage = sql_storage
        self._ingestion_service = ingestion_service
        self._retrieval_service = retrieval_service

    async def ingest(self, file: UploadFile):
        """Store and index a file."""
        return await self._ingestion_service.ingest(file)

    async def retrieve(self, user_query: str):
        """Return evidence without asking the final-answer LLM."""
        return await self._retrieval_service.retrieve(user_query)

    async def _resolve_file_info(
        self,
        chunk: RetrievedChunk,
    ) -> tuple[str, str, str] | None:
        """
        Resolve a chunk's attachment.

        Prefer the chunk-specific file. Otherwise fall back to the original
        source file stored in the document metadata table.

        Returns:
            (file_path, file_content_type, file_filename)
        """

        metadata = chunk.metadata

        # Prefer the chunk-specific file.
        chunk_file_path = metadata.get("chunk_file_path")

        if isinstance(chunk_file_path, str) and chunk_file_path:
            file_content_type = metadata.get("chunk_file_content_type")
            file_filename = metadata.get("chunk_file_filename")

            if (
                isinstance(file_content_type, str)
                and file_content_type
                and isinstance(file_filename, str)
                and file_filename
            ):
                return (
                    chunk_file_path,
                    file_content_type,
                    file_filename,
                )

            # The current chunk metadata doesn't store these two fields,
            # so fall through to the source-file lookup below.

        # Fall back to the original source file.
        source_row = await self._sql_storage.get(
            settings.DOCUMENT_METADATA_TABLE_NAME,
            lambda table: table.c.source_id == chunk.source_id,
        )

        if not source_row:
            return None

        file_path = source_row.get("file_path")
        file_content_type = source_row.get("file_content_type")
        file_filename = source_row.get("file_filename")

        if not (
            isinstance(file_path, str)
            and file_path
            and isinstance(file_content_type, str)
            and file_content_type
            and isinstance(file_filename, str)
            and file_filename
        ):
            return None

        return (
            file_path,
            file_content_type,
            file_filename,
        )

    async def _attachment_for_chunk(
        self,
        chunk: RetrievedChunk,
    ) -> LLMAttachment | None:
        """Load a requested chunk/source file as an LLM attachment."""

        if not chunk.metadata.get("chunk_attach_file_to_llm", False):
            return None

        file_info = await self._resolve_file_info(chunk)

        if file_info is None:
            return None

        file_path, file_content_type, _file_filename = file_info

        try:
            file_bytes = await self._file_storage.read_bytes(file_path)
        except FileNotFoundError:
            return None

        return LLMAttachment(
            file_bytes,
            file_content_type,
        )

    async def _build_attachments(
        self,
        chunks: Sequence[RetrievedChunk],
    ) -> tuple[LLMAttachment, ...]:
        """Load all requested attachments concurrently."""

        tasks = [
            self._attachment_for_chunk(chunk)
            for chunk in chunks
            if chunk.metadata.get(
                "chunk_attach_file_to_llm",
                False,
            )
        ]

        if not tasks:
            return ()

        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        return tuple(result for result in results if isinstance(result, LLMAttachment))

    async def answer(
        self,
        user_query: str,
    ) -> RagAnswer:
        """Build a grounded prompt, attach requested files, and answer it."""

        chunks = await self.retrieve(user_query)

        evidence = (
            "\n\n".join(
                f"Source: {chunk.source_id}:{chunk.chunk_id}\n{chunk.text}"
                for chunk in chunks
            )
            or "(no evidence retrieved)"
        )

        prompt = render_template(
            ANSWER_PROMPT_TEMPLATE,
            {
                "query": user_query,
                "evidence": evidence,
            },
        )

        attachments = await self._build_attachments(chunks)

        answer = await self._llm.answer(
            prompt,
            attachments=attachments,
        )

        return RagAnswer(
            query=user_query,
            answer=answer,
            chunks=chunks,
        )
