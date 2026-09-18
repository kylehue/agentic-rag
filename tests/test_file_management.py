import asyncio
from io import BytesIO
from pathlib import Path

from app.database import CHUNK_TABLE_NAME, DOCUMENT_METADATA_TABLE_NAME
from app.plugin.registry import PluginRegistry
from app.retrievers.base import Retriever
from app.services.ingestion import IngestionService
from app.store_file.local import LocalFileStorage
from app.store_sql.local import LocalSqlStorage

from fakes import FakeEmbedder, FakeLLM, FakeVectorStorage, build_rag_service

DOCUMENTS = DOCUMENT_METADATA_TABLE_NAME
CHUNKS = CHUNK_TABLE_NAME


class NoopRetriever(Retriever):
    async def retrieve(self, user_query, where=None):
        return []


def build_service(tmp_path):
    sql_storage = LocalSqlStorage(storage_dir=tmp_path / "sql")
    file_storage = LocalFileStorage(storage_dir=tmp_path / "file")
    vector_storage = FakeVectorStorage()
    # No plugins are registered: list/delete operate on stored rows, not on
    # ingestion, so the registry is unused here.
    service = IngestionService(
        registry=PluginRegistry(),
        llm=FakeLLM(),
        embedder=FakeEmbedder(),
        vector_storage=vector_storage,
        sql_storage=sql_storage,
        file_storage=file_storage,
    )
    return service, sql_storage, file_storage, vector_storage


async def _upload(file_storage, name, content_type, data):
    return await file_storage.upload(
        file=BytesIO(data),
        file_filename=name,
        file_content_type=content_type,
        file_dir="documents/",
    )


async def seed_tree(sql_storage, file_storage, vector_storage):
    """Two origin files: origin-1 (chat-1) emits a table (child-1), and
    origin-2 (chat-2) stands alone. Returns the three stored file paths."""
    path_origin1 = await _upload(file_storage, "o1.txt", "text/plain", b"report body")
    path_child1 = await _upload(
        file_storage, "c1.csv", "text/csv", b"region,amount\nnorth,10\n"
    )
    path_origin2 = await _upload(file_storage, "o2.txt", "text/plain", b"other body")

    await sql_storage.upsert(
        DOCUMENTS,
        [
            {
                "source_id": "origin-1",
                "file_path": path_origin1,
                "file_content_type": "text/plain",
                "file_filename": "o1.txt",
                "file_orig_filename": "report.txt",
                "is_origin": True,
                "chat_id": "chat-1",
            },
            {
                "source_id": "child-1",
                "file_path": path_child1,
                "file_content_type": "text/csv",
                "file_filename": "c1.csv",
                "file_orig_filename": "report_table_1.csv",
                "is_origin": False,
                "chat_id": "chat-1",
            },
            {
                "source_id": "origin-2",
                "file_path": path_origin2,
                "file_content_type": "text/plain",
                "file_filename": "o2.txt",
                "file_orig_filename": "other.txt",
                "is_origin": True,
                "chat_id": "chat-2",
            },
        ],
        ["source_id"],
    )
    await sql_storage.upsert(
        CHUNKS,
        [
            {
                "chunk_id": "c1",
                "source_id": "origin-1",
                "parent_source_id": None,
                "origin_source_id": "origin-1",
                "plugin": "text",
                "text": "report body",
                "metadata": {},
                "chat_id": "chat-1",
            },
            {
                "chunk_id": "c2",
                "source_id": "child-1",
                "parent_source_id": "origin-1",
                "origin_source_id": "origin-1",
                "plugin": "table",
                "text": "table",
                "metadata": {},
                "chat_id": "chat-1",
            },
            {
                "chunk_id": "c3",
                "source_id": "origin-2",
                "parent_source_id": None,
                "origin_source_id": "origin-2",
                "plugin": "text",
                "text": "other body",
                "metadata": {},
                "chat_id": "chat-2",
            },
        ],
        ["chunk_id"],
    )
    await vector_storage.add(
        ["c1", "c2", "c3"],
        [[0.1]] * 3,
        [{"chat_id": "chat-1"}, {"chat_id": "chat-1"}, {"chat_id": "chat-2"}],
    )
    return path_origin1, path_child1, path_origin2


def _row_ids(rows, key):
    return {row[key] for row in rows}


# --- list_files ---


def test_list_files_returns_only_origins_in_the_chat(tmp_path):
    service, sql_storage, file_storage, _ = build_service(tmp_path)

    async def flow():
        await sql_storage.create_tables()
        await seed_tree(sql_storage, file_storage, FakeVectorStorage())
        result = await service.list_files(chat_id="chat-1")
        await sql_storage.close()
        return result

    result = asyncio.run(flow())

    # Only the origin upload in chat-1: not the emitted child, not chat-2's file.
    assert [doc["source_id"] for doc in result] == ["origin-1"]
    assert result[0]["file_orig_filename"] == "report.txt"


def test_list_files_without_chat_returns_all_origins(tmp_path):
    service, sql_storage, file_storage, _ = build_service(tmp_path)

    async def flow():
        await sql_storage.create_tables()
        await seed_tree(sql_storage, file_storage, FakeVectorStorage())
        result = await service.list_files()
        await sql_storage.close()
        return result

    result = asyncio.run(flow())

    assert _row_ids(result, "source_id") == {"origin-1", "origin-2"}


# --- list_file_chunks ---


def test_list_file_chunks_includes_emitted_descendants(tmp_path):
    service, sql_storage, file_storage, _ = build_service(tmp_path)

    async def flow():
        await sql_storage.create_tables()
        await seed_tree(sql_storage, file_storage, FakeVectorStorage())
        result = await service.list_file_chunks("origin-1", chat_id="chat-1")
        await sql_storage.close()
        return result

    result = asyncio.run(flow())

    # The origin's own chunk (c1) and its emitted table's chunk (c2).
    assert _row_ids(result, "chunk_id") == {"c1", "c2"}


def test_list_file_chunks_scoped_to_the_chat(tmp_path):
    service, sql_storage, file_storage, _ = build_service(tmp_path)

    async def flow():
        await sql_storage.create_tables()
        await seed_tree(sql_storage, file_storage, FakeVectorStorage())
        result = await service.list_file_chunks("origin-1", chat_id="chat-2")
        await sql_storage.close()
        return result

    result = asyncio.run(flow())

    # origin-1's chunks live in chat-1, so chat-2 sees none.
    assert result == []


# --- delete_file ---


def test_delete_file_removes_the_whole_emission_tree(tmp_path):
    service, sql_storage, file_storage, vector_storage = build_service(tmp_path)

    async def flow():
        await sql_storage.create_tables()
        path_origin1, path_child1, path_origin2 = await seed_tree(
            sql_storage, file_storage, vector_storage
        )
        deleted = await service.delete_file("origin-1", chat_id="chat-1")
        chunk_rows = await sql_storage.get_all(CHUNKS)
        doc_rows = await sql_storage.get_all(DOCUMENTS)
        await sql_storage.close()
        return deleted, chunk_rows, doc_rows, (
            path_origin1,
            path_child1,
            path_origin2,
        )

    deleted, chunk_rows, doc_rows, (
        path_origin1,
        path_child1,
        path_origin2,
    ) = asyncio.run(flow())

    # Two files removed: the origin and its emitted child.
    assert deleted == 2
    # Only chat-2's chunk and document remain.
    assert _row_ids(chunk_rows, "chunk_id") == {"c3"}
    assert _row_ids(doc_rows, "source_id") == {"origin-2"}
    # The origin and child bytes are gone from disk; chat-2's file is intact.
    assert not Path(path_origin1).exists()
    assert not Path(path_child1).exists()
    assert Path(path_origin2).exists()
    # The tree's vectors were deleted.
    assert sorted(vector_storage.deleted[0]) == ["c1", "c2"]


def test_delete_file_is_scoped_to_the_chat(tmp_path):
    service, sql_storage, file_storage, vector_storage = build_service(tmp_path)

    async def flow():
        await sql_storage.create_tables()
        await seed_tree(sql_storage, file_storage, vector_storage)
        deleted = await service.delete_file("origin-1", chat_id="chat-2")
        chunk_rows = await sql_storage.get_all(CHUNKS)
        await sql_storage.close()
        return deleted, chunk_rows

    deleted, chunk_rows = asyncio.run(flow())

    # origin-1 is in chat-1, so deleting it through chat-2 removes nothing.
    assert deleted == 0
    assert _row_ids(chunk_rows, "chunk_id") == {"c1", "c2", "c3"}
    assert vector_storage.deleted == []


def test_delete_file_not_found_returns_zero(tmp_path):
    service, sql_storage, file_storage, _ = build_service(tmp_path)

    async def flow():
        await sql_storage.create_tables()
        await seed_tree(sql_storage, file_storage, FakeVectorStorage())
        deleted = await service.delete_file("nonexistent", chat_id="chat-1")
        await sql_storage.close()
        return deleted

    assert asyncio.run(flow()) == 0


# --- delete_chat ---


def test_delete_chat_removes_everything_in_the_chat(tmp_path):
    service, sql_storage, file_storage, vector_storage = build_service(tmp_path)

    async def flow():
        await sql_storage.create_tables()
        path_origin1, path_child1, path_origin2 = await seed_tree(
            sql_storage, file_storage, vector_storage
        )
        await service.delete_chat("chat-1")
        chunk_rows = await sql_storage.get_all(CHUNKS)
        doc_rows = await sql_storage.get_all(DOCUMENTS)
        await sql_storage.close()
        return chunk_rows, doc_rows, (path_origin1, path_child1, path_origin2)

    chunk_rows, doc_rows, (
        path_origin1,
        path_child1,
        path_origin2,
    ) = asyncio.run(flow())

    # chat-1's chunks and documents are gone; chat-2's remain.
    assert _row_ids(chunk_rows, "chunk_id") == {"c3"}
    assert _row_ids(doc_rows, "source_id") == {"origin-2"}
    # chat-1's file bytes removed (origin + emitted child), chat-2's intact.
    assert not Path(path_origin1).exists()
    assert not Path(path_child1).exists()
    assert Path(path_origin2).exists()
    # chat-1's vectors deleted.
    assert sorted(vector_storage.deleted[0]) == ["c1", "c2"]


# --- file metadata and links ---


def test_get_file_metadata_returns_the_document(tmp_path):
    service, sql_storage, file_storage, _ = build_service(tmp_path)

    async def flow():
        await sql_storage.create_tables()
        await seed_tree(sql_storage, file_storage, FakeVectorStorage())
        found = await service.get_file_metadata("origin-1")
        missing = await service.get_file_metadata("nope")
        await sql_storage.close()
        return found, missing

    found, missing = asyncio.run(flow())
    assert found is not None
    assert found["source_id"] == "origin-1"
    assert found["file_orig_filename"] == "report.txt"
    assert found["is_origin"] is True
    assert found["chat_id"] == "chat-1"
    assert missing is None


def test_get_file_link_returns_a_serveable_link(tmp_path):
    service, sql_storage, file_storage, _ = build_service(tmp_path)

    async def flow():
        await sql_storage.create_tables()
        await seed_tree(sql_storage, file_storage, FakeVectorStorage())
        link = await service.get_file_link("origin-1")
        missing = await service.get_file_link("nope")
        await sql_storage.close()
        return link, missing

    link, missing = asyncio.run(flow())
    assert link is not None and link.startswith("/files/")
    assert missing is None


def test_local_file_storage_create_link_is_by_stored_name(tmp_path):
    storage = LocalFileStorage(storage_dir=tmp_path / "file")
    assert storage.create_link(f"{tmp_path}/file/documents/abc123.pdf") == "/files/abc123.pdf"


# --- RagService delegation ---


def test_rag_service_delegates_file_management(tmp_path):
    from app.services.rag import RagService

    sql_storage = LocalSqlStorage(storage_dir=tmp_path / "sql")
    file_storage = LocalFileStorage(storage_dir=tmp_path / "file")
    vector_storage = FakeVectorStorage()
    rag = RagService(
        llm=FakeLLM(),
        embedder=FakeEmbedder(),
        retriever=NoopRetriever(),
        vector_storage=vector_storage,
        sql_storage=sql_storage,
        file_storage=file_storage,
    )

    async def flow():
        await sql_storage.create_tables()
        await seed_tree(sql_storage, file_storage, vector_storage)
        files = await rag.list_files(chat_id="chat-1")
        chunks = await rag.list_file_chunks("origin-1", chat_id="chat-1")
        deleted = await rag.delete_file("origin-1", chat_id="chat-1")
        await sql_storage.close()
        return files, chunks, deleted

    files, chunks, deleted = asyncio.run(flow())

    assert [doc["source_id"] for doc in files] == ["origin-1"]
    assert _row_ids(chunks, "chunk_id") == {"c1", "c2"}
    assert deleted == 2


def test_rag_service_delegates_file_metadata_and_link(tmp_path):
    from app.services.rag import RagService

    sql_storage = LocalSqlStorage(storage_dir=tmp_path / "sql")
    file_storage = LocalFileStorage(storage_dir=tmp_path / "file")
    vector_storage = FakeVectorStorage()
    rag = RagService(
        llm=FakeLLM(),
        embedder=FakeEmbedder(),
        retriever=NoopRetriever(),
        vector_storage=vector_storage,
        sql_storage=sql_storage,
        file_storage=file_storage,
    )

    async def flow():
        await sql_storage.create_tables()
        await seed_tree(sql_storage, file_storage, vector_storage)
        meta = await rag.get_file_metadata("origin-1")
        link = await rag.get_file_link("origin-1")
        await sql_storage.close()
        return meta, link

    meta, link = asyncio.run(flow())

    assert meta is not None and meta["source_id"] == "origin-1"
    assert link is not None and link.startswith("/files/")
