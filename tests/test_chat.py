import asyncio

import pytest

from app.database import DOCUMENT_METADATA_TABLE_NAME
from app.errors.chat import ChatForbiddenError, ChatNotFoundError
from app.services.chat import ChatService
from app.store_sql.local import LocalSqlStorage


def make_service(tmp_path) -> ChatService:
    sql_storage = LocalSqlStorage(storage_dir=tmp_path / "sql")
    service = ChatService(sql_storage=sql_storage)
    # Create the schema (the chats and documents tables).
    asyncio.run(sql_storage.create_tables())
    return service


def add_document(sql_storage, source_id, chat_id, is_origin=True):
    asyncio.run(
        sql_storage.upsert(
            DOCUMENT_METADATA_TABLE_NAME,
            [
                {
                    "source_id": source_id,
                    "file_path": f"documents/{source_id}",
                    "file_content_type": "text/plain",
                    "file_filename": f"{source_id}.txt",
                    "file_orig_filename": f"{source_id}.txt",
                    "is_origin": is_origin,
                    "chat_id": chat_id,
                }
            ],
        )
    )


def test_get_or_create_makes_a_chat(tmp_path):
    service = make_service(tmp_path)

    chat_id = asyncio.run(service.get_or_create("alice"))

    assert chat_id
    # Asking for the same chat again returns it.
    assert asyncio.run(service.get_or_create("alice", chat_id)) == chat_id


def test_get_or_create_unknown_chat(tmp_path):
    service = make_service(tmp_path)

    with pytest.raises(ChatNotFoundError):
        asyncio.run(service.get_or_create("alice", "nope"))


def test_get_or_create_rejects_another_users_chat(tmp_path):
    service = make_service(tmp_path)
    alice = asyncio.run(service.get_or_create("alice"))

    with pytest.raises(ChatForbiddenError):
        asyncio.run(service.get_or_create("bob", alice))


def test_username_of(tmp_path):
    service = make_service(tmp_path)
    chat_id = asyncio.run(service.get_or_create("alice"))

    assert asyncio.run(service.username_of(chat_id)) == "alice"
    assert asyncio.run(service.username_of("nope")) is None


def test_list_chats_derives_sources_from_documents(tmp_path):
    service = make_service(tmp_path)
    sql_storage = service._sql_storage
    c1 = asyncio.run(service.get_or_create("alice"))
    c2 = asyncio.run(service.get_or_create("alice"))
    asyncio.run(service.get_or_create("bob"))

    add_document(sql_storage, "s1", c1)
    add_document(sql_storage, "s2", c1)
    add_document(sql_storage, "s3", c2, is_origin=False)  # emitted file: not a source
    asyncio.run(service.get_or_create("bob"))

    chats = asyncio.run(service.list_chats("alice"))
    by_id = {chat["chat_id"]: chat for chat in chats}

    # Only alice's chats are listed...
    assert set(by_id) == {c1, c2}
    # ...with their ingested origin sources.
    assert by_id[c1]["sources"] == ["s1", "s2"]
    assert by_id[c2]["sources"] == []
