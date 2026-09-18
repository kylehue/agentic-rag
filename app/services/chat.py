import time
from uuid import uuid4

from app.database import CHATS_TABLE_NAME, DOCUMENT_METADATA_TABLE_NAME
from app.errors.chat import ChatForbiddenError, ChatNotFoundError
from app.store_sql.base import SqlStorage


class ChatService:
    """User chats.

    A chat is the unit of a user's working space: the chunks ingested into
    it (stamped with the chat id at ingestion, which bounds its retrieval)
    and the agent conversation thread that runs on it. Decoupled from the
    RAG services: it only needs SQL storage.
    """

    def __init__(self, *, sql_storage: SqlStorage) -> None:
        self._sql_storage = sql_storage

    async def create(self, username: str) -> str:
        chat_id = uuid4().hex
        await self._sql_storage.upsert(
            CHATS_TABLE_NAME,
            [
                {
                    "chat_id": chat_id,
                    "username": username,
                    "created_at": time.time(),
                }
            ],
        )
        return chat_id

    async def delete(self, chat_id: str) -> None:
        """Remove the chat's row (its data and history are deleted separately)."""
        await self._sql_storage.delete(
            CHATS_TABLE_NAME,
            condition=lambda t: t.c.chat_id == chat_id,
        )

    async def get_or_create(self, username: str, chat_id: str | None = None) -> str:
        """The chat id to use: the given one (verified as this user's) or a
        new one."""
        if chat_id is None:
            return await self.create(username)

        chat = await self.username_row(chat_id)
        if chat is None:
            raise ChatNotFoundError(chat_id)
        if chat["username"] != username:
            raise ChatForbiddenError(chat_id)
        return chat_id

    async def username_of(self, chat_id: str) -> str | None:
        """The owner of a chat, or None when it does not exist."""
        chat = await self.username_row(chat_id)
        return chat["username"] if chat is not None else None

    async def username_row(self, chat_id: str) -> dict | None:
        return await self._sql_storage.get(
            CHATS_TABLE_NAME,
            condition=lambda t: t.c.chat_id == chat_id,
        )

    async def list_chats(self, username: str) -> list[dict]:
        """The user's chats, each with the origin source ids ingested into
        it (derived from the documents table; no separate bookkeeping)."""
        chats = list(
            await self._sql_storage.get_all(
                CHATS_TABLE_NAME,
                condition=lambda t: t.c.username == username,
            )
        )
        chat_ids = [chat["chat_id"] for chat in chats]
        if not chat_ids:
            return []

        documents = await self._sql_storage.get_all(
            DOCUMENT_METADATA_TABLE_NAME,
            condition=lambda t: t.c.chat_id.in_(chat_ids) & t.c.is_origin.is_(True),
        )
        by_chat: dict[str, list[str]] = {}
        for row in documents:
            by_chat.setdefault(row["chat_id"], []).append(row["source_id"])

        return [
            {
                "chat_id": chat["chat_id"],
                "created_at": chat["created_at"],
                "sources": by_chat.get(chat["chat_id"], []),
            }
            for chat in chats
        ]
