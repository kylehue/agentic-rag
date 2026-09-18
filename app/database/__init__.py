from app.database.base import Base
from app.database.chunks import Chunk
from app.database.documents import Document
from app.database.chats import Chat
from app.database.users import User
from app.database.auth_tokens import AuthToken

# The tables' names, re-exported for call sites. The literals live in each
# model's __tablename__; these are stable aliases, not second definitions.
CHUNK_TABLE_NAME = Chunk.__tablename__
DOCUMENT_METADATA_TABLE_NAME = Document.__tablename__
USERS_TABLE_NAME = User.__tablename__
AUTH_TOKENS_TABLE_NAME = AuthToken.__tablename__
CHATS_TABLE_NAME = Chat.__tablename__

# Every model, in display order. Importing them is what registers each table
# on Base.metadata (and hence what create_tables knows about).
_MODELS = (Chunk, Document, Chat, User, AuthToken)


def agent_table_docs() -> str:
    """A readable description of the ingestion tables, for the agent prompt.

    Only tables that opt in via ``__expose_docs_to_agent__`` are included.
    Each is rendered with its purpose (the class doc) and one line per column
    (the column's name and its doc), so the prompt stays in step with the
    schema instead of being a hand-maintained copy.
    """
    blocks = []
    for model in _MODELS:
        if not getattr(model, "__expose_docs_to_agent__", False):
            continue
        purpose = " ".join((model.__doc__ or "").split())
        lines = [f"- `{model.__tablename__}`: {purpose}"]
        for column in model.__table__.columns:
            col_doc = " ".join((column.doc or "").split())
            lines.append(f"  - `{column.name}`: {col_doc}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


__all__ = [
    "AUTH_TOKENS_TABLE_NAME",
    "CHATS_TABLE_NAME",
    "CHUNK_TABLE_NAME",
    "DOCUMENT_METADATA_TABLE_NAME",
    "USERS_TABLE_NAME",
    "Base",
    "AuthToken",
    "Chat",
    "Chunk",
    "Document",
    "User",
    "agent_table_docs",
]
