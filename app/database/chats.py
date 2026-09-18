from sqlalchemy import Column, REAL, String

from app.database.base import Base


class Chat(Base):
    """A user's working space.

    A chat is the unit of a user's documents and conversation: the chunks and
    files ingested into it are stamped with its id (which bounds their
    retrieval), and it is also the thread the agent's conversation runs on.
    """

    __tablename__ = "__chats__"

    chat_id = Column(
        String,
        primary_key=True,
        doc="The chat's id; also the id of the agent's conversation thread.",
    )
    username = Column(String, nullable=False, doc="The chat's owner.")
    created_at = Column(REAL, nullable=False, doc="Unix time the chat was created.")
