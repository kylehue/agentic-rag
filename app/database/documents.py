from sqlalchemy import Boolean, Column, Integer, String

from app.database.base import Base


class Document(Base):
    """One stored file: a user upload, or a file a plugin emitted from one.
    is_origin marks the files the user actually uploaded."""

    __tablename__ = "__documents__"
    __expose_docs_to_agent__ = True

    id = Column(Integer, primary_key=True, autoincrement=True, doc="Internal row id.")
    source_id = Column(
        String,
        unique=True,
        nullable=False,
        doc="The file's unique source id. Origin files and plugin-emitted files each have one.",
    )
    file_path = Column(String, nullable=False, doc="Where the file's bytes are stored.")
    file_content_type = Column(
        String, nullable=False, doc="The file's MIME content type."
    )
    file_filename = Column(
        String, nullable=False, doc="The stored (renamed) filename."
    )
    file_orig_filename = Column(
        String, nullable=False, doc="The filename the user uploaded."
    )
    is_origin = Column(
        Boolean,
        nullable=False,
        doc="True for files the user uploaded directly, false for files a plugin emitted from them.",
    )
    chat_id = Column(
        String,
        nullable=True,
        doc="The chat this file was ingested into. Null for data predating chats.",
    )
