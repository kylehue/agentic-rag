from sqlalchemy import JSON, Column, Integer, String, Text

from app.database.base import Base


class Chunk(Base):
    """One searchable piece of an ingested document, with the ids that trace
    it back through the file tree it came from."""

    __tablename__ = "__chunks__"
    __expose_docs_to_agent__ = True

    id = Column(Integer, primary_key=True, autoincrement=True, doc="Internal row id.")
    chunk_id = Column(
        String,
        unique=True,
        nullable=False,
        doc="The chunk's stable id, used for retrieval and citations.",
    )
    source_id = Column(
        String,
        nullable=False,
        doc="The source id of the file this chunk was produced from. For the upload's own chunks it equals origin_source_id; for chunks carved out of an emitted file (a table or image) it is that emitted file's own source id.",
    )
    parent_source_id = Column(
        String,
        nullable=True,
        doc="The source id of the file that emitted this one, one level up the tree. Null for the upload's own chunks.",
    )
    origin_source_id = Column(
        String,
        nullable=False,
        doc="The source id of the top-level file the user uploaded. The same for every chunk in that file's tree, so it is the citation key.",
    )
    plugin = Column(String, nullable=False, doc="Which plugin produced this chunk.")
    text = Column(
        Text,
        nullable=False,
        doc="The chunk's text. For a table chunk this is the table's description and a small sample, not its full data.",
    )
    chunk_metadata = Column(
        "metadata",
        JSON,
        nullable=True,
        doc="Free-form JSON metadata. A table chunk carries its schema, row and column counts, and page number.",
    )
    chat_id = Column(
        String,
        nullable=True,
        doc="The chat this chunk was ingested into. Null for data predating chats.",
    )
