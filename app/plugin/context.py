from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class IngestionFile:
    """A file to be ingested: its bytes, identity, and context.

    Used both for top-level ingestion and for files emitted by a plugin
    to be subprocessed by another plugin.

    ``description`` is optional context about the file. When the file is
    (sub)processed, it is available to plugins via
    ``IngestionContext.file.description`` so they can feed it to an LLM.

    ``source_id`` is this file's own id, generated automatically when the
    file is created.
    """

    filename: str
    content_type: str
    file_bytes: bytes
    description: str | None = None
    source_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class IngestionContext:
    """Read-only details about the file being ingested.

    ``file`` is the file of this run; ``parent_file`` is the file that emitted
    it (``None`` for top-level ingestion); ``origin_file`` is the top-most file
    in the emission chain -- the file the whole tree of emitted files traces
    back to (``file`` itself for top-level ingestion).
    """

    file: IngestionFile
    origin_file: IngestionFile
    parent_file: IngestionFile | None = None


@dataclass(frozen=True)
class RetrievalContext:
    """Read-only details about the retrieval request being processed."""

    user_query: str
