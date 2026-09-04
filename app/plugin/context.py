from dataclasses import dataclass, field

from unstructured.documents.elements import Element


@dataclass(frozen=True)
class IngestionContext:
    """Read-only details about the document being ingested."""

    source_id: str
    source_filename: str
    source_content_type: str
    source_bytes: bytes
    elements: list[Element] = field(default_factory=list)
    parent_source_id: str | None = None


@dataclass(frozen=True)
class EmittedFile:
    """A file emitted by a plugin so it can be ingested by another plugin."""

    filename: str
    content_type: str
    file_bytes: bytes
    source_id: str
