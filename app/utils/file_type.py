from unstructured.file_utils.model import FileType

from app.models.chunk import ChunkCategory

DOCUMENT_TYPES = {
    "pdf",
    "docx",
    "doc",
    "txt",
    "md",
    "pptx",
    "ppt",
}

SPREADSHEET_TYPES = {
    "csv",
    "xlsx",
    "xls",
}

IMAGE_TYPES = {
    "png",
    "jpg",
    "jpeg",
}


def detect_document_category(filename: str) -> ChunkCategory:
    normalized = filename.split(".")[-1]
    if normalized in SPREADSHEET_TYPES:
        return ChunkCategory.SPREADSHEET
    elif normalized in IMAGE_TYPES:
        return ChunkCategory.IMAGE
    else:
        return ChunkCategory.DOCUMENT
