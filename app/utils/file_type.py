from unstructured.file_utils.model import FileType

from app.models.document import DocumentCategory

DOCUMENT_TYPES = {
    "pdf",
    "docx",
    "txt",
    "md",
    "pptx",
}

SPREADSHEET_TYPES = {
    "csv",
    "xlsx",
}

IMAGE_TYPES = {
    "png",
    "jpg",
    "jpeg",
}


def detect_document_category(file_extension: str) -> DocumentCategory:
    normalized = (
        file_extension[1:] if file_extension.startswith(".") else file_extension
    )
    if normalized in SPREADSHEET_TYPES:
        return DocumentCategory.SPREADSHEET
    elif normalized in IMAGE_TYPES:
        return DocumentCategory.IMAGE
    else:
        return DocumentCategory.DOCUMENT
