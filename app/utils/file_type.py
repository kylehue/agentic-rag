from unstructured.file_utils.model import FileType

from app.models.document import DocumentCategory

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


def detect_document_category(filename: str) -> DocumentCategory:
    normalized = filename.split(".")[-1]
    if normalized in SPREADSHEET_TYPES:
        return DocumentCategory.SPREADSHEET
    elif normalized in IMAGE_TYPES:
        return DocumentCategory.IMAGE
    else:
        return DocumentCategory.DOCUMENT
