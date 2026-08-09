class DocumentNotFoundError(Exception):
    def __init__(self, document_id: str):
        self.document_id = document_id
        super().__init__(f"Document '{document_id}' not found.")


class InvalidDocumentError(Exception):
    def __init__(self, message: str = "Invalid document."):
        super().__init__(message)
