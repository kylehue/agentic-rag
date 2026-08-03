class DocumentNotFoundError(Exception):
    def __init__(self, document_id: str):
        self.document_id = document_id
        super().__init__(f"Document '{document_id}' not found.")
