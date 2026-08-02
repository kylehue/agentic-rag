from pydantic import BaseModel


class Document(BaseModel):
    id: str
    filename: str
    path: str


class DocumentUploadResponse(BaseModel):
    message: str
    document: Document


class DocumentDeleteResponse(BaseModel):
    message: str
