from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.errors.document import DocumentNotFoundError, InvalidDocumentError


def register_exception_handlers(app: FastAPI):
    @app.exception_handler(DocumentNotFoundError)
    async def document_not_found_handler(
        request: Request,
        exc: DocumentNotFoundError,
    ):
        return JSONResponse(
            status_code=404,
            content={
                "detail": str(exc),
            },
        )

    @app.exception_handler(InvalidDocumentError)
    async def invalid_document_handler(
        request: Request,
        exc: InvalidDocumentError,
    ):
        return JSONResponse(
            status_code=404,
            content={
                "detail": str(exc),
            },
        )
