from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.errors.auth import AuthError, UserExistsError
from app.errors.chat import ChatForbiddenError, ChatNotFoundError
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

    @app.exception_handler(UserExistsError)
    async def user_exists_handler(
        request: Request,
        exc: UserExistsError,
    ):
        return JSONResponse(
            status_code=409,
            content={
                "detail": str(exc),
            },
        )

    @app.exception_handler(AuthError)
    async def auth_error_handler(
        request: Request,
        exc: AuthError,
    ):
        return JSONResponse(
            status_code=401,
            content={
                "detail": str(exc),
            },
        )

    @app.exception_handler(ChatNotFoundError)
    async def chat_not_found_handler(
        request: Request,
        exc: ChatNotFoundError,
    ):
        return JSONResponse(
            status_code=404,
            content={
                "detail": str(exc),
            },
        )

    @app.exception_handler(ChatForbiddenError)
    async def chat_forbidden_handler(
        request: Request,
        exc: ChatForbiddenError,
    ):
        return JSONResponse(
            status_code=403,
            content={
                "detail": str(exc),
            },
        )
