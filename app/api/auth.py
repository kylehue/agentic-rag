from fastapi import APIRouter, Depends, HTTPException, Request

from app.api_schemas.auth import (
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    UserResponse,
)
from app.container import auth_service

router = APIRouter(prefix="/auth", tags=["Auth"])


async def require_user(request: Request) -> str:
    """The authenticated username, from the Bearer token.

    The single seam between the auth domain and the rest of the API: other
    routers depend on this, never on the auth service directly.
    """
    header = request.headers.get("Authorization", "")
    token = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
    username = await auth_service.verify_token(token)
    if username is None:
        raise HTTPException(status_code=401, detail="Missing or invalid token.")
    return username


@router.post("/register", status_code=201)
async def register(request: RegisterRequest):
    await auth_service.register(request.username, request.password)
    return UserResponse(username=request.username)


@router.post("/login")
async def login(request: LoginRequest):
    token = await auth_service.login(request.username, request.password)
    return LoginResponse(username=request.username, token=token)


@router.get("/me")
async def me(user: str = Depends(require_user)):
    return UserResponse(username=user)
