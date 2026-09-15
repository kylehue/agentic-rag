from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.api_schemas.auth import LoginRequest, RegisterRequest, UserResponse
from app.container import auth_service
from app.core.config import settings

router = APIRouter(prefix="/auth", tags=["Auth"])

# The cookie that carries the session token (set at login, cleared at
# logout). HttpOnly so scripts cannot read it; SameSite=Lax so cross-site
# requests never carry it.
COOKIE_NAME = "auth_token"


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        secure=settings.SESSION_COOKIE_SECURE,
        path="/",
    )


async def require_user(request: Request) -> str:
    """The authenticated username, from the session cookie.

    The single seam between the auth domain and the rest of the API: other
    routers depend on this, never on the auth service directly.
    """
    token = request.cookies.get(COOKIE_NAME, "")
    username = await auth_service.verify_token(token)
    if username is None:
        raise HTTPException(status_code=401, detail="Missing or invalid session.")
    return username


@router.post("/register", status_code=201)
async def register(request: RegisterRequest):
    await auth_service.register(request.username, request.password)
    return UserResponse(username=request.username)


@router.post("/login")
async def login(response: Response, request: LoginRequest):
    token = await auth_service.login(request.username, request.password)
    _set_session_cookie(response, token)
    return UserResponse(username=request.username)


@router.post("/logout")
async def logout(response: Response):
    # Idempotent and unauthenticated: it clears the client's cookie. The
    # token row it pointed to, if any, simply goes unused (same as any
    # other login's row when its cookie is lost).
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"detail": "Logged out."}


@router.get("/me")
async def me(user: str = Depends(require_user)):
    return UserResponse(username=user)
