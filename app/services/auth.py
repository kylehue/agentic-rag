import secrets
import time

import bcrypt
from sqlalchemy import Column, REAL, String

from app.core.config import settings
from app.errors.auth import AuthError, UserExistsError
from app.store_sql.base import SqlStorage

# The part of a token stored as its lookup key (96 bits of entropy).
TOKEN_PREFIX_LENGTH = 16


class AuthService:
    """Simple username/password authentication.

    Decoupled from the RAG services: it only needs SQL storage. Passwords
    and login tokens are bcrypt hashes (the salt is embedded in the hash).
    A token is looked up by its unique prefix, then verified with bcrypt.
    """

    def __init__(self, *, sql_storage: SqlStorage) -> None:
        self._sql_storage = sql_storage

    async def initialize(self) -> None:
        await self._sql_storage.ensure_table(
            settings.USERS_TABLE_NAME,
            [
                Column("username", String, primary_key=True),
                Column("password_hash", String, nullable=False),
                Column("created_at", REAL, nullable=False),
            ],
        )
        await self._sql_storage.ensure_table(
            settings.AUTH_TOKENS_TABLE_NAME,
            [
                Column("token_hash", String, primary_key=True),
                # Lookup key for the (salted, unguessable) hash. Nullable so
                # pre-prefix rows in existing databases simply stop working.
                Column("token_id", String, nullable=True),
                Column("username", String, nullable=False),
                Column("created_at", REAL, nullable=False),
            ],
        )

    async def register(self, username: str, password: str) -> None:
        existing = await self._sql_storage.get(
            settings.USERS_TABLE_NAME,
            condition=lambda t: t.c.username == username,
        )
        if existing is not None:
            raise UserExistsError(username)

        await self._sql_storage.upsert(
            settings.USERS_TABLE_NAME,
            [
                {
                    "username": username,
                    "password_hash": self._hash_password(password),
                    "created_at": time.time(),
                }
            ],
        )

    async def login(self, username: str, password: str) -> str:
        """Verify the credentials and return a new API token."""
        user = await self._sql_storage.get(
            settings.USERS_TABLE_NAME,
            condition=lambda t: t.c.username == username,
        )
        if user is None or not self._verify_password(password, user["password_hash"]):
            raise AuthError()

        token = secrets.token_urlsafe(32)
        await self._sql_storage.upsert(
            settings.AUTH_TOKENS_TABLE_NAME,
            [
                {
                    "token_hash": self._hash_token(token),
                    "token_id": token[:TOKEN_PREFIX_LENGTH],
                    "username": username,
                    "created_at": time.time(),
                }
            ],
        )
        return token

    async def verify_token(self, token: str) -> str | None:
        """The username a token belongs to, or None when it is unknown."""
        if not token or len(token) < TOKEN_PREFIX_LENGTH:
            return None
        row = await self._sql_storage.get(
            settings.AUTH_TOKENS_TABLE_NAME,
            condition=lambda t: t.c.token_id == token[:TOKEN_PREFIX_LENGTH],
        )
        if row is None or not self._verify_token(token, row["token_hash"]):
            return None
        return row["username"]

    @staticmethod
    def _hash_password(password: str) -> str:
        # bcrypt hashes embed their own salt; it also only consumes the
        # first 72 bytes of the input.
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode(
            "ascii"
        )

    @staticmethod
    def _verify_password(password: str, password_hash: str) -> bool:
        try:
            return bcrypt.checkpw(
                password.encode("utf-8"), password_hash.encode("ascii")
            )
        except ValueError:
            return False

    @staticmethod
    def _hash_token(token: str) -> str:
        return bcrypt.hashpw(token.encode("utf-8"), bcrypt.gensalt()).decode("ascii")

    @staticmethod
    def _verify_token(token: str, token_hash: str) -> bool:
        try:
            return bcrypt.checkpw(token.encode("utf-8"), token_hash.encode("ascii"))
        except ValueError:
            return False
