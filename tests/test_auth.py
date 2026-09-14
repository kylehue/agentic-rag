import asyncio

import pytest

from app.core.config import settings
from app.errors.auth import AuthError, UserExistsError
from app.services.auth import AuthService
from app.store_sql.local import LocalSqlStorage


def make_service(tmp_path) -> tuple[AuthService, LocalSqlStorage]:
    sql_storage = LocalSqlStorage(storage_dir=tmp_path / "sql")
    service = AuthService(sql_storage=sql_storage)
    asyncio.run(service.initialize())
    return service, sql_storage


def test_register_then_login(tmp_path):
    service, _ = make_service(tmp_path)

    asyncio.run(service.register("alice", "wonderland"))
    token = asyncio.run(service.login("alice", "wonderland"))

    assert token
    assert asyncio.run(service.verify_token(token)) == "alice"


def test_register_duplicate_user(tmp_path):
    service, _ = make_service(tmp_path)

    asyncio.run(service.register("alice", "wonderland"))

    with pytest.raises(UserExistsError):
        asyncio.run(service.register("alice", "again"))


def test_login_rejects_bad_credentials(tmp_path):
    service, _ = make_service(tmp_path)
    asyncio.run(service.register("alice", "wonderland"))

    with pytest.raises(AuthError):
        asyncio.run(service.login("alice", "wrong"))
    with pytest.raises(AuthError):
        asyncio.run(service.login("bob", "wonderland"))


def test_verify_rejects_unknown_or_empty_tokens(tmp_path):
    service, _ = make_service(tmp_path)
    asyncio.run(service.register("alice", "wonderland"))
    asyncio.run(service.login("alice", "wonderland"))

    assert asyncio.run(service.verify_token("not-a-token")) is None
    assert asyncio.run(service.verify_token("")) is None


def test_tokens_and_passwords_are_stored_hashed(tmp_path):
    service, sql_storage = make_service(tmp_path)
    asyncio.run(service.register("alice", "wonderland"))
    token = asyncio.run(service.login("alice", "wonderland"))

    users = list(asyncio.run(sql_storage.get_all(settings.USERS_TABLE_NAME)))
    # bcrypt: the hash embeds its own salt and never the password.
    assert users[0]["password_hash"] != "wonderland"
    assert users[0]["password_hash"].startswith("$2")

    tokens = list(asyncio.run(sql_storage.get_all(settings.AUTH_TOKENS_TABLE_NAME)))
    assert len(tokens) == 1
    # The raw token is never stored, only its hash.
    assert tokens[0]["token_hash"] != token
    assert asyncio.run(service.verify_token(token)) == "alice"
