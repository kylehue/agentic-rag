from sqlalchemy import Column, REAL, String

from app.database.base import Base


class AuthToken(Base):
    """A login session token, stored only as a bcrypt hash and looked up by
    its unique prefix (``token_id``)."""

    __tablename__ = "__auth_tokens__"

    token_hash = Column(String, primary_key=True)
    token_id = Column(String, nullable=True)
    username = Column(String, nullable=False)
    created_at = Column(REAL, nullable=False)
