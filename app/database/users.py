from sqlalchemy import Column, REAL, String

from app.database.base import Base


class User(Base):
    """A registered user (username + bcrypt password hash)."""

    __tablename__ = "__users__"

    username = Column(String, primary_key=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(REAL, nullable=False)
