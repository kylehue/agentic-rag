from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """The declarative base the app's tables map onto.

    Each table is one ORM model in its own module (one file per table); the
    models register themselves on this base's metadata, which is what the
    SQL storage's ``create_tables`` applies to the database.
    """
