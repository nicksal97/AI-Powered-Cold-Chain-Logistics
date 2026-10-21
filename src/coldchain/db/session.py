from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from coldchain.config import get_settings


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    s = get_settings()
    return create_engine(
        url or s.database_url,
        pool_size=s.db_pool_size,
        pool_pre_ping=True,
        future=True,
    )


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    factory = sessionmaker(bind=engine or get_engine(), expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ping(engine: Engine | None = None) -> bool:
    try:
        with (engine or get_engine()).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
