import app.models  # noqa: F401
import pytest
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.session import build_engine
from app.services.workflow_service import WorkflowService


@pytest.fixture()
def session_factory():
    engine = build_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(bind=engine)
    SessionTesting = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return SessionTesting


@pytest.fixture()
def workflow_service(session_factory):
    return WorkflowService(session_factory=session_factory)
