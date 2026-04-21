from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.session import SessionLocal  # noqa: E402
from app.services.seed_service import SeedService  # noqa: E402
from app.services.workflow_service import WorkflowService  # noqa: E402


def session_factory():
    return SessionLocal()


if __name__ == '__main__':
    workflow_id = SeedService(WorkflowService(session_factory=session_factory)).seed_demo_workflow()
    print(f'Seeded demo workflow_id={workflow_id}')
