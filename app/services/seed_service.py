from pathlib import Path

from app.services.workflow_service import WorkflowService


class SeedService:
    def __init__(self, workflow_service: WorkflowService):
        self.workflow_service = workflow_service

    def seed_demo_workflow(self) -> int:
        sample_path = Path(__file__).resolve().parent.parent / 'sample_data' / 'sample_brd.md'
        content = sample_path.read_text(encoding='utf-8')
        return self.workflow_service.create_workflow_from_brd('Demo BRD - Customer Portal', content, max_retry=2)
