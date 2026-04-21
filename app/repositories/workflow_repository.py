from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import WorkflowExecution


class WorkflowRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_workflow(self, brd_id: int, max_retry: int, metadata_json: dict | None = None) -> WorkflowExecution:
        workflow = WorkflowExecution(
            brd_id=brd_id,
            status='NEW',
            current_agent='ORCHESTRATOR',
            max_retry=max_retry,
            metadata_json=metadata_json or {},
        )
        self.session.add(workflow)
        self.session.flush()
        return workflow

    def get(self, workflow_id: int) -> WorkflowExecution | None:
        return self.session.get(WorkflowExecution, workflow_id)

    def list(self) -> list[WorkflowExecution]:
        stmt = select(WorkflowExecution).order_by(WorkflowExecution.id.desc())
        return list(self.session.scalars(stmt).all())

    def update_status(self, workflow: WorkflowExecution, status: str, current_agent: str | None = None) -> WorkflowExecution:
        workflow.status = status
        workflow.current_agent = current_agent
        if workflow.started_at is None:
            workflow.started_at = datetime.now(timezone.utc)
        self.session.add(workflow)
        self.session.flush()
        return workflow

    def mark_finished(self, workflow: WorkflowExecution, status: str) -> WorkflowExecution:
        workflow.status = status
        workflow.ended_at = datetime.now(timezone.utc)
        self.session.add(workflow)
        self.session.flush()
        return workflow

    def delete(self, workflow_id: int) -> bool:
        workflow = self.get(workflow_id)
        if not workflow:
            return False

        from sqlalchemy import update
        from app.models.audit import AgentRun, StateTransition, EventLog
        from app.models.brd import Feature, UserStory, AcceptanceCriteria
        from app.models.bug import BugReport
        from app.models.task import Task, BacklogItem

        try:
            brd_id = workflow.brd_id

            # 1. Break circular dependency: Task <-> BugReport
            # Set latest_bug_id to NULL for all tasks in this workflow
            self.session.execute(
                update(Task)
                .where(Task.workflow_id == workflow_id)
                .values(latest_bug_id=None)
            )
            self.session.flush()

            # 2. Delete dependent entities in correct order
            
            # A. Delete entities that reference Tasks but are NOT the WorkflowExecution
            
            # Delete BugReports first (references Task.id)
            self.session.query(BugReport).filter(BugReport.workflow_id == workflow_id).delete(synchronize_session=False)
            
            # Delete AcceptanceCriteria (references Task.id and UserStory.id)
            self.session.query(AcceptanceCriteria).filter(AcceptanceCriteria.workflow_id == workflow_id).delete(synchronize_session=False)
            
            # Delete Audit/Logs (these reference Task.id and must be deleted BEFORE Task)
            self.session.query(EventLog).filter(EventLog.workflow_id == workflow_id).delete(synchronize_session=False)
            self.session.query(StateTransition).filter(StateTransition.workflow_id == workflow_id).delete(synchronize_session=False)
            self.session.query(AgentRun).filter(AgentRun.workflow_id == workflow_id).delete(synchronize_session=False)
            
            # B. Now delete the Tasks themselves
            self.session.query(Task).filter(Task.workflow_id == workflow_id).delete(synchronize_session=False)
            
            # C. Delete other entities belonging to the workflow
            
            # Delete BacklogItems
            self.session.query(BacklogItem).filter(BacklogItem.workflow_id == workflow_id).delete(synchronize_session=False)
            
            # Delete UserStories
            self.session.query(UserStory).filter(UserStory.workflow_id == workflow_id).delete(synchronize_session=False)
            
            # Delete Features
            self.session.query(Feature).filter(Feature.workflow_id == workflow_id).delete(synchronize_session=False)


            # 3. Delete the workflow itself
            self.session.delete(workflow)
            self.session.flush()

            # 4. Clean up BRD if orphan
            from app.models.brd import BRD
            # Check if any OTHER workflow still uses this brd
            other_exists = self.session.query(WorkflowExecution).filter(
                WorkflowExecution.brd_id == brd_id,
                WorkflowExecution.id != workflow_id
            ).first()
            
            if not other_exists:
                self.session.query(BRD).filter(BRD.id == brd_id).delete(synchronize_session=False)

            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            raise e


