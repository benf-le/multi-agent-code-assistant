from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AgentRun, EventLog, StateTransition


class AuditRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_agent_run(self, workflow_id: int, agent_name: str, status: str, task_id: int | None = None, input_payload: dict | None = None, output_payload: dict | None = None, error_message: str | None = None) -> AgentRun:
        run = AgentRun(
            workflow_id=workflow_id,
            task_id=task_id,
            agent_name=agent_name,
            status=status,
            input_payload=input_payload or {},
            output_payload=output_payload or {},
            error_message=error_message,
            ended_at=datetime.now(timezone.utc),
        )
        self.session.add(run)
        self.session.flush()
        return run

    def create_transition(self, workflow_id: int, to_status: str, task_id: int | None = None, from_status: str | None = None, agent_name: str | None = None, reason: str | None = None, metadata_json: dict | None = None) -> StateTransition:
        transition = StateTransition(
            workflow_id=workflow_id,
            task_id=task_id,
            from_status=from_status,
            to_status=to_status,
            agent_name=agent_name,
            reason=reason,
            metadata_json=metadata_json or {},
        )
        self.session.add(transition)
        self.session.flush()
        return transition

    def create_event(self, workflow_id: int, event_type: str, message: str, task_id: int | None = None, agent_name: str | None = None, payload: dict | None = None) -> EventLog:
        event = EventLog(
            workflow_id=workflow_id,
            task_id=task_id,
            event_type=event_type,
            agent_name=agent_name,
            message=message,
            payload=payload or {},
        )
        self.session.add(event)
        self.session.flush()
        return event

    def list_events(self, workflow_id: int, task_id: int | None = None) -> list[EventLog]:
        stmt = select(EventLog).where(EventLog.workflow_id == workflow_id)
        if task_id is not None:
            stmt = stmt.where(EventLog.task_id == task_id)
        stmt = stmt.order_by(EventLog.id)
        return list(self.session.scalars(stmt).all())

    def list_transitions(self, workflow_id: int | None = None, task_id: int | None = None) -> list[StateTransition]:
        stmt = select(StateTransition)
        if workflow_id is not None:
            stmt = stmt.where(StateTransition.workflow_id == workflow_id)
        if task_id is not None:
            stmt = stmt.where(StateTransition.task_id == task_id)
        stmt = stmt.order_by(StateTransition.id)
        return list(self.session.scalars(stmt).all())

    def list_agent_runs(self, workflow_id: int) -> list[AgentRun]:
        stmt = select(AgentRun).where(AgentRun.workflow_id == workflow_id).order_by(AgentRun.id)
        return list(self.session.scalars(stmt).all())
