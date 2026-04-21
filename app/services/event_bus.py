from dataclasses import dataclass
from typing import Protocol

from app.repositories.audit_repository import AuditRepository


@dataclass
class EventMessage:
    workflow_id: int
    event_type: str
    message: str
    agent_name: str | None = None
    task_id: int | None = None
    payload: dict | None = None


class EventPublisher(Protocol):
    def publish(self, event: EventMessage) -> None: ...


class DatabaseEventBus:
    def __init__(self, audit_repo: AuditRepository):
        self.audit_repo = audit_repo

    def publish(self, event: EventMessage) -> None:
        self.audit_repo.create_event(
            workflow_id=event.workflow_id,
            task_id=event.task_id,
            event_type=event.event_type,
            agent_name=event.agent_name,
            message=event.message,
            payload=event.payload or {},
        )
