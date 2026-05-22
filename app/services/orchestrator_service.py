from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.core.enums import AgentName
from app.repositories.audit_repository import AuditRepository
from app.repositories.brd_repository import BRDRepository
from app.repositories.task_repository import TaskRepository
from app.repositories.workflow_repository import WorkflowRepository
from app.services.event_bus import DatabaseEventBus, EventMessage


@dataclass
class OrchestratorContext:
    session: Session
    brd_repo: BRDRepository
    workflow_repo: WorkflowRepository
    task_repo: TaskRepository
    audit_repo: AuditRepository
    event_bus: DatabaseEventBus


class OrchestratorService:
    def __init__(self, context: OrchestratorContext):
        self.ctx = context

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def check_cancellation(self, workflow_id: int) -> bool:
        """Returns True if the workflow has been cancelled."""
        # Use a fresh query to avoid identity map cache
        self.ctx.session.expire_all() # Or more specifically expire the workflow
        workflow = self.ctx.workflow_repo.get(workflow_id)
        if workflow:
            self.ctx.session.refresh(workflow)
        return workflow is not None and workflow.status == 'CANCELLED'

    def update_workflow_status(self, workflow_id: int, to_status: str, agent_name: str, reason: str, task_id: int | None = None, metadata: dict | None = None) -> None:
        workflow = self.ctx.workflow_repo.get(workflow_id)
        if workflow is None:
            raise ValueError(f'Workflow {workflow_id} not found')
        
        # Ensure we have the latest status from DB
        self.ctx.session.refresh(workflow)
        if workflow.status == 'CANCELLED' and to_status != 'CANCELLED':
            raise InterruptedError(f"Cannot update status of cancelled workflow {workflow_id}")

        from_status = workflow.status
        self.ctx.workflow_repo.update_status(workflow, status=to_status, current_agent=agent_name)
        self.ctx.audit_repo.create_transition(
            workflow_id=workflow_id,
            task_id=task_id,
            from_status=from_status,
            to_status=to_status,
            agent_name=agent_name,
            reason=reason,
            metadata_json=metadata or {},
        )
        self.ctx.event_bus.publish(EventMessage(
            workflow_id=workflow_id,
            task_id=task_id,
            event_type='workflow.status.changed',
            agent_name=agent_name,
            message=reason,
            payload={'from_status': from_status, 'to_status': to_status, **(metadata or {})},
        ))
        self.ctx.session.commit()

    def update_task_status(self, workflow_id: int, task_id: int, to_status: str, agent_name: str, reason: str, metadata: dict | None = None) -> None:
        # Check workflow cancellation first
        if self.check_cancellation(workflow_id):
            raise InterruptedError(f"Workflow {workflow_id} was cancelled.")

        task = self.ctx.task_repo.get_task(task_id)
        if task is None:
            raise ValueError(f'Task {task_id} not found')
        from_status = task.status
        self.ctx.task_repo.update_task_status(task, status=to_status, current_agent=agent_name)
        self.ctx.audit_repo.create_transition(
            workflow_id=workflow_id,
            task_id=task_id,
            from_status=from_status,
            to_status=to_status,
            agent_name=agent_name,
            reason=reason,
            metadata_json=metadata or {},
        )
        self.ctx.event_bus.publish(EventMessage(
            workflow_id=workflow_id,
            task_id=task_id,
            event_type='task.status.changed',
            agent_name=agent_name,
            message=reason,
            payload={'from_status': from_status, 'to_status': to_status, **(metadata or {})},
        ))
        self.ctx.session.commit()

    def record_agent_run(self, workflow_id: int, agent_name: str, status: str, task_id: int | None = None, input_payload: dict | None = None, output_payload: dict | None = None, error_message: str | None = None) -> None:
        self.ctx.audit_repo.create_agent_run(
            workflow_id=workflow_id,
            task_id=task_id,
            agent_name=agent_name,
            status=status,
            input_payload=input_payload,
            output_payload=output_payload,
            error_message=error_message,
        )
        self.ctx.session.commit()

    def mark_workflow_finished(self, workflow_id: int, status: str, reason: str) -> None:
        workflow = self.ctx.workflow_repo.get(workflow_id)
        if workflow is None:
            raise ValueError(f'Workflow {workflow_id} not found')
        
        self.ctx.session.refresh(workflow)
        if workflow.status == 'CANCELLED':
            return # Already cancelled, don't overwrite with DONE
            
        from_status = workflow.status
        self.ctx.workflow_repo.mark_finished(workflow, status=status)
        self.ctx.audit_repo.create_transition(
            workflow_id=workflow_id,
            from_status=from_status,
            to_status=status,
            agent_name=AgentName.ORCHESTRATOR.value,
            reason=reason,
        )
        self.ctx.event_bus.publish(EventMessage(
            workflow_id=workflow_id,
            event_type='workflow.finished',
            agent_name=AgentName.ORCHESTRATOR.value,
            message=reason,
            payload={'from_status': from_status, 'to_status': status},
        ))
        self.ctx.session.commit()

    def serialize_task(self, task_id: int) -> dict[str, Any]:
        task = self.ctx.task_repo.get_task(task_id)
        if task is None:
            raise ValueError(f'Task {task_id} not found')
        criteria = self.ctx.task_repo.get_acceptance_criteria_for_task(task_id)
        latest_bug = self.ctx.task_repo.get_bug(task.latest_bug_id) if task.latest_bug_id else None
        return {
            'id': task.id,
            'task_id': f'TASK-{task.task_number:03d}',
            'workflow_id': task.workflow_id,
            'task_number': task.task_number,
            'title': task.title,
            'description': task.description,
            'assignee_team': task.assignee_team,
            'status': task.status,
            'retry_count': task.retry_count,
            'max_retry': task.max_retry,
            'current_agent': task.current_agent,
            'required_markers': task.required_markers,
            'input_context': task.input_context,
            'output_context': task.output_context,
            'acceptance_criteria': [item.text for item in criteria],
            'latest_bug': None if latest_bug is None else {
                'id': latest_bug.id,
                'title': latest_bug.title,
                'description': latest_bug.description,
                'severity': latest_bug.severity,
                'failed_criteria': latest_bug.failed_criteria,
            },
        }
