from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.repositories.audit_repository import AuditRepository
from app.repositories.task_repository import TaskRepository
from app.schemas.common import EventLogRead, StateTransitionRead
from app.schemas.task import TaskDetailResponse, TaskRead

router = APIRouter(prefix='/tasks', tags=['tasks'])


@router.get('', response_model=list[TaskRead])
def list_tasks(status: str | None = Query(default=None), assignee_team: str | None = Query(default=None), retry_count_gte: int | None = Query(default=None), db: Session = Depends(get_db)):
    tasks = TaskRepository(db).list_tasks()
    if status is not None:
        tasks = [task for task in tasks if task.status == status]
    if assignee_team is not None:
        tasks = [task for task in tasks if task.assignee_team == assignee_team]
    if retry_count_gte is not None:
        tasks = [task for task in tasks if task.retry_count >= retry_count_gte]
    return tasks


@router.get('/{task_id}', response_model=TaskDetailResponse)
def get_task_detail(task_id: int, db: Session = Depends(get_db)):
    task_repo = TaskRepository(db)
    audit_repo = AuditRepository(db)
    task = task_repo.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail='Task not found')
    criteria = task_repo.get_acceptance_criteria_for_task(task_id)
    bugs = task_repo.list_bugs_for_task(task_id)
    return TaskDetailResponse(
        task=task,
        acceptance_criteria=[item.text for item in criteria],
        bugs=[{'id': b.id, 'title': b.title, 'description': b.description, 'severity': b.severity, 'failed_criteria': b.failed_criteria, 'status': b.status, 'created_at': b.created_at} for b in bugs],
        transitions=audit_repo.list_transitions(task_id=task_id),
        events=audit_repo.list_events(workflow_id=task.workflow_id, task_id=task_id),
    )


@router.get('/{task_id}/transitions', response_model=list[StateTransitionRead])
def list_task_transitions(task_id: int, db: Session = Depends(get_db)):
    return AuditRepository(db).list_transitions(task_id=task_id)


@router.get('/{task_id}/events', response_model=list[EventLogRead])
def list_task_events(task_id: int, db: Session = Depends(get_db)):
    task = TaskRepository(db).get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail='Task not found')
    return AuditRepository(db).list_events(workflow_id=task.workflow_id, task_id=task_id)
