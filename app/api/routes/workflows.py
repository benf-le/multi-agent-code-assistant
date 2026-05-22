import json
import asyncio
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.repositories.audit_repository import AuditRepository
from app.repositories.task_repository import TaskRepository
from app.repositories.workflow_repository import WorkflowRepository
from app.schemas.brd import WorkflowCreateRequest, WorkflowCreateResponse
from app.schemas.common import EventLogRead, StateTransitionRead
from app.schemas.task import TaskRead
from app.schemas.workflow import TriggerWorkflowResponse, WorkflowDetailResponse, WorkflowRead
from app.services.workflow_service import WorkflowService
from app.services.log_streamer import log_streamer

router = APIRouter(prefix='/workflows', tags=['workflows'])


def _session_factory():
    from app.db.session import SessionLocal
    return SessionLocal()


@router.post('', response_model=WorkflowCreateResponse)
def create_workflow(payload: WorkflowCreateRequest):
    service = WorkflowService(session_factory=_session_factory)
    workflow_id = service.create_workflow_from_brd(payload.title, payload.brd_content, payload.max_retry)
    return WorkflowCreateResponse(workflow_id=workflow_id, message='Workflow created successfully.')


@router.get('', response_model=list[WorkflowRead])
def list_workflows(db: Session = Depends(get_db)):
    return WorkflowRepository(db).list()


@router.get('/{workflow_id}', response_model=WorkflowDetailResponse)
def get_workflow_detail(workflow_id: int, db: Session = Depends(get_db)):
    workflow_repo = WorkflowRepository(db)
    task_repo = TaskRepository(db)
    audit_repo = AuditRepository(db)
    workflow = workflow_repo.get(workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail='Workflow not found')
    return WorkflowDetailResponse(
        workflow=workflow,
        tasks=task_repo.list_tasks(workflow_id=workflow_id),
        transitions=audit_repo.list_transitions(workflow_id=workflow_id),
        events=audit_repo.list_events(workflow_id=workflow_id),
    )


@router.delete('/{workflow_id}')
def delete_workflow(workflow_id: int, db: Session = Depends(get_db)):
    workflow_repo = WorkflowRepository(db)
    try:
        success = workflow_repo.delete(workflow_id)
        if not success:
            raise HTTPException(status_code=404, detail='Workflow not found')
        return {'message': 'Workflow deleted successfully'}
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        # Log the full error for server logs
        import logging
        logging.getLogger(__name__).error(f"Error deleting workflow {workflow_id}: {error_msg}", exc_info=True)
        
        if "ForeignKeyViolation" in error_msg or "foreign key constraint" in error_msg:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot delete workflow because it is still referenced by other data: {error_msg}"
            )
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {error_msg}")



@router.post('/{workflow_id}/run', response_model=TriggerWorkflowResponse)
def trigger_workflow(workflow_id: int, background_tasks: BackgroundTasks, sync: bool = False):
    service = WorkflowService(session_factory=_session_factory)
    if sync:
        service.run_workflow(workflow_id)
    else:
        background_tasks.add_task(service.run_workflow, workflow_id)
    return TriggerWorkflowResponse(workflow_id=workflow_id, status='RUNNING', message='Workflow execution has been triggered.')


@router.post('/{workflow_id}/stop')
def stop_workflow(workflow_id: int, db: Session = Depends(get_db)):
    workflow_repo = WorkflowRepository(db)
    workflow = workflow_repo.get(workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail='Workflow not found')
    
    if workflow.status in ['DONE', 'CANCELLED', 'BLOCKED', 'MAX_RETRY_EXCEEDED']:
        return {'message': f'Workflow is already in a terminal state: {workflow.status}'}

    workflow_repo.update_status(workflow, status='CANCELLED', current_agent='SYSTEM')
    db.commit()
    return {'message': 'Workflow cancellation requested.'}


@router.post('/{workflow_id}/resume', response_model=TriggerWorkflowResponse)
def resume_workflow(workflow_id: int, background_tasks: BackgroundTasks, sync: bool = False):
    service = WorkflowService(session_factory=_session_factory)
    if sync:
        service.resume_workflow(workflow_id)
    else:
        background_tasks.add_task(service.resume_workflow, workflow_id)
    return TriggerWorkflowResponse(workflow_id=workflow_id, status='RESUMING', message='Workflow resumption has been triggered.')


@router.get('/{workflow_id}/events', response_model=list[EventLogRead])
def list_workflow_events(workflow_id: int, db: Session = Depends(get_db)):
    return AuditRepository(db).list_events(workflow_id=workflow_id)


@router.get('/{workflow_id}/transitions', response_model=list[StateTransitionRead])
def list_workflow_transitions(workflow_id: int, db: Session = Depends(get_db)):
    return AuditRepository(db).list_transitions(workflow_id=workflow_id)


@router.get('/{workflow_id}/tasks', response_model=list[TaskRead])
def list_workflow_tasks(workflow_id: int, db: Session = Depends(get_db)):
    return TaskRepository(db).list_tasks(workflow_id=workflow_id)


@router.get('/{workflow_id}/export')
def export_workflow_logs(workflow_id: int):
    service = WorkflowService(session_factory=_session_factory)
    try:
        return service.export_execution_log(workflow_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get('/{workflow_id}/logs/stream')
async def stream_workflow_logs(workflow_id: int):
    async def event_generator():
        try:
            async for log_entry in log_streamer.subscribe(workflow_id):
                yield f"data: {json.dumps(log_entry)}\n\n"
        except asyncio.CancelledError:
            pass
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get('/{workflow_id}/logs')
def get_workflow_logs(workflow_id: int):
    return log_streamer.get_logs(workflow_id)
