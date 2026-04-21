from datetime import datetime
from pydantic import BaseModel

from app.schemas.common import EventLogSummary, StateTransitionRead
from app.schemas.task import TaskSummary


class WorkflowRead(BaseModel):
    id: int
    brd_id: int
    status: str
    current_agent: str | None
    max_retry: int
    metadata_json: dict
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {'from_attributes': True}


class WorkflowDetailResponse(BaseModel):
    workflow: WorkflowRead
    tasks: list[TaskSummary]
    transitions: list[StateTransitionRead]
    events: list[EventLogSummary]


class TriggerWorkflowResponse(BaseModel):
    workflow_id: int
    status: str
    message: str
