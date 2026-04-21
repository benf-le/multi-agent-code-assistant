from datetime import datetime
from pydantic import BaseModel

from app.schemas.common import EventLogRead, StateTransitionRead
from app.schemas.task import TaskRead


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
    tasks: list[TaskRead]
    transitions: list[StateTransitionRead]
    events: list[EventLogRead]


class TriggerWorkflowResponse(BaseModel):
    workflow_id: int
    status: str
    message: str
