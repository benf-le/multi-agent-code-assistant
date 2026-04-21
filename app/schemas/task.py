from datetime import datetime
from pydantic import BaseModel

from app.schemas.common import EventLogRead, StateTransitionRead


class TaskSummary(BaseModel):
    id: int
    workflow_id: int
    backlog_item_id: int | None
    task_number: int
    title: str
    description: str
    assignee_team: str
    status: str
    retry_count: int
    max_retry: int
    current_agent: str | None
    required_markers: list[str]
    latest_bug_id: int | None
    created_at: datetime
    updated_at: datetime

    model_config = {'from_attributes': True}


class TaskRead(TaskSummary):
    input_context: dict
    output_context: dict


class TaskDetailResponse(BaseModel):
    task: TaskRead
    acceptance_criteria: list[str]
    bugs: list[dict]
    transitions: list[StateTransitionRead]
    events: list[EventLogRead]
