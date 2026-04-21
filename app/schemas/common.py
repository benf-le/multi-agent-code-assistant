from datetime import datetime
from pydantic import BaseModel


class EventLogSummary(BaseModel):
    id: int
    workflow_id: int
    task_id: int | None
    event_type: str
    agent_name: str | None
    message: str
    created_at: datetime

    model_config = {'from_attributes': True}


class EventLogRead(EventLogSummary):
    payload: dict


class StateTransitionRead(BaseModel):
    id: int
    workflow_id: int
    task_id: int | None
    from_status: str | None
    to_status: str
    agent_name: str | None
    reason: str | None
    metadata_json: dict
    created_at: datetime

    model_config = {'from_attributes': True}
