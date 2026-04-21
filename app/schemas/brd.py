from pydantic import BaseModel, Field


class WorkflowCreateRequest(BaseModel):
    title: str = Field(..., examples=['Customer Portal BRD'])
    brd_content: str
    max_retry: int = 2


class WorkflowCreateResponse(BaseModel):
    workflow_id: int
    message: str
