from pydantic import BaseModel, Field


class BacklogItemOut(BaseModel):
    title: str
    description: str
    team: str

class StoryOut(BaseModel):
    title: str
    description: str
    priority: str = 'MEDIUM'
    acceptance_criteria: list[str] = Field(default_factory=list)


class TaskOut(BaseModel):
    title: str
    description: str
    assignee_team: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    required_markers: list[str] = Field(default_factory=list)
    input_context: dict = Field(default_factory=dict)


class POResult(BaseModel):
    feature_summary: str
    user_stories: list[StoryOut] = Field(default_factory=list)
    backlog_items: list[BacklogItemOut] = Field(default_factory=list)
    implementation_tasks: list[TaskOut] = Field(default_factory=list)


class DevResult(BaseModel):
    file_name: str = Field(description="The suggested name for the source code file, e.g., 'auth.py' or 'App.tsx'")
    code: str
    unit_tests: str
    implementation_notes: str
    included_markers: list[str] = Field(default_factory=list)


class QCResult(BaseModel):
    passed: bool
    status: str
    validation_report: str
    failed_criteria: list[str] = Field(default_factory=list)
    severity: str = 'MEDIUM'
