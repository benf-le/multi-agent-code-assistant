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
    feature_summary: str = Field(default="", description="Summary of the analyzed features")
    user_stories: list[StoryOut] = Field(default_factory=list)
    backlog_items: list[BacklogItemOut] = Field(default_factory=list)
    implementation_tasks: list[TaskOut] = Field(default_factory=list)


class DevResult(BaseModel):
    file_path: str = Field(
        description="The relative file path within the project structure, e.g., 'app/api/admin.py', 'app/services/auth_service.py', 'tests/test_admin.py'. "
                    "Must reflect a proper project layout (no flat names like 'admin_api.py' at root)."
    )
    code: str
    unit_tests: str = Field(default="", description="The unit tests for the implementation")
    implementation_notes: str = Field(default="", description="Notes about the implementation")
    included_markers: list[str] = Field(default_factory=list)

    @property
    def file_name(self) -> str:
        """Backward-compat: basename of file_path."""
        return self.file_path.split('/')[-1].split('\\')[-1]


class QCResult(BaseModel):
    passed: bool
    status: str = Field(default="COMPLETED", description="Status of the validation (e.g., PASSED, FAILED)")
    validation_report: str = Field(default="", description="Detailed report of the validation")
    failed_criteria: list[str] = Field(default_factory=list)
    severity: str = 'MEDIUM'
