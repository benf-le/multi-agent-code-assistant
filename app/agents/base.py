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


class POReviewIssue(BaseModel):
    """A single validation issue found during PO review."""
    category: str = Field(description="Category of the issue: 'story_format', 'completeness', 'task_clarity', 'duplicate', 'traceability'")
    severity: str = Field(default='MEDIUM', description="Issue severity: LOW, MEDIUM, HIGH, CRITICAL")
    description: str = Field(description="Clear description of what is wrong")
    affected_items: list[str] = Field(default_factory=list, description="Identifiers of affected stories/tasks/backlog items")


class POReviewResult(BaseModel):
    """Structured result from the PO review validation gate."""
    decision: str = Field(description="Review decision: 'PASS' or 'NEEDS_REVISION'")
    issues: list[POReviewIssue] = Field(default_factory=list, description="List of validation issues found")
    suggestions: list[str] = Field(default_factory=list, description="Actionable improvement suggestions")
    summary: str = Field(default="", description="Brief summary of the review outcome")


class QCResult(BaseModel):
    passed: bool
    status: str = Field(default="COMPLETED", description="Status of the validation (e.g., PASSED, FAILED)")
    validation_report: str = Field(default="", description="Detailed report of the validation")
    failed_criteria: list[str] = Field(default_factory=list)
    severity: str = 'MEDIUM'
