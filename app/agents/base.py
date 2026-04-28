from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# Enums
# ============================================================

class Priority(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"



class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ReviewDecision(StrEnum):
    PASS = "PASS"
    NEEDS_REVISION = "NEEDS_REVISION"


class ReviewIssueCategory(StrEnum):
    STORY_FORMAT = "story_format"
    COMPLETENESS = "completeness"
    BACKLOG_QUALITY = "backlog_quality"
    TASK_CLARITY = "task_clarity"
    DUPLICATE = "duplicate"
    TRACEABILITY = "traceability"
    ACCEPTANCE_CRITERIA = "acceptance_criteria"
    MARKER_QUALITY = "marker_quality"
    SCHEMA_COMPLIANCE = "schema_compliance"


class QCStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    NEEDS_REVISION = "NEEDS_REVISION"


class CriterionStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    UNSUPPORTED = "UNSUPPORTED"


class Team(StrEnum):
    PRODUCT = "product"
    BACKEND = "backend"
    FRONTEND = "frontend"
    FULLSTACK = "fullstack"
    MOBILE = "mobile"
    DATA = "data"
    DEVOPS = "devops"
    PLATFORM = "platform"
    QA = "qa"
    SECURITY = "security"
    DESIGN = "design"
    UNKNOWN = "unknown"


# ============================================================
# Base helpers
# ============================================================

_ID_PATTERNS = {
    "story_id": re.compile(r"^US-\d{3}$"),
    "backlog_item_id": re.compile(r"^BL-\d{3}$"),
    "task_id": re.compile(r"^TASK-\d{3}$"),
}

_MARKER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

GENERIC_MARKERS_BLACKLIST = {
    "api_endpoint_created",
    "request_validation",
    "input_validation",
    "cache_invalidated",
    "data_persistence_verified",
    "logging_configured",
    "tests_added",
    "unit_tested",
    "migration_created",
    "seed_data_created",
    "index_created",
    "secure",
    "works_correctly",
    "backend_done",
    "frontend_done",
}


def _clean_text(value: str, field_name: str = "value") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _clean_string_list(values: list[str], field_name: str = "list") -> list[str]:
    if values is None:
        return []

    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must contain only strings")
        item = value.strip()
        if not item:
            raise ValueError(f"{field_name} must not contain empty strings")
        cleaned.append(item)

    return cleaned


class StrictArtifactModel(BaseModel):
    """
    Shared base model.

    extra='forbid' is useful in production because it prevents the model
    from silently returning fields that your app does not understand.
    """

    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
        "validate_assignment": True,
    }


# ============================================================
# PO output schema
# ============================================================

class StoryOut(StrictArtifactModel):
    story_id: str = Field(
        description="Stable user story id. Must use format US-001, US-002, etc."
    )
    title: str = Field(
        description="Short user story title."
    )
    description: str = Field(
        description='Delivery-ready user story, preferably using: "As a ..., I want ..., so that ...".'
    )
    priority: Priority = Field(
        default=Priority.MEDIUM,
        description="Business or delivery priority."
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Specific, testable acceptance criteria for this user story."
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Smallest reasonable assumptions made because the BRD was incomplete or ambiguous."
    )
    source_references: list[str] = Field(
        default_factory=list,
        description="Optional references to BRD sections, headings, bullets, or quotes that support this story."
    )

    @field_validator("story_id")
    @classmethod
    def validate_story_id(cls, value: str) -> str:
        value = _clean_text(value, "story_id")
        if not _ID_PATTERNS["story_id"].match(value):
            raise ValueError("story_id must use format US-001")
        return value

    @field_validator("title", "description")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _clean_text(value)

    @field_validator("acceptance_criteria", "assumptions", "source_references")
    @classmethod
    def validate_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_string_list(values)


class BacklogItemOut(StrictArtifactModel):
    backlog_item_id: str = Field(
        description="Stable backlog item id. Must use format BL-001, BL-002, etc."
    )
    title: str = Field(
        description="Short backlog item title."
    )
    description: str = Field(
        description="Concrete description of the backlog item."
    )
    team: Team = Field(
        default=Team.UNKNOWN,
        description="Primary team or implementation domain responsible for this backlog item."
    )
    related_user_story_ids: list[str] = Field(
        default_factory=list,
        description="User story ids covered by this backlog item."
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions relevant to this backlog item."
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Optional high-level acceptance criteria for this backlog item."
    )
    source_references: list[str] = Field(
        default_factory=list,
        description="Optional references to BRD sections supporting this backlog item."
    )

    @field_validator("backlog_item_id")
    @classmethod
    def validate_backlog_item_id(cls, value: str) -> str:
        value = _clean_text(value, "backlog_item_id")
        if not _ID_PATTERNS["backlog_item_id"].match(value):
            raise ValueError("backlog_item_id must use format BL-001")
        return value

    @field_validator("title", "description")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _clean_text(value)

    @field_validator("related_user_story_ids")
    @classmethod
    def validate_related_user_story_ids(cls, values: list[str]) -> list[str]:
        values = _clean_string_list(values, "related_user_story_ids")
        invalid = [value for value in values if not _ID_PATTERNS["story_id"].match(value)]
        if invalid:
            raise ValueError(f"invalid related_user_story_ids: {invalid}")
        return values

    @field_validator("assumptions", "acceptance_criteria", "source_references")
    @classmethod
    def validate_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_string_list(values)


class TaskOut(StrictArtifactModel):
    task_id: str = Field(
        description="Stable implementation task id. Must use format TASK-001, TASK-002, etc."
    )
    title: str = Field(
        description="Short implementation task title."
    )
    description: str = Field(
        description="Required non-empty description explaining the concrete implementation work in 1-3 sentences."
    )
    assignee_team: Team = Field(
        description="Exactly one team responsible for implementing this task."
    )
    related_user_story_ids: list[str] = Field(
        default_factory=list,
        description="User story ids this task implements."
    )
    related_backlog_item_ids: list[str] = Field(
        default_factory=list,
        description="Backlog item ids this task belongs to."
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Task-specific acceptance criteria. Must be concrete and testable."
    )
    required_markers: list[str] = Field(
        default_factory=list,
        description="Concrete snake_case markers that DEV and QC can verify."
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions relevant to this implementation task."
    )
    input_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Required non-empty machine-readable context for downstream DEV agent."
    )

    @field_validator("input_context")
    @classmethod
    def validate_input_context(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("input_context must be a dictionary")
        if len(value) < 3:
            raise ValueError("input_context must contain at least 3 concrete keys")

        # Allow empty lists/dicts for specific keys that commonly have no items
        keys_allowing_empty = {
            "dependencies",
            "out_of_scope",
            "expected_inputs",
            "expected_outputs",
            "open_questions",
        }

        for k, v in value.items():
            if not str(k).strip():
                raise ValueError("input_context contains empty key")
            # Only block if value is truly null/empty-string, OR if it's empty list/dict AND not in allowed set
            if v in (None, "") or (not v and k not in keys_allowing_empty):
                raise ValueError(f"input_context key '{k}' has invalid empty value: {v}")

        return value

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        value = _clean_text(value, "task_id")
        if not _ID_PATTERNS["task_id"].match(value):
            raise ValueError("task_id must use format TASK-001")
        return value

    @field_validator("title", "description")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _clean_text(value)

    @field_validator("related_user_story_ids")
    @classmethod
    def validate_related_user_story_ids(cls, values: list[str]) -> list[str]:
        values = _clean_string_list(values, "related_user_story_ids")
        invalid = [value for value in values if not _ID_PATTERNS["story_id"].match(value)]
        if invalid:
            raise ValueError(f"invalid related_user_story_ids: {invalid}")
        return values

    @field_validator("related_backlog_item_ids")
    @classmethod
    def validate_related_backlog_item_ids(cls, values: list[str]) -> list[str]:
        values = _clean_string_list(values, "related_backlog_item_ids")
        invalid = [value for value in values if not _ID_PATTERNS["backlog_item_id"].match(value)]
        if invalid:
            raise ValueError(f"invalid related_backlog_item_ids: {invalid}")
        return values

    @field_validator("acceptance_criteria", "assumptions")
    @classmethod
    def validate_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_string_list(values)

    @field_validator("required_markers")
    @classmethod
    def validate_required_markers(cls, values: list[str]) -> list[str]:
        values = _clean_string_list(values, "required_markers")
        invalid = [value for value in values if not _MARKER_PATTERN.match(value)]
        if invalid:
            raise ValueError(
                f"invalid required_markers: {invalid}. "
                "Markers must be snake_case identifiers with no spaces."
            )

        generic = [v for v in values if v in GENERIC_MARKERS_BLACKLIST]
        if generic:
            raise ValueError(
                f"Banned generic markers found: {generic}. "
                "Please use specific markers like 'user_login_api_returns_401_for_invalid_password'."
            )
        return values


class POResult(StrictArtifactModel):
    feature_summary: str = Field(
        default="",
        description="Summary of the analyzed features."
    )
    user_stories: list[StoryOut] = Field(default_factory=list)
    backlog_items: list[BacklogItemOut] = Field(default_factory=list)
    implementation_tasks: list[TaskOut] = Field(default_factory=list)
    resolution_map: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of review issue_id or category to resolution description."
    )

    @field_validator("feature_summary")
    @classmethod
    def validate_feature_summary(cls, value: str) -> str:
        if value is None:
            return ""
        return value.strip()

    @model_validator(mode="after")
    def validate_cross_references(self) -> "POResult":
        story_ids = {story.story_id for story in self.user_stories}
        backlog_ids = {item.backlog_item_id for item in self.backlog_items}

        duplicate_story_ids = _find_duplicates([story.story_id for story in self.user_stories])
        duplicate_backlog_ids = _find_duplicates([item.backlog_item_id for item in self.backlog_items])
        duplicate_task_ids = _find_duplicates([task.task_id for task in self.implementation_tasks])

        if duplicate_story_ids:
            raise ValueError(f"duplicate story ids: {duplicate_story_ids}")
        if duplicate_backlog_ids:
            raise ValueError(f"duplicate backlog item ids: {duplicate_backlog_ids}")
        if duplicate_task_ids:
            raise ValueError(f"duplicate task ids: {duplicate_task_ids}")

        for item in self.backlog_items:
            missing = [sid for sid in item.related_user_story_ids if sid not in story_ids]
            if missing:
                raise ValueError(
                    f"{item.backlog_item_id} references unknown user stories: {missing}"
                )

        for task in self.implementation_tasks:
            missing_stories = [
                sid for sid in task.related_user_story_ids if sid not in story_ids
            ]
            missing_backlog_items = [
                bid for bid in task.related_backlog_item_ids if bid not in backlog_ids
            ]

            if missing_stories:
                raise ValueError(
                    f"{task.task_id} references unknown user stories: {missing_stories}"
                )

            if missing_backlog_items:
                raise ValueError(
                    f"{task.task_id} references unknown backlog items: {missing_backlog_items}"
                )

        return self


def _find_duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)

    return sorted(duplicates)


# ============================================================
# PO review schema
# ============================================================

class POReviewIssue(StrictArtifactModel):
    category: ReviewIssueCategory = Field(
        description="Category of the issue."
    )
    severity: Severity = Field(
        default=Severity.MEDIUM,
        description="Issue severity."
    )
    description: str = Field(
        description="Clear description of what is wrong."
    )
    affected_items: list[str] = Field(
        default_factory=list,
        description="Identifiers of affected stories, backlog items, tasks, criteria, or markers."
    )
    suggestion: str = Field(
        default="",
        description="Actionable suggestion for fixing this issue."
    )

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return _clean_text(value, "description")

    @field_validator("affected_items")
    @classmethod
    def validate_affected_items(cls, values: list[str]) -> list[str]:
        return _clean_string_list(values, "affected_items")

    @field_validator("suggestion")
    @classmethod
    def validate_suggestion(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("suggestion cannot be empty. Provide actionable steps to fix the issue.")
        return cleaned


class POReviewResult(StrictArtifactModel):
    decision: ReviewDecision = Field(
        description="Review decision. PASS only when no issues exist."
    )
    issues: list[POReviewIssue] = Field(
        default_factory=list,
        description="List of validation issues found."
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="General actionable improvement suggestions."
    )
    summary: str = Field(
        default="",
        description="Brief summary of the review outcome."
    )

    @field_validator("suggestions")
    @classmethod
    def validate_suggestions(cls, values: list[str]) -> list[str]:
        return _clean_string_list(values, "suggestions")

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        if value is None:
            return ""
        return value.strip()

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> "POReviewResult":
        if self.decision == ReviewDecision.PASS and self.issues:
            raise ValueError("decision cannot be PASS when issues are present")

        if self.decision == ReviewDecision.NEEDS_REVISION and not self.issues:
            raise ValueError("decision NEEDS_REVISION must include at least one issue")

        return self


# ============================================================
# DEV output schema
# ============================================================

class ImplementedFile(StrictArtifactModel):
    file_path: str = Field(
        description=(
            "Relative file path inside the project, e.g. "
            "'app/api/orders.py', 'app/services/auth_service.py'."
        )
    )
    code: str = Field(
        description="Complete file content. No markdown fences."
    )

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, value: str) -> str:
        value = _clean_text(value, "file_path")

        if "\\" in value:
            value = value.replace("\\", "/")

        if value.startswith("/") or value.startswith("../") or "/../" in value:
            raise ValueError("file_path must be a safe relative path")

        if "/" not in value:
            raise ValueError("file_path must include a subdirectory, e.g. app/services/foo.py")

        return value

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _clean_text(value, "code")


class TestFile(StrictArtifactModel):
    file_path: str = Field(
        description="Relative test file path, e.g. 'tests/test_orders.py'."
    )
    code: str = Field(
        description="Complete test file content. No markdown fences."
    )

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, value: str) -> str:
        value = _clean_text(value, "file_path")

        if "\\" in value:
            value = value.replace("\\", "/")

        if value.startswith("/") or value.startswith("../") or "/../" in value:
            raise ValueError("file_path must be a safe relative path")

        if "/" not in value:
            raise ValueError("file_path must include a subdirectory, e.g. tests/test_foo.py")

        return value

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _clean_text(value, "code")


class DevResult(StrictArtifactModel):
    task_id: str = Field(
        description="Task id this implementation satisfies, e.g. TASK-001."
    )
    files: list[ImplementedFile] = Field(
        default_factory=list,
        description="Implementation files created or modified."
    )
    unit_tests: list[TestFile] = Field(
        default_factory=list,
        description="Unit test files for the implementation."
    )
    implementation_notes: str = Field(
        default="",
        description="Detailed notes mapping implementation to acceptance criteria, assumptions, markers, and bug fixes."
    )
    included_markers: list[str] = Field(
        default_factory=list,
        description="Only markers actually implemented and evidenced."
    )
    known_limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations caused by missing context or scope boundaries."
    )

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        value = _clean_text(value, "task_id")
        if not _ID_PATTERNS["task_id"].match(value):
            raise ValueError("task_id must use format TASK-001")
        return value

    @field_validator("implementation_notes")
    @classmethod
    def validate_implementation_notes(cls, value: str) -> str:
        if value is None:
            return ""
        return value.strip()

    @field_validator("included_markers")
    @classmethod
    def validate_included_markers(cls, values: list[str]) -> list[str]:
        values = _clean_string_list(values, "included_markers")
        invalid = [value for value in values if not _MARKER_PATTERN.match(value)]
        if invalid:
            raise ValueError(
                f"invalid included_markers: {invalid}. "
                "Markers must be snake_case identifiers with no spaces."
            )

        generic = [v for v in values if v in GENERIC_MARKERS_BLACKLIST]
        if generic:
            raise ValueError(f"Banned generic markers found in included_markers: {generic}")

        return values

    @field_validator("known_limitations")
    @classmethod
    def validate_known_limitations(cls, values: list[str]) -> list[str]:
        return _clean_string_list(values, "known_limitations")

    @model_validator(mode="after")
    def validate_dev_output_has_content(self) -> "DevResult":
        # Allow task with only files, only unit_tests, or both.
        if not self.files and not self.unit_tests:
            raise ValueError("DevResult must include at least one implementation file or unit test file")

        return self


# ============================================================
# QC output schema
# ============================================================

class CriterionValidationResult(StrictArtifactModel):
    criterion: str = Field(
        description="The acceptance criterion being evaluated."
    )
    status: CriterionStatus = Field(
        description="PASSED, FAILED, or UNSUPPORTED."
    )
    evidence: str = Field(
        default="",
        description="Specific evidence from code, tests, notes, or markers."
    )
    feedback: str = Field(
        default="",
        description="Actionable feedback if failed or unsupported."
    )

    @field_validator("criterion")
    @classmethod
    def validate_criterion(cls, value: str) -> str:
        return _clean_text(value, "criterion")

    @field_validator("evidence", "feedback")
    @classmethod
    def validate_optional_text(cls, value: str) -> str:
        if value is None:
            return ""
        return value.strip()


class MarkerValidationResult(StrictArtifactModel):
    marker: str = Field(
        description="The required marker being evaluated."
    )
    status: CriterionStatus = Field(
        description="PASSED, FAILED, or UNSUPPORTED."
    )
    evidence: str = Field(
        default="",
        description="Evidence from code, tests, notes, or included_markers."
    )
    feedback: str = Field(
        default="",
        description="Actionable feedback if marker is missing or unsupported."
    )

    @field_validator("marker")
    @classmethod
    def validate_marker(cls, value: str) -> str:
        value = _clean_text(value, "marker")
        if not _MARKER_PATTERN.match(value):
            raise ValueError("marker must be snake_case with no spaces")
        return value

    @field_validator("evidence", "feedback")
    @classmethod
    def validate_optional_text(cls, value: str) -> str:
        if value is None:
            return ""
        return value.strip()


class QCResult(StrictArtifactModel):
    task_id: str = Field(
        description="Task id being validated."
    )
    passed: bool = Field(
        description="True only if all acceptance criteria and required markers are satisfied with evidence."
    )
    status: QCStatus = Field(
        description="Validation status."
    )
    criteria_results: list[CriterionValidationResult] = Field(
        default_factory=list,
        description="Per-criterion validation results."
    )
    marker_results: list[MarkerValidationResult] = Field(
        default_factory=list,
        description="Per-marker validation results."
    )
    validation_report: str = Field(
        default="",
        description="Detailed validation report."
    )
    failed_criteria: list[str] = Field(
        default_factory=list,
        description="Criteria that failed or lacked evidence."
    )
    severity: Severity = Field(
        default=Severity.MEDIUM,
        description="Impact severity of validation findings."
    )

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        value = _clean_text(value, "task_id")
        if not _ID_PATTERNS["task_id"].match(value):
            raise ValueError("task_id must use format TASK-001")
        return value

    @field_validator("validation_report")
    @classmethod
    def validate_validation_report(cls, value: str) -> str:
        if value is None:
            return ""
        return value.strip()

    @field_validator("failed_criteria")
    @classmethod
    def validate_failed_criteria(cls, values: list[str]) -> list[str]:
        return _clean_string_list(values, "failed_criteria")

    @model_validator(mode="after")
    def validate_qc_consistency(self) -> "QCResult":
        if self.passed:
            if self.status != QCStatus.PASSED:
                raise ValueError("If passed is true, status must be PASSED")
            if self.failed_criteria:
                raise ValueError("If passed is true, failed_criteria must be empty")

            bad_criteria = [
                result.criterion
                for result in self.criteria_results
                if result.status != CriterionStatus.PASSED
            ]
            bad_markers = [
                result.marker
                for result in self.marker_results
                if result.status != CriterionStatus.PASSED
            ]

            if bad_criteria or bad_markers:
                raise ValueError(
                    "If passed is true, all criteria_results and marker_results must be PASSED"
                )

        if not self.passed:
            if self.status == QCStatus.PASSED:
                raise ValueError("If passed is false, status cannot be PASSED")

        return self
