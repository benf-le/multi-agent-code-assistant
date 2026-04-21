from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    workflow_id: int
    brd_id: int
    brd_content: str
    feature_summary: str
    user_stories: list[dict[str, Any]]
    acceptance_criteria: list[dict[str, Any]]
    backlog: list[dict[str, Any]]
    current_task: dict[str, Any] | None
    task_queue: list[dict[str, Any]]
    task_history: list[dict[str, Any]]
    dev_output: dict[str, Any] | None
    qc_result: dict[str, Any] | None
    bug_reports: list[dict[str, Any]]
    retry_count: int
    max_retry: int
    current_agent: str
    status: str
    event_logs: list[dict[str, Any]]
    timestamps: dict[str, Any]
    blocked_tasks: list[dict[str, Any]]
    completed_tasks: list[dict[str, Any]]
    po_result: dict[str, Any]
