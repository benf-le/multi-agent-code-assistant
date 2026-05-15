from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    workflow_id: int
    brd_id: int
    brd_content: str
    feature_summary: str
    user_stories: list[dict[str, Any]]
    acceptance_criteria: list[dict[str, Any]]
    backlog: list[dict[str, Any]]
    po_result: dict[str, Any]
    po_review_result: dict[str, Any] | None
    po_review_retries: int
    current_task: dict[str, Any] | None
    task_queue: list[dict[str, Any]]
    task_history: list[dict[str, Any]]
    blocked_tasks: list[dict[str, Any]]
    completed_tasks: list[dict[str, Any]]
    dev_output: dict[str, Any] | None
    dependency_result: dict[str, Any] | None
    build_result: dict[str, Any] | None
    qc_result: dict[str, Any] | None
    bug_reports: list[dict[str, Any]]
    retry_count: int
    max_retry: int
    current_agent: str
    status: str
    event_logs: list[dict[str, Any]]
    timestamps: dict[str, Any]
    # Tracing
    loop_signatures: list[str]
    visited_nodes: list[str]
    route_decisions: list[dict[str, Any]]
    po_local_issues: list[dict[str, Any]]
    po_review_issues_history: list[dict[str, Any]]
    # Project state — in-memory project tracking
    current_project: dict[str, str]       # committed/accepted project state
    candidate_project: dict[str, str]     # temporary state after applying task, used for QC
    
    # Final Project QA State
    final_project_qa_status: str | None
    final_project_qa_attempts: int
    final_project_qa_report: dict[str, Any] | None
    final_project_fix_task: dict[str, Any] | None
    final_project_build_logs: list[dict[str, Any]] | None
    final_project_attempted_fixes: list[str] | None

