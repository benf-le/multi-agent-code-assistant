from app.core.config import get_settings
from app.graph.state import WorkflowState
from app.core.logging_helper import WorkflowLogger


def route_by_qc_result(state: WorkflowState) -> str:
    """Route after QC validation within a single-task graph run."""
    qc_result = state.get('qc_result') or {}
    current_task = state.get('current_task') or {}
    task_id = current_task.get('id')
    retry_count = int(state.get('retry_count', 0))
    max_retry = int(state.get('max_retry', 0))
    
    route = 'retry'
    reason = "QC failed, retries remain."

    if qc_result.get('passed'):
        route = 'pass'
        reason = "QC passed successfully."
    elif retry_count >= max_retry:
        route = 'max_retry'
        reason = f"QC failed and max retries ({max_retry}) reached."
    else:
        # Loop detection
        signature = f"{task_id}:{retry_count}:{qc_result.get('validation_report', '')[:80]}"
        loop_sigs = state.get('loop_signatures', [])
        settings = get_settings()
        count = sum(1 for s in loop_sigs if s == signature)
        if count >= settings.max_loop_signatures:
            route = 'loop_block'
            reason = f"Loop detected: signature seen {count} times."

    # Log the decision
    WorkflowLogger.info("workflow.route.decision",
        workflow_id=state.get('workflow_id'),
        task_id=task_id,
        task_number=current_task.get('task_number'),
        route=route,
        reason=reason,
        retry_count=retry_count,
        max_retry=max_retry
    )
    
    # Update trace
    state['route_decisions'] = [
        *state.get('route_decisions', []),
        {'node': 'qc_validate', 'route': route, 'reason': reason}
    ]

    return route
