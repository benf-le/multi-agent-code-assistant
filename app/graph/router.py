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


def route_by_build_result(state: WorkflowState) -> str:
    """Route after build verification within a single-task graph run."""
    build_result = state.get('build_result') or {}
    current_task = state.get('current_task') or {}
    task_id = current_task.get('id')
    retry_count = int(state.get('retry_count', 0))
    max_retry = int(state.get('max_retry', 0))
    
    if build_result.get('passed'):
        route = 'pass'
        reason = "Build passed successfully."
    elif retry_count >= max_retry:
        route = 'max_retry'
        reason = f"Build failed and max retries ({max_retry}) reached."
    else:
        route = 'retry'
        reason = "Build failed, retries remain."

    WorkflowLogger.info("workflow.route.build_decision",
        workflow_id=state.get('workflow_id'),
        task_id=task_id,
        task_number=current_task.get('task_number'),
        route=route,
        reason=reason,
        retry_count=retry_count,
        max_retry=max_retry
    )
    
    state['route_decisions'] = [
        *state.get('route_decisions', []),
        {'node': 'build_candidate', 'route': route, 'reason': reason}
    ]

    return route


def route_by_po_review(state: WorkflowState) -> str:
    """Route after PO review validation within the PO phase graph.

    Returns:
        'pass'   — review passed, continue to END
        'retry'  — review failed, retries remain, loop back to po_analyze_brd
        'fail'   — review failed, retries exhausted, go to po_review_failed
    """
    po_review = state.get('po_review_result') or {}
    decision = po_review.get('decision', 'NEEDS_REVISION')
    retries = int(state.get('po_review_retries', 0))
    settings = get_settings()
    max_retries = settings.po_review_max_retries

    route = 'fail'
    reason = 'PO review failed with no retries remaining.'

    if decision == 'PASS':
        route = 'pass'
        reason = 'PO review passed all validation checks.'
    elif retries < max_retries:
        route = 'retry'
        reason = f'PO review failed (attempt {retries + 1}/{max_retries}), retrying PO phase.'
    else:
        reason = f'PO review failed after {retries} retries. Max retries ({max_retries}) exhausted.'

    # Log the decision
    WorkflowLogger.info("workflow.route.po_review_decision",
        workflow_id=state.get('workflow_id'),
        route=route,
        reason=reason,
        decision=decision,
        retries=retries,
        max_retries=max_retries,
        issues_count=len(po_review.get('issues', []))
    )

    return route


def route_by_po_local_validate(state: WorkflowState) -> str:
    """Route after PO local rule-based validation.

    Returns:
        'pass'   — local validation passed, continue to po_review
        'retry'  — local validation failed, retries remain, loop back to po_analyze_brd
        'fail'   — local validation failed, retries exhausted, go to po_review_failed
    """
    local_issues = state.get('po_local_issues', [])
    retries = int(state.get('po_review_retries', 0))
    settings = get_settings()
    max_retries = settings.po_review_max_retries

    route = 'pass'
    reason = 'PO local validation passed all rule-based checks.'

    if local_issues:
        if retries < max_retries:
            route = 'retry'
            reason = f'PO local validation failed (attempt {retries}/{max_retries}), retrying PO phase.'
        else:
            route = 'fail'
            reason = f'PO local validation failed after {retries} retries. Max retries ({max_retries}) exhausted.'

    # Log the decision
    WorkflowLogger.info("workflow.route.po_local_decision",
        workflow_id=state.get('workflow_id'),
        route=route,
        reason=reason,
        retries=retries,
        max_retries=max_retries,
        issues_count=len(local_issues)
    )

    # Update trace
    state['route_decisions'] = [
        *state.get('route_decisions', []),
        {'node': 'po_local_validate', 'route': route, 'reason': reason}
    ]

    return route
