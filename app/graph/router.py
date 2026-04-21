from app.core.enums import WorkflowStatus
from app.graph.state import WorkflowState


def route_by_qc_result(state: WorkflowState) -> str:
    qc_result = state.get('qc_result') or {}
    if qc_result.get('passed'):
        return 'pass'
    retry_count = int(state.get('retry_count', 0))
    max_retry = int(state.get('max_retry', 0))
    if retry_count < max_retry:
        return 'retry'
    return 'max_retry'


def route_after_task_resolution(state: WorkflowState) -> str:
    return 'has_more' if state.get('task_queue', []) else 'finished'


def route_start(state: WorkflowState) -> str:
    """Determine the entry node based on the current workflow status."""
    status = state.get('status')

    if status == WorkflowStatus.NEW.value:
        return 'ingest_brd'
    if status == WorkflowStatus.PO_ANALYZING.value:
        # If it was analyzing, we might need to re-analyze or skip to story creation if result exists.
        # For simplicity, we restart analysis to ensure consistency.
        return 'po_analyze_brd'
    if status == WorkflowStatus.BACKLOG_CREATED.value:
        return 'dispatch_to_dev'
    if status in [
        WorkflowStatus.TASK_READY_FOR_DEV.value,
        WorkflowStatus.DEV_IN_PROGRESS.value,
        WorkflowStatus.REOPENED_FOR_DEV.value,
        WorkflowStatus.BUG_CREATED.value
    ]:
        return 'dev_implement'
    if status == WorkflowStatus.DEV_DONE.value:
        return 'dispatch_to_qc'
    if status == WorkflowStatus.QC_IN_PROGRESS.value:
        return 'qc_validate'
    
    # Default fallback
    return 'ingest_brd'
