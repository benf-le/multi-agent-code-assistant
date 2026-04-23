from app.core.config import get_settings
from app.graph.state import WorkflowState


def route_by_qc_result(state: WorkflowState) -> str:
    """Route after QC validation within a single-task graph run.

    Returns one of:
      'pass'       — QC passed, task is done
      'retry'      — QC failed but retries remain
      'max_retry'  — QC failed and retries exhausted
      'loop_block' — loop detector triggered (no progress detected)
    """
    qc_result = state.get('qc_result') or {}
    if qc_result.get('passed'):
        return 'pass'

    retry_count = int(state.get('retry_count', 0))
    max_retry = int(state.get('max_retry', 0))

    if retry_count >= max_retry:
        return 'max_retry'

    # --- Loop detection ---
    # Build a signature for the current state of this task
    current_task = state.get('current_task') or {}
    signature = f"{current_task.get('id')}:{retry_count}:{qc_result.get('validation_report', '')[:80]}"
    loop_sigs = state.get('loop_signatures', [])

    settings = get_settings()
    count = sum(1 for s in loop_sigs if s == signature)
    if count >= settings.max_loop_signatures:
        return 'loop_block'

    return 'retry'
