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
