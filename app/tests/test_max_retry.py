from app.repositories.task_repository import TaskRepository
from app.repositories.workflow_repository import WorkflowRepository


def test_max_retry_exceeded_blocks_workflow(workflow_service, session_factory):
    workflow_id = workflow_service.create_workflow_from_brd(
        title='Test Max Retry',
        brd_content="- a\n- b\n",
        max_retry=0,
    )
    result = workflow_service.run_workflow(workflow_id)
    assert result['status'] == 'BLOCKED'

    with session_factory() as session:
        workflow = WorkflowRepository(session).get(workflow_id)
        tasks = TaskRepository(session).list_tasks(workflow_id)
        assert workflow is not None
        assert workflow.status == 'BLOCKED'
        assert any(task.status == 'BLOCKED' for task in tasks)
