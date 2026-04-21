from app.repositories.audit_repository import AuditRepository
from app.repositories.task_repository import TaskRepository
from app.repositories.workflow_repository import WorkflowRepository


def test_qc_fail_then_loop_back_to_dev_then_pass(workflow_service, session_factory):
    workflow_id = workflow_service.create_workflow_from_brd(
        title='Test BRD',
        brd_content="- users can register\n- users can log in\n- users can update profile\n",
        max_retry=2,
    )
    result = workflow_service.run_workflow(workflow_id)
    assert result['status'] == 'DONE'
    assert len(result['bug_reports']) >= 1

    with session_factory() as session:
        workflow = WorkflowRepository(session).get(workflow_id)
        tasks = TaskRepository(session).list_tasks(workflow_id)
        transitions = AuditRepository(session).list_transitions(workflow_id=workflow_id)
        assert workflow is not None
        assert workflow.status == 'DONE'
        assert any(task.retry_count >= 1 for task in tasks)
        assert any(t.to_status == 'BUG_CREATED' for t in transitions)
        assert any(t.to_status == 'QC_PASSED' for t in transitions)
