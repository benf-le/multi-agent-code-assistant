import pytest
from app.graph.nodes import WorkflowNodes, should_skip_db_or_migration_check
from app.graph.state import WorkflowState
from app.core.enums import WorkflowStatus
from unittest.mock import MagicMock

def test_ensure_dependencies_integer_task_id():
    orchestrator = MagicMock()
    orchestrator.check_cancellation.return_value = False
    nodes = WorkflowNodes(orchestrator, MagicMock(), MagicMock(), MagicMock(), MagicMock())

    state: WorkflowState = {
        'workflow_id': 1,
        'current_task': {
            'id': 1,  # Integer ID, which caused AttributeError
            'task_number': 1,
            'title': 'Project Foundation'
        },
        'dev_output': {},
        'candidate_project': {}
    }

    # Should not crash with AttributeError
    updated_state = nodes.ensure_dependencies(state)
    assert updated_state['dependency_result']['skipped'] is True
    assert updated_state['status'] == WorkflowStatus.DEV_DONE.value

def test_ensure_dependencies_validation_failure():
    orchestrator = MagicMock()
    orchestrator.check_cancellation.return_value = False
    nodes = WorkflowNodes(orchestrator, MagicMock(), MagicMock(), MagicMock(), MagicMock())

    state: WorkflowState = {
        'workflow_id': 2,
        'current_task': {
            'id': 2,
            'task_number': 2,
            'title': 'Implement API'
        },
        'dev_output': {
            'dependencies': [123, 456]  # Bad format, should fail validation
        },
        'candidate_project': {}
    }

    updated_state = nodes.ensure_dependencies(state)
    assert updated_state['status'] == WorkflowStatus.DEPENDENCY_FAILED.value
    assert updated_state['dependency_result']['category'] == 'validation_error'
    assert "must contain only strings" in updated_state['dependency_result']['error_summary']

def test_should_skip_db_or_migration_check():
    assert should_skip_db_or_migration_check("migrations/001_initial.py") is True
    assert should_skip_db_or_migration_check("app/alembic/env.py") is True
    assert should_skip_db_or_migration_check("scripts/seed_database.py") is True
    assert should_skip_db_or_migration_check("init_db.sql") is True
    assert should_skip_db_or_migration_check("app/models/user.py") is False
    assert should_skip_db_or_migration_check("app/main.py") is False
    assert should_skip_db_or_migration_check(123) is False

def test_ensure_dependencies_db_only_task():
    orchestrator = MagicMock()
    orchestrator.check_cancellation.return_value = False
    nodes = WorkflowNodes(orchestrator, MagicMock(), MagicMock(), MagicMock(), MagicMock())

    state: WorkflowState = {
        'workflow_id': 3,
        'current_task': {
            'id': 3,
            'task_number': 3,
            'title': 'Create initial schema'
        },
        'dev_output': {
            'files': [
                {'file_path': 'migrations/001_initial.py', 'code': '# code'},
                {'file_path': 'init_db.sql', 'code': 'CREATE TABLE;'}
            ]
        },
        'candidate_project': {}
    }

    updated_state = nodes.ensure_dependencies(state)
    assert updated_state['dependency_result']['skipped_gate'] is True
    assert updated_state['status'] == WorkflowStatus.DEV_DONE.value
    assert "only modifies DB/migration files" in updated_state['dependency_result']['error_summary']
