from app.agents.po_agent import MockPOAgent
from app.agents.base import POResult


def test_brd_to_user_stories_and_backlog():
    brd = "- users can register with email and password\n- users can log in with valid credentials\n- users can update profile information\n"
    result = MockPOAgent().analyze(brd)
    assert len(result.user_stories) >= 2
    assert len(result.backlog_items) == len(result.implementation_tasks)
    assert 'register' in result.feature_summary


def test_po_result_accepts_resolution_map_list_values():
    result = POResult.model_validate(
        {
            "feature_summary": "Registration",
            "user_stories": [],
            "backlog_items": [],
            "implementation_tasks": [],
            "resolution_map": {
                "TASK-001": [
                    "Revised AC-4 to specify observable output.",
                    "Added installation and usage notes.",
                ],
            },
        }
    )

    assert result.resolution_map == {
        "TASK-001": "Revised AC-4 to specify observable output.; Added installation and usage notes."
    }
