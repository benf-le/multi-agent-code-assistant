from app.agents.po_agent import MockPOAgent


def test_brd_to_user_stories_and_backlog():
    brd = "- users can register with email and password\n- users can log in with valid credentials\n- users can update profile information\n"
    result = MockPOAgent().analyze(brd)
    assert len(result.user_stories) >= 2
    assert len(result.backlog_items) == len(result.implementation_tasks)
    assert 'register' in result.feature_summary
