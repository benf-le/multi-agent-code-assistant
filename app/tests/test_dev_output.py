from app.agents.dev_agent import MockDevAgent


def test_dev_output_contains_code_and_tests():
    task = {'title': 'Backend implementation task 1', 'required_markers': ['register', 'login'], 'input_context': {'expected_failures': 0}, 'retry_count': 0}
    result = MockDevAgent().implement(task, acceptance_criteria=['a', 'b'], bug_reports=[])
    assert 'def generated_feature' in result.code
    assert 'def test_generated_feature' in result.unit_tests
    assert set(result.included_markers) == {'register', 'login'}
