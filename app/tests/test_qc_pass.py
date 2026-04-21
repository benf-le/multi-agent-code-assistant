from app.agents.qc_agent import MockQCAgent


def test_qc_pass_done():
    task = {'required_markers': ['register', 'login']}
    dev_output = {'included_markers': ['register', 'login']}
    result = MockQCAgent().validate(task, acceptance_criteria=['a', 'b'], dev_output=dev_output)
    assert result.passed is True
    assert result.status == 'QC_PASSED'
