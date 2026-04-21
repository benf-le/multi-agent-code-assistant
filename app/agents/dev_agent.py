from app.agents.base import DevResult


class MockDevAgent:
    def implement(self, task: dict, acceptance_criteria: list[str], bug_reports: list[dict] | None = None) -> DevResult:
        bug_reports = bug_reports or []
        markers = list(task.get('required_markers', []))
        expected_failures = int(task.get('input_context', {}).get('expected_failures', 0))
        retry_count = int(task.get('retry_count', 0))

        if retry_count < expected_failures and len(markers) > 1:
            included_markers = markers[:-1]
            note = 'Initial implementation skipped one marker to simulate QC loop.'
        else:
            included_markers = markers
            note = 'Implementation includes all required markers and latest bug fixes.'

        extra = ''
        if bug_reports:
            extra = f" Latest bug addressed: {bug_reports[-1]['title']}"

        code = '\n'.join([
            f"# task: {task['title']}",
            'def generated_feature():',
            '    return {',
            *[f"        '{marker}': True," for marker in included_markers],
            '    }',
        ])
        unit_tests = '\n'.join([
            'def test_generated_feature():',
            '    result = generated_feature()',
            *[f"    assert result['{marker}'] is True" for marker in included_markers],
        ])
        return DevResult(code=code, unit_tests=unit_tests, implementation_notes=note + extra, included_markers=included_markers)
