from app.agents.base import DevResult, ImplementedFile, TestFile


class MockDevAgent:
    def implement(
        self,
        task: dict,
        acceptance_criteria: list[str],
        bug_reports: list[dict] | None = None,
        project_context: str | None = None,
        current_project_snapshot: dict[str, str] | None = None,
    ) -> DevResult:
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
        unit_tests_code = '\n'.join([
            'def test_generated_feature():',
            '    result = generated_feature()',
            *[f"    assert result['{marker}'] is True" for marker in included_markers],
        ])

        # Create nested objects
        files = [
            ImplementedFile(file_path="app/generated_feature.py", code=code)
        ]
        unit_tests = [
            TestFile(file_path="tests/test_generated_feature.py", code=unit_tests_code)
        ]

        return DevResult(
            task_id=task.get('task_id', 'TASK-001'),
            files=files,
            unit_tests=unit_tests,
            implementation_notes=note + extra,
            included_markers=included_markers,
            known_limitations=[]
        )
