from app.agents.base import QCResult


class MockQCAgent:
    def validate(self, task: dict, acceptance_criteria: list[str], dev_output: dict) -> QCResult:
        required_markers = task.get('required_markers', [])
        included_markers = dev_output.get('included_markers', [])
        missing = [marker for marker in required_markers if marker not in included_markers]
        if missing:
            return QCResult(
                passed=False,
                status='QC_FAILED',
                validation_report='Validation failed because implementation is missing required markers.',
                failed_criteria=missing,
                severity='HIGH',
            )
        return QCResult(
            passed=True,
            status='QC_PASSED',
            validation_report='All acceptance criteria are satisfied.',
            failed_criteria=[],
            severity='LOW',
        )
