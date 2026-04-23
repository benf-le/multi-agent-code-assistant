from typing import Callable
from copy import deepcopy
import os
from pathlib import Path

from app.agents.openai_agents import POAgent, DevAgent, QCAgent
from app.core.enums import AgentName, TaskStatus, WorkflowStatus
from app.graph.state import WorkflowState
from app.services.orchestrator_service import OrchestratorService


class WorkflowNodes:
    def __init__(self, orchestrator: OrchestratorService, po_agent: POAgent, dev_agent: DevAgent, qc_agent: QCAgent):
        self.orchestrator = orchestrator
        self.po_agent = po_agent
        self.dev_agent = dev_agent
        self.qc_agent = qc_agent

    def _check_cancelled(self, workflow_id: int):
        if self.orchestrator.check_cancellation(workflow_id):
            raise InterruptedError(f"Workflow {workflow_id} was cancelled by user.")

    # ──────────────────────────────────────────────────────────────────────
    #  PO Phase Nodes (used by build_po_graph)
    # ──────────────────────────────────────────────────────────────────────

    def ingest_brd(self, state: WorkflowState) -> WorkflowState:
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        updated['timestamps'] = {**updated.get('timestamps', {}), 'ingest_brd': self.orchestrator.now_iso()}
        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.NEW.value, AgentName.ORCHESTRATOR.value, 'BRD ingested into workflow state.')
        updated['status'] = WorkflowStatus.NEW.value
        updated['current_agent'] = AgentName.ORCHESTRATOR.value
        return updated

    def orchestrator_init(self, state: WorkflowState) -> WorkflowState:
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        updated['event_logs'] = updated.get('event_logs', [])
        updated['task_history'] = updated.get('task_history', [])
        updated['completed_tasks'] = updated.get('completed_tasks', [])
        updated['blocked_tasks'] = updated.get('blocked_tasks', [])
        updated['bug_reports'] = updated.get('bug_reports', [])
        updated['current_task'] = None
        updated['dev_output'] = None
        updated['qc_result'] = None
        return updated

    def po_analyze_brd(self, state: WorkflowState) -> WorkflowState:
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.PO_ANALYZING.value, AgentName.PO.value, 'PO agent is analyzing BRD.')
        self._check_cancelled(updated['workflow_id'])
        result = self.po_agent.analyze(updated['brd_content'])
        self.orchestrator.ctx.brd_repo.create_feature(updated['workflow_id'], result.feature_summary)
        self.orchestrator.record_agent_run(updated['workflow_id'], AgentName.PO.value, 'SUCCESS', input_payload={'brd_id': updated['brd_id']}, output_payload=result.model_dump())
        self.orchestrator.ctx.session.commit()
        updated['feature_summary'] = result.feature_summary
        updated['po_result'] = result.model_dump()
        updated['current_agent'] = AgentName.PO.value
        updated['status'] = WorkflowStatus.PO_ANALYZING.value
        return updated

    def po_create_user_stories(self, state: WorkflowState) -> WorkflowState:
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        po_result = updated['po_result']
        persisted_stories: list[dict] = []
        persisted_criteria: list[dict] = []
        for story in po_result['user_stories']:
            db_story = self.orchestrator.ctx.brd_repo.create_user_story(updated['workflow_id'], story['title'], story['description'], story['priority'])
            persisted_stories.append({'id': db_story.id, **story})
            for text in story['acceptance_criteria']:
                criterion = self.orchestrator.ctx.brd_repo.create_acceptance_criterion(updated['workflow_id'], text, user_story_id=db_story.id)
                persisted_criteria.append({'id': criterion.id, 'text': text, 'user_story_id': db_story.id})
        self.orchestrator.ctx.session.commit()
        updated['user_stories'] = persisted_stories
        updated['acceptance_criteria'] = persisted_criteria
        return updated

    def po_create_backlog_and_tasks(self, state: WorkflowState) -> WorkflowState:
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        po_result = updated['po_result']
        backlog_items = po_result['backlog_items']
        tasks = po_result['implementation_tasks']
        persisted_backlog: list[dict] = []
        persisted_tasks: list[dict] = []

        # Step 1: Create all backlog items first
        db_backlog_map: list[int] = []  # list of db backlog IDs by index
        for idx, backlog_item in enumerate(backlog_items, start=1):
            db_backlog = self.orchestrator.ctx.task_repo.create_backlog_item(updated['workflow_id'], backlog_item['title'], backlog_item['description'], backlog_item['team'], idx)
            persisted_backlog.append({'id': db_backlog.id, **backlog_item})
            db_backlog_map.append(db_backlog.id)

        # Step 2: Create tasks, linking each to a backlog item by index.
        # If there are more tasks than backlog items, extra tasks link to the last backlog item.
        for task_idx, task_data in enumerate(tasks):
            backlog_id = db_backlog_map[min(task_idx, len(db_backlog_map) - 1)] if db_backlog_map else None
            db_task = self.orchestrator.ctx.task_repo.create_task(
                workflow_id=updated['workflow_id'],
                backlog_item_id=backlog_id,
                title=task_data['title'],
                description=task_data['description'],
                assignee_team=task_data['assignee_team'],
                max_retry=updated['max_retry'],
                required_markers=task_data['required_markers'],
                input_context=task_data['input_context'],
            )
            for text in task_data['acceptance_criteria']:
                self.orchestrator.ctx.task_repo.attach_acceptance_criterion(updated['workflow_id'], db_task.id, text)
            persisted_tasks.append(self.orchestrator.serialize_task(db_task.id))

        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.BACKLOG_CREATED.value, AgentName.PO.value, 'PO agent created backlog and implementation tasks.')
        updated['backlog'] = persisted_backlog
        updated['task_queue'] = persisted_tasks
        updated['status'] = WorkflowStatus.BACKLOG_CREATED.value
        updated['current_agent'] = AgentName.PO.value
        updated.pop('po_result', None)
        return updated

    # ──────────────────────────────────────────────────────────────────────
    #  Task Execution Nodes (used by build_task_graph)
    # ──────────────────────────────────────────────────────────────────────

    def dispatch_to_dev(self, state: WorkflowState) -> WorkflowState:
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        current_task = updated['current_task']
        if current_task is None:
            raise ValueError("dispatch_to_dev called with no current_task — state was not properly initialized by the service layer.")
        task_display = f"t-{current_task['task_number']:03d}"
        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.TASK_READY_FOR_DEV.value, AgentName.ORCHESTRATOR.value, f"Task {task_display} dispatched to DEV.", task_id=current_task['id'])
        self.orchestrator.update_task_status(updated['workflow_id'], current_task['id'], TaskStatus.READY.value, AgentName.ORCHESTRATOR.value, 'Task prepared for DEV implementation.')
        updated['status'] = WorkflowStatus.TASK_READY_FOR_DEV.value
        updated['current_agent'] = AgentName.ORCHESTRATOR.value
        updated['retry_count'] = current_task['retry_count']
        return updated

    def dev_implement(self, state: WorkflowState) -> WorkflowState:
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        current_task = updated['current_task']
        task_display = f"t-{current_task['task_number']:03d}"
        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.DEV_IN_PROGRESS.value, AgentName.DEV.value, f"DEV is implementing task {task_display}.", task_id=current_task['id'])
        self.orchestrator.update_task_status(updated['workflow_id'], current_task['id'], TaskStatus.DEV_IN_PROGRESS.value, AgentName.DEV.value, f"DEV started implementation for {task_display}.")
        self._check_cancelled(updated['workflow_id'])
        bug_reports = self.orchestrator.ctx.task_repo.list_bugs_for_task(current_task['id'])
        dev_result = self.dev_agent.implement(
            task=current_task,
            acceptance_criteria=current_task['acceptance_criteria'],
            bug_reports=[{'id': b.id, 'title': b.title, 'description': b.description, 'failed_criteria': b.failed_criteria} for b in bug_reports],
        )
        output_payload = dev_result.model_dump()
        db_task = self.orchestrator.ctx.task_repo.get_task(current_task['id'])
        assert db_task is not None
        self.orchestrator.ctx.task_repo.save_task_output(db_task, output_payload)
        task_display = f"t-{current_task['task_number']:03d}"
        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.DEV_DONE.value, AgentName.DEV.value, f"DEV completed task {task_display}.", task_id=current_task['id'])
        self.orchestrator.update_task_status(updated['workflow_id'], current_task['id'], TaskStatus.DEV_DONE.value, AgentName.DEV.value, f"DEV finished implementation for {task_display}.")
        self.orchestrator.record_agent_run(updated['workflow_id'], AgentName.DEV.value, 'SUCCESS', task_id=current_task['id'], input_payload={'task': current_task}, output_payload=output_payload)
        current_task['output_context'] = output_payload
        
        # Save physical files
        self._save_physical_files(updated['workflow_id'], current_task, dev_result)

        updated['current_task'] = current_task
        updated['dev_output'] = output_payload
        updated['status'] = WorkflowStatus.DEV_DONE.value
        updated['current_agent'] = AgentName.DEV.value
        return updated

    def _save_physical_files(self, workflow_id: int, task: dict, dev_result):
        """Save all task outputs into a single unified project directory.
        
        All files are written to generated_code/wf_{id}/project/ using the
        file_path returned by the LLM (e.g. 'app/api/admin.py').  Unit tests
        are placed under tests/ and implementation notes under docs/.
        """
        try:
            project_dir = Path("generated_code") / f"wf_{workflow_id}" / "project"

            # ── 1. Main source file ──────────────────────────────────────────
            # file_path is something like 'app/api/admin.py'
            # Make sure it doesn't escape the project dir (basic sanitize)
            raw_path = dev_result.file_path.lstrip("/\\").replace("..", "")
            dest_file = project_dir / raw_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            dest_file.write_text(dev_result.code, encoding="utf-8")

            # ── 2. Unit tests ────────────────────────────────────────────────
            if dev_result.unit_tests:
                # Mirror the source path under tests/, e.g.
                #   app/api/admin.py  →  tests/app/api/test_admin.py
                parts = Path(raw_path).parts          # ('app', 'api', 'admin.py')
                stem = Path(parts[-1]).stem            # 'admin'
                suffix = Path(parts[-1]).suffix        # '.py'
                test_rel = Path("tests").joinpath(*parts[:-1]) / f"test_{stem}{suffix}"
                test_file = project_dir / test_rel
                test_file.parent.mkdir(parents=True, exist_ok=True)
                test_file.write_text(dev_result.unit_tests, encoding="utf-8")

            # ── 3. Implementation notes ──────────────────────────────────────
            if dev_result.implementation_notes:
                notes_rel = Path("docs") / f"{Path(raw_path).stem}_notes.md"
                notes_file = project_dir / notes_rel
                notes_file.parent.mkdir(parents=True, exist_ok=True)
                header = (
                    f"# Task t-{task['task_number']:03d}: {task.get('title', '')}\n\n"
                    f"**Source file:** `{raw_path}`\n\n"
                )
                notes_file.write_text(header + dev_result.implementation_notes, encoding="utf-8")

            print(f"[wf_{workflow_id}] Saved '{raw_path}' → project/")
        except Exception as e:
            print(f"[wf_{workflow_id}] Error saving physical files: {e}")

    def dispatch_to_qc(self, state: WorkflowState) -> WorkflowState:
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        current_task = updated['current_task']
        task_display = f"t-{current_task['task_number']:03d}"
        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.QC_IN_PROGRESS.value, AgentName.ORCHESTRATOR.value, f"Task {task_display} dispatched to QC.", task_id=current_task['id'])
        self.orchestrator.update_task_status(updated['workflow_id'], current_task['id'], TaskStatus.QC_IN_PROGRESS.value, AgentName.ORCHESTRATOR.value, f"Task {task_display} prepared for QC validation.")
        updated['status'] = WorkflowStatus.QC_IN_PROGRESS.value
        updated['current_agent'] = AgentName.ORCHESTRATOR.value
        return updated

    def qc_validate(self, state: WorkflowState) -> WorkflowState:
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        current_task = updated['current_task']
        self._check_cancelled(updated['workflow_id'])
        qc_result = self.qc_agent.validate(current_task, current_task['acceptance_criteria'], updated['dev_output'] or {})
        task_display = f"t-{current_task['task_number']:03d}"
        self.orchestrator.record_agent_run(updated['workflow_id'], AgentName.QC.value, 'SUCCESS', task_id=current_task['id'], input_payload={'task': current_task, 'dev_output': updated.get('dev_output')}, output_payload=qc_result.model_dump())
        self.orchestrator.update_workflow_status(updated['workflow_id'], qc_result.status, AgentName.QC.value, f"QC validated task {task_display}: {qc_result.status}", task_id=current_task['id'])
        task_status = TaskStatus.DONE.value if qc_result.passed else TaskStatus.QC_FAILED.value
        self.orchestrator.update_task_status(updated['workflow_id'], current_task['id'], task_status, AgentName.QC.value, qc_result.validation_report)
        updated['qc_result'] = qc_result.model_dump()
        updated['status'] = qc_result.status
        updated['current_agent'] = AgentName.QC.value

        # Record loop signature for detection (only on failure, since pass exits immediately)
        if not qc_result.passed:
            sig = f"{current_task.get('id')}:{updated.get('retry_count', 0)}:{qc_result.validation_report[:80] if qc_result.validation_report else ''}"
            updated['loop_signatures'] = [*updated.get('loop_signatures', []), sig]

        return updated

    def create_bug(self, state: WorkflowState) -> WorkflowState:
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        current_task = updated['current_task']
        qc_result = updated['qc_result'] or {}
        task_display = f"t-{current_task['task_number']:03d}"
        bug = self.orchestrator.ctx.task_repo.create_bug(
            workflow_id=updated['workflow_id'],
            task_id=current_task['id'],
            title=f"Bug for task {task_display}",
            description=qc_result.get('validation_report', 'QC validation failed.'),
            severity=qc_result.get('severity', 'MEDIUM'),
            failed_criteria=qc_result.get('failed_criteria', []),
        )
        db_task = self.orchestrator.ctx.task_repo.get_task(current_task['id'])
        assert db_task is not None
        self.orchestrator.ctx.task_repo.increment_retry(db_task)
        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.BUG_CREATED.value, AgentName.QC.value, f"QC created bug {bug.id} for task {task_display}.", task_id=current_task['id'])
        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.REOPENED_FOR_DEV.value, AgentName.ORCHESTRATOR.value, f"Task {task_display} reopened for DEV after bug {bug.id}.", task_id=current_task['id'])
        self.orchestrator.update_task_status(updated['workflow_id'], current_task['id'], TaskStatus.REOPENED.value, AgentName.ORCHESTRATOR.value, f"Task {task_display} reopened for DEV re-implementation.")
        current_task = self.orchestrator.serialize_task(current_task['id'])
        updated['current_task'] = current_task
        updated['bug_reports'] = [*updated.get('bug_reports', []), {'id': bug.id, 'title': bug.title, 'description': bug.description, 'severity': bug.severity, 'failed_criteria': bug.failed_criteria, 'task_id': current_task['id']}]
        updated['retry_count'] = current_task['retry_count']
        updated['status'] = WorkflowStatus.REOPENED_FOR_DEV.value
        updated['current_agent'] = AgentName.ORCHESTRATOR.value
        return updated

    def mark_task_done(self, state: WorkflowState) -> WorkflowState:
        """Mark the current task as DONE. Does NOT advance to the next task.

        The service layer is responsible for picking the next task and
        invoking a new graph run.
        """
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        current_task = self.orchestrator.serialize_task(updated['current_task']['id'])
        updated['completed_tasks'] = [*updated.get('completed_tasks', []), current_task]
        updated['task_history'] = [*updated.get('task_history', []), current_task]
        updated['current_task'] = None
        updated['dev_output'] = None
        updated['qc_result'] = None
        updated['status'] = WorkflowStatus.DONE.value
        updated['current_agent'] = AgentName.ORCHESTRATOR.value
        return updated

    def max_retry_exceeded(self, state: WorkflowState) -> WorkflowState:
        """Mark the current task as BLOCKED due to max retry or loop detection.

        Does NOT advance to the next task. The service layer handles that.
        """
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        current_task = updated['current_task']
        task_display = f"t-{current_task['task_number']:03d}"

        # Determine if this was triggered by loop detection or genuine max_retry
        loop_sigs = updated.get('loop_signatures', [])
        if loop_sigs:
            sig = loop_sigs[-1] if loop_sigs else ''
            count = sum(1 for s in loop_sigs if s == sig)
            if count >= 3:
                reason = f"Task {task_display} blocked by loop detection (no progress after {count} identical cycles)."
            else:
                reason = f"Task {task_display} exceeded max retry limit."
        else:
            reason = f"Task {task_display} exceeded max retry limit."

        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.MAX_RETRY_EXCEEDED.value, AgentName.ORCHESTRATOR.value, reason, task_id=current_task['id'])
        self.orchestrator.update_task_status(updated['workflow_id'], current_task['id'], TaskStatus.BLOCKED.value, AgentName.ORCHESTRATOR.value, f"Max retry exceeded for {task_display}. Task blocked.")
        blocked = self.orchestrator.serialize_task(current_task['id'])
        updated['blocked_tasks'] = [*updated.get('blocked_tasks', []), blocked]
        updated['task_history'] = [*updated.get('task_history', []), blocked]
        updated['current_task'] = None
        updated['dev_output'] = None
        updated['qc_result'] = None
        updated['status'] = WorkflowStatus.MAX_RETRY_EXCEEDED.value
        updated['current_agent'] = AgentName.ORCHESTRATOR.value
        return updated
