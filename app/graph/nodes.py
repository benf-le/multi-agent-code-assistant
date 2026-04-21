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
        for idx, backlog_item in enumerate(backlog_items, start=1):
            db_backlog = self.orchestrator.ctx.task_repo.create_backlog_item(updated['workflow_id'], backlog_item['title'], backlog_item['description'], backlog_item['team'], idx)
            persisted_backlog.append({'id': db_backlog.id, **backlog_item})
            task_data = tasks[idx - 1]
            db_task = self.orchestrator.ctx.task_repo.create_task(
                workflow_id=updated['workflow_id'],
                backlog_item_id=db_backlog.id,
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

    def dispatch_to_dev(self, state: WorkflowState) -> WorkflowState:
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        if updated.get('current_task') is None:
            if not updated.get('task_queue'):
                return updated
            updated['current_task'] = updated['task_queue'].pop(0)
        current_task = updated['current_task']
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
        try:
            base_dir = Path("generated_code") / f"wf_{workflow_id}" / f"task_{task['id']}"
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Save implementation code
            (base_dir / dev_result.file_name).write_text(dev_result.code, encoding="utf-8")
            
            # Save unit tests
            (base_dir / "tests.txt").write_text(dev_result.unit_tests, encoding="utf-8")
            
            # Save implementation notes
            (base_dir / "notes.txt").write_text(dev_result.implementation_notes, encoding="utf-8")
            
            print(f"Physical files saved to: {base_dir}")
        except Exception as e:
            print(f"Error saving physical files: {e}")

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
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        current_task = updated['current_task']
        task_display = f"t-{current_task['task_number']:03d}"
        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.MAX_RETRY_EXCEEDED.value, AgentName.ORCHESTRATOR.value, f"Task {task_display} exceeded max retry limit.", task_id=current_task['id'])
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

    def finalize_workflow(self, state: WorkflowState) -> WorkflowState:
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        final_status = WorkflowStatus.BLOCKED.value if updated.get('blocked_tasks') else WorkflowStatus.DONE.value
        self.orchestrator.mark_workflow_finished(updated['workflow_id'], final_status, 'Workflow finished after processing all tasks.')
        updated['status'] = final_status
        updated['current_agent'] = AgentName.ORCHESTRATOR.value
        updated['timestamps'] = {**updated.get('timestamps', {}), 'completed_at': self.orchestrator.now_iso()}
        return updated
