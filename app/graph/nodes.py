from typing import Callable
from copy import deepcopy
import os
from pathlib import Path

from app.agents.base import DevResult, POResult
from app.agents.openai_agents import POAgent, POReviewAgent, DevAgent, QCAgent, to_plain_dict, validate_po_result_locally, filter_active_po_issues
from app.core.enums import AgentName, TaskStatus, WorkflowStatus
from app.graph.state import WorkflowState
from app.services.orchestrator_service import OrchestratorService
from app.core.logging_helper import WorkflowLogger


class WorkflowNodes:
    def __init__(self, orchestrator: OrchestratorService, po_agent: POAgent, po_review_agent: POReviewAgent, dev_agent: DevAgent, qc_agent: QCAgent):
        self.orchestrator = orchestrator
        self.po_agent = po_agent
        self.po_review_agent = po_review_agent
        self.dev_agent = dev_agent
        self.qc_agent = qc_agent

    def _check_cancelled(self, workflow_id: int):
        if self.orchestrator.check_cancellation(workflow_id):
            raise InterruptedError(f"Workflow {workflow_id} was cancelled by user.")

    def _log_node_enter(self, name: str, state: WorkflowState):
        current_task = state.get('current_task') or {}
        WorkflowLogger.info("workflow.node.enter",
            workflow_id=state.get('workflow_id'),
            node_name=name,
            task_id=current_task.get('id'),
            task_number=current_task.get('task_number'),
            retry_count=state.get('retry_count'),
            status=state.get('status')
        )

    def _log_node_exit(self, name: str, state: WorkflowState):
        current_task = state.get('current_task') or {}
        WorkflowLogger.info("workflow.node.exit",
            workflow_id=state.get('workflow_id'),
            node_name=name,
            task_id=current_task.get('id'),
            task_number=current_task.get('task_number'),
            status=state.get('status')
        )

    # ──────────────────────────────────────────────────────────────────────
    #  PO Phase Nodes (used by build_po_graph)
    # ──────────────────────────────────────────────────────────────────────

    def ingest_brd(self, state: WorkflowState) -> WorkflowState:
        self._log_node_enter('ingest_brd', state)
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        updated['visited_nodes'] = [*updated.get('visited_nodes', []), 'ingest_brd']
        updated['timestamps'] = {**updated.get('timestamps', {}), 'ingest_brd': self.orchestrator.now_iso()}
        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.NEW.value, AgentName.ORCHESTRATOR.value, 'BRD ingested into workflow state.')
        updated['status'] = WorkflowStatus.NEW.value
        updated['current_agent'] = AgentName.ORCHESTRATOR.value
        self._log_node_exit('ingest_brd', updated)
        return updated

    def orchestrator_init(self, state: WorkflowState) -> WorkflowState:
        self._log_node_enter('orchestrator_init', state)
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        updated['visited_nodes'] = [*updated.get('visited_nodes', []), 'orchestrator_init']
        updated['event_logs'] = updated.get('event_logs', [])
        updated['task_history'] = updated.get('task_history', [])
        updated['completed_tasks'] = updated.get('completed_tasks', [])
        updated['blocked_tasks'] = updated.get('blocked_tasks', [])
        updated['bug_reports'] = updated.get('bug_reports', [])
        updated['route_decisions'] = updated.get('route_decisions', [])
        updated['current_task'] = None
        updated['dev_output'] = None
        updated['qc_result'] = None
        self._log_node_exit('orchestrator_init', updated)
        return updated

    def po_analyze_brd(self, state: WorkflowState) -> WorkflowState:
        self._log_node_enter('po_analyze_brd', state)
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        updated['visited_nodes'] = [*updated.get('visited_nodes', []), 'po_analyze_brd']
        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.PO_ANALYZING.value, AgentName.PO.value, 'PO agent is analyzing BRD.')
        self._check_cancelled(updated['workflow_id'])
        
        # Extract active feedback for the PO Agent.
        # We only pass ACTIVE BLOCKING issues to the PO agent for fixing.
        # Historical issues are kept in po_review_issues_history for the Reviewer's convergence check.
        latest_review = updated.get('po_review_result') or {}
        active_review_issues = latest_review.get('issues', []) if latest_review.get('decision') == 'NEEDS_REVISION' else []
        local_issues = updated.get('po_local_issues', [])
        
        all_active_issues = [*active_review_issues, *local_issues]

        previous_po_result = updated.get('po_result')

        self.orchestrator.ctx.session.commit() # End transaction before long LLM call
        result = self.po_agent.analyze(
            brd_content=updated['brd_content'],
            previous_result=previous_po_result,
            review_issues=all_active_issues
        )
        self.orchestrator.ctx.brd_repo.create_feature(updated['workflow_id'], result.feature_summary)
        self.orchestrator.record_agent_run(updated['workflow_id'], AgentName.PO.value, 'SUCCESS', input_payload={'brd_id': updated['brd_id']}, output_payload=result.model_dump())
        self.orchestrator.ctx.session.commit()
        updated['feature_summary'] = result.feature_summary
        updated['po_result'] = result.model_dump()
        updated['po_local_issues'] = [] # Clear local issues after they are addressed
        updated['po_review_result'] = None # Clear stale review result
        updated['current_agent'] = AgentName.PO.value
        updated['status'] = WorkflowStatus.PO_ANALYZING.value
        self._log_node_exit('po_analyze_brd', updated)
        return updated

    def po_create_user_stories(self, state: WorkflowState) -> WorkflowState:
        """Parse user stories from PO result into state (no DB persistence yet — deferred to po_review)."""
        self._log_node_enter('po_create_user_stories', state)
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        updated['visited_nodes'] = [*updated.get('visited_nodes', []), 'po_create_user_stories']
        po_result = updated['po_result']

        # Build story dicts from po_result — NOT persisted to DB yet.
        staged_stories: list[dict] = []
        staged_criteria: list[dict] = []
        for idx, story in enumerate(po_result['user_stories']):
            story_entry = {'_idx': idx, **story}
            staged_stories.append(story_entry)
            for text in story['acceptance_criteria']:
                staged_criteria.append({'text': text, '_story_idx': idx})

        updated['user_stories'] = staged_stories
        updated['acceptance_criteria'] = staged_criteria
        self._log_node_exit('po_create_user_stories', updated)
        return updated

    def po_create_backlog_and_tasks(self, state: WorkflowState) -> WorkflowState:
        """Stage backlog and tasks in state (no DB persistence yet — deferred to po_review)."""
        self._log_node_enter('po_create_backlog_and_tasks', state)
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        updated['visited_nodes'] = [*updated.get('visited_nodes', []), 'po_create_backlog_and_tasks']
        po_result = updated['po_result']
        backlog_items = po_result['backlog_items']
        tasks = po_result['implementation_tasks']

        # Stage backlog items (NOT persisted to DB yet)
        staged_backlog: list[dict] = []
        for idx, backlog_item in enumerate(backlog_items, start=1):
            staged_backlog.append({'_idx': idx, **backlog_item})

        # Stage tasks (NOT persisted to DB yet)
        staged_tasks: list[dict] = []
        for task_idx, task_data in enumerate(tasks):
            staged_tasks.append({'_task_idx': task_idx, **task_data})

        updated['backlog'] = staged_backlog
        updated['task_queue'] = staged_tasks
        updated['status'] = WorkflowStatus.BACKLOG_CREATED.value
        updated['current_agent'] = AgentName.PO.value
        # Keep po_result for the review agent — it will be cleared after review
        self._log_node_exit('po_create_backlog_and_tasks', updated)
        return updated

    def po_local_validate(self, state: WorkflowState) -> WorkflowState:
        """Perform rule-based validation on PO result before calling the LLM reviewer."""
        self._log_node_enter('po_local_validate', state)
        updated = deepcopy(state)
        updated['visited_nodes'] = [*updated.get('visited_nodes', []), 'po_local_validate']
        
        po_result_dict = updated.get('po_result')
        if not po_result_dict:
            return updated
            
        try:
            po_result = POResult.model_validate(po_result_dict)
            local_issues = validate_po_result_locally(po_result)
            blocking_issues = filter_active_po_issues(local_issues)
            updated['po_local_issues'] = blocking_issues
            
            if blocking_issues:
                WorkflowLogger.warning("workflow.po_local_validate.failed",
                    workflow_id=updated['workflow_id'],
                    issues_count=len(blocking_issues),
                    message=f"PO local validation failed with {len(blocking_issues)} blocking issues."
                )
                updated['po_review_retries'] = updated.get('po_review_retries', 0) + 1
            else:
                WorkflowLogger.info("workflow.po_local_validate.passed",
                    workflow_id=updated['workflow_id'],
                    message="PO local validation passed."
                )
        except Exception as exc:
            WorkflowLogger.error("workflow.po_local_validate.error",
                workflow_id=updated['workflow_id'],
                error=str(exc)
            )
            # Treat schema validation error as a local issue to trigger retry
            updated['po_local_issues'] = [{
                "category": "schema_compliance",
                "severity": "HIGH",
                "description": f"Pydantic validation error: {str(exc)}",
                "suggestion": "Fix the JSON structure to match the required schema."
            }]
            updated['po_review_retries'] = updated.get('po_review_retries', 0) + 1

        self._log_node_exit('po_local_validate', updated)
        return updated

    # ──────────────────────────────────────────────────────────────────────
    #  PO Review Gate (validation before persistence)
    # ──────────────────────────────────────────────────────────────────────

    def po_review(self, state: WorkflowState) -> WorkflowState:
        """Validate PO output via the review agent.

        On PASS: persists all stories, backlog items, and tasks to DB.
        On NEEDS_REVISION: increments retry counter; nothing is persisted.
        """
        self._log_node_enter('po_review', state)
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        updated['visited_nodes'] = [*updated.get('visited_nodes', []), 'po_review']

        WorkflowLogger.info("workflow.po_review.started",
            workflow_id=updated['workflow_id'],
            message="PO review started"
        )

        self.orchestrator.update_workflow_status(
            updated['workflow_id'],
            WorkflowStatus.PO_REVIEW_IN_PROGRESS.value,
            AgentName.PO_REVIEW.value,
            'PO review agent is validating PO output.'
        )

        po_result = updated.get('po_result') or {}

        # Run the review agent with history for convergence
        self.orchestrator.ctx.session.commit() # End transaction before long LLM call
        review_result = self.po_review_agent.review(
            brd_content=updated['brd_content'],
            feature_summary=updated.get('feature_summary', ''),
            user_stories=po_result.get('user_stories', []),
            backlog_items=po_result.get('backlog_items', []),
            implementation_tasks=po_result.get('implementation_tasks', []),
            previous_issues=updated.get('po_review_issues_history', []),
        )

        review_dict = review_result.model_dump()
        updated['po_review_result'] = review_dict
        
        # Accumulate issues into history if revision is needed
        if review_dict.get('decision') == 'NEEDS_REVISION':
            new_issues = review_dict.get('issues', [])
            blocking_new_issues = filter_active_po_issues(new_issues)
            updated['po_review_issues_history'] = [
                *updated.get('po_review_issues_history', []),
                *blocking_new_issues
            ]
        
        updated['current_agent'] = AgentName.PO_REVIEW.value

        # Record the review run in audit trail
        self.orchestrator.record_agent_run(
            updated['workflow_id'],
            AgentName.PO_REVIEW.value,
            'SUCCESS',
            input_payload={'brd_id': updated.get('brd_id')},
            output_payload=review_dict,
        )

        if review_result.decision == 'PASS':
            WorkflowLogger.info("workflow.po_review.passed",
                workflow_id=updated['workflow_id'],
                message="PO review passed"
            )
            # ── Persist all PO artifacts to DB ──
            updated = self._persist_po_artifacts(updated)
            updated['status'] = WorkflowStatus.BACKLOG_CREATED.value
            # Clean up po_result now that everything is persisted
            updated.pop('po_result', None)
        else:
            issues_summary = [f"[{i.get('category', '?')}] {i.get('description', '')}" for i in review_dict.get('issues', [])]
            WorkflowLogger.warning("workflow.po_review.failed",
                workflow_id=updated['workflow_id'],
                issues=issues_summary,
                message=f"PO review failed with {len(issues_summary)} issues"
            )
            updated['po_review_retries'] = updated.get('po_review_retries', 0) + 1
            updated['status'] = WorkflowStatus.PO_REVIEW_FAILED.value

        self.orchestrator.ctx.session.commit()
        self._log_node_exit('po_review', updated)
        return updated

    def _persist_po_artifacts(self, state: dict) -> dict:
        """Persist all staged PO artifacts (stories, backlog, tasks) to DB.

        Called only when po_review passes. Returns updated state with DB IDs.
        """
        workflow_id = state['workflow_id']
        po_result = state.get('po_result') or {}

        # ── 1. Persist user stories ──────────────────────────────────────
        persisted_stories: list[dict] = []
        persisted_criteria: list[dict] = []
        story_db_map: dict[int, int] = {}  # _idx -> db story id

        for story in po_result.get('user_stories', []):
            db_story = self.orchestrator.ctx.brd_repo.create_user_story(
                workflow_id, story['title'], story['description'], story.get('priority', 'MEDIUM')
            )
            persisted_stories.append({'id': db_story.id, **story})
            # Track mapping for criteria
            story_idx = len(persisted_stories) - 1
            story_db_map[story_idx] = db_story.id
            for text in story.get('acceptance_criteria', []):
                criterion = self.orchestrator.ctx.brd_repo.create_acceptance_criterion(
                    workflow_id, text, user_story_id=db_story.id
                )
                persisted_criteria.append({'id': criterion.id, 'text': text, 'user_story_id': db_story.id})

        state['user_stories'] = persisted_stories
        state['acceptance_criteria'] = persisted_criteria

        # ── 2. Persist backlog items ─────────────────────────────────────
        backlog_items = po_result.get('backlog_items', [])
        tasks = po_result.get('implementation_tasks', [])
        persisted_backlog: list[dict] = []
        persisted_tasks: list[dict] = []

        db_backlog_map: list[int] = []
        for idx, backlog_item in enumerate(backlog_items, start=1):
            db_backlog = self.orchestrator.ctx.task_repo.create_backlog_item(
                workflow_id, backlog_item['title'], backlog_item['description'], backlog_item['team'], idx
            )
            persisted_backlog.append({'id': db_backlog.id, **backlog_item})
            db_backlog_map.append(db_backlog.id)

        # ── 3. Persist tasks ─────────────────────────────────────────────
        for task_idx, task_data in enumerate(tasks):
            backlog_id = db_backlog_map[min(task_idx, len(db_backlog_map) - 1)] if db_backlog_map else None
            db_task = self.orchestrator.ctx.task_repo.create_task(
                workflow_id=workflow_id,
                backlog_item_id=backlog_id,
                title=task_data['title'],
                description=task_data['description'],
                assignee_team=task_data['assignee_team'],
                max_retry=state['max_retry'],
                required_markers=task_data['required_markers'],
                input_context=task_data['input_context'],
            )
            for text in task_data.get('acceptance_criteria', []):
                self.orchestrator.ctx.task_repo.attach_acceptance_criterion(workflow_id, db_task.id, text)
            persisted_tasks.append(self.orchestrator.serialize_task(db_task.id))

        self.orchestrator.update_workflow_status(
            workflow_id, WorkflowStatus.BACKLOG_CREATED.value,
            AgentName.PO.value, 'PO artifacts persisted after review passed.'
        )

        state['backlog'] = persisted_backlog
        state['task_queue'] = persisted_tasks
        return state

    def po_review_failed(self, state: WorkflowState) -> WorkflowState:
        """Terminal node: PO review exhausted all retries — mark workflow as failed."""
        self._log_node_enter('po_review_failed', state)
        updated = deepcopy(state)
        updated['visited_nodes'] = [*updated.get('visited_nodes', []), 'po_review_failed']

        review_result = updated.get('po_review_result') or {}
        issues = review_result.get('issues', [])

        WorkflowLogger.error("workflow.po_review.exhausted",
            workflow_id=updated['workflow_id'],
            retries=updated.get('po_review_retries', 0),
            issues_count=len(issues),
            message=f"PO review failed with issues after all retries exhausted"
        )

        self.orchestrator.update_workflow_status(
            updated['workflow_id'],
            WorkflowStatus.PO_REVIEW_FAILED.value,
            AgentName.PO_REVIEW.value,
            f'PO review failed after {updated.get("po_review_retries", 0)} retries. '
            f'{len(issues)} unresolved issues. Workflow cannot proceed to DEV.',
        )
        self.orchestrator.ctx.session.commit()

        updated['status'] = WorkflowStatus.PO_REVIEW_FAILED.value
        updated['current_agent'] = AgentName.PO_REVIEW.value
        self._log_node_exit('po_review_failed', updated)
        return updated

    # ──────────────────────────────────────────────────────────────────────
    #  Task Execution Nodes (used by build_task_graph)
    # ──────────────────────────────────────────────────────────────────────

    def dispatch_to_dev(self, state: WorkflowState) -> WorkflowState:
        self._log_node_enter('dispatch_to_dev', state)
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        updated['visited_nodes'] = [*updated.get('visited_nodes', []), 'dispatch_to_dev']
        current_task = updated['current_task']
        if current_task is None:
            raise ValueError("dispatch_to_dev called with no current_task — state was not properly initialized by the service layer.")
        task_display = f"t-{current_task['task_number']:03d}"
        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.TASK_READY_FOR_DEV.value, AgentName.ORCHESTRATOR.value, f"Task {task_display} dispatched to DEV.", task_id=current_task['id'])
        self.orchestrator.update_task_status(updated['workflow_id'], current_task['id'], TaskStatus.READY.value, AgentName.ORCHESTRATOR.value, 'Task prepared for DEV implementation.')
        updated['status'] = WorkflowStatus.TASK_READY_FOR_DEV.value
        updated['current_agent'] = AgentName.ORCHESTRATOR.value
        updated['retry_count'] = current_task['retry_count']
        self._log_node_exit('dispatch_to_dev', updated)
        return updated

    def dev_implement(self, state: WorkflowState) -> WorkflowState:
        self._log_node_enter('dev_implement', state)
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        updated['visited_nodes'] = [*updated.get('visited_nodes', []), 'dev_implement']
        current_task = updated['current_task']
        task_display = f"t-{current_task['task_number']:03d}"
        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.DEV_IN_PROGRESS.value, AgentName.DEV.value, f"DEV is implementing task {task_display}.", task_id=current_task['id'])
        self.orchestrator.update_task_status(updated['workflow_id'], current_task['id'], TaskStatus.DEV_IN_PROGRESS.value, AgentName.DEV.value, f"DEV started implementation for {task_display}.")
        self._check_cancelled(updated['workflow_id'])
        bug_reports = self.orchestrator.ctx.task_repo.list_bugs_for_task(current_task['id'])
        # Try to read project context (README.md) from the physical files
        project_context = None
        try:
            project_dir = Path("generated_code") / f"wf_{updated['workflow_id']}" / "project"
            readme_file = project_dir / "readme.md"
            if not readme_file.exists():
                # Fallback to uppercase README.md
                readme_file = project_dir / "README.md"
            
            if readme_file.exists():
                project_context = readme_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[wf_{updated['workflow_id']}] Error reading project context: {e}")

        self.orchestrator.ctx.session.commit() # End transaction before long LLM call
        dev_result = self.dev_agent.implement(
            task=current_task,
            acceptance_criteria=current_task['acceptance_criteria'],
            bug_reports=[{'id': b.id, 'title': b.title, 'description': b.description, 'failed_criteria': b.failed_criteria} for b in bug_reports],
            project_context=project_context,
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
        self._log_node_exit('dev_implement', updated)
        return updated

    def _save_physical_files(self, workflow_id: int, task: dict, dev_result: DevResult):
        """Save all task outputs into a single unified project directory.
        
        All files are written to generated_code/wf_{id}/project/ using the
        file_path returned by the LLM.
        """
        try:
            project_dir = Path("generated_code") / f"wf_{workflow_id}" / "project"

            # ── 1. Save implementation files ──────────────────────────────────
            for imp_file in dev_result.files:
                # Basic sanitize: remove leading slash/backslash and '..'
                raw_path = imp_file.file_path.lstrip("/\\").replace("..", "")
                dest_file = project_dir / raw_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                dest_file.write_text(imp_file.code, encoding="utf-8")
                print(f"[wf_{workflow_id}] Saved file: {raw_path}")

            # ── 2. Save unit tests ────────────────────────────────────────────
            for test_file_obj in dev_result.unit_tests:
                raw_path = test_file_obj.file_path.lstrip("/\\").replace("..", "")
                dest_file = project_dir / raw_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                dest_file.write_text(test_file_obj.code, encoding="utf-8")
                print(f"[wf_{workflow_id}] Saved test: {raw_path}")

            # ── 3. Implementation notes ──────────────────────────────────────
            if dev_result.implementation_notes:
                # Create a generic name for notes based on task number
                notes_rel = Path("docs") / f"task_{task['task_number']:03d}_notes.md"
                notes_file = project_dir / notes_rel
                notes_file.parent.mkdir(parents=True, exist_ok=True)
                header = (
                    f"# Task t-{task['task_number']:03d}: {task.get('title', '')}\n\n"
                    f"**Task ID:** `{task.get('task_id', 'N/A')}`\n\n"
                )
                notes_file.write_text(header + dev_result.implementation_notes, encoding="utf-8")

        except Exception as e:
            print(f"[wf_{workflow_id}] Error saving physical files: {e}")

    def dispatch_to_qc(self, state: WorkflowState) -> WorkflowState:
        self._log_node_enter('dispatch_to_qc', state)
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        updated['visited_nodes'] = [*updated.get('visited_nodes', []), 'dispatch_to_qc']
        current_task = updated['current_task']
        task_display = f"t-{current_task['task_number']:03d}"
        self.orchestrator.update_workflow_status(updated['workflow_id'], WorkflowStatus.QC_IN_PROGRESS.value, AgentName.ORCHESTRATOR.value, f"Task {task_display} dispatched to QC.", task_id=current_task['id'])
        self.orchestrator.update_task_status(updated['workflow_id'], current_task['id'], TaskStatus.QC_IN_PROGRESS.value, AgentName.ORCHESTRATOR.value, f"Task {task_display} prepared for QC validation.")
        updated['status'] = WorkflowStatus.QC_IN_PROGRESS.value
        updated['current_agent'] = AgentName.ORCHESTRATOR.value
        self._log_node_exit('dispatch_to_qc', updated)
        return updated

    def qc_validate(self, state: WorkflowState) -> WorkflowState:
        self._log_node_enter('qc_validate', state)
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        updated['visited_nodes'] = [*updated.get('visited_nodes', []), 'qc_validate']
        current_task = updated['current_task']
        self._check_cancelled(updated['workflow_id'])
        # Try to read project context (README.md) from the physical files
        project_context = None
        try:
            project_dir = Path("generated_code") / f"wf_{updated['workflow_id']}" / "project"
            readme_file = project_dir / "readme.md"
            if not readme_file.exists():
                # Fallback to uppercase README.md
                readme_file = project_dir / "README.md"
            
            if readme_file.exists():
                project_context = readme_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[wf_{updated['workflow_id']}] Error reading project context: {e}")

        self.orchestrator.ctx.session.commit() # End transaction before long LLM call
        qc_result = self.qc_agent.validate(current_task, current_task['acceptance_criteria'], updated['dev_output'] or {}, project_context=project_context)
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

        self._log_node_exit('qc_validate', updated)
        return updated

    def create_bug(self, state: WorkflowState) -> WorkflowState:
        self._log_node_enter('create_bug', state)
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        updated['visited_nodes'] = [*updated.get('visited_nodes', []), 'create_bug']
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
        self._log_node_exit('create_bug', updated)
        return updated

    def mark_task_done(self, state: WorkflowState) -> WorkflowState:
        """Mark the current task as DONE. Does NOT advance to the next task."""
        self._log_node_enter('mark_task_done', state)
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        updated['visited_nodes'] = [*updated.get('visited_nodes', []), 'mark_task_done']
        current_task = self.orchestrator.serialize_task(updated['current_task']['id'])
        updated['completed_tasks'] = [*updated.get('completed_tasks', []), current_task]
        updated['task_history'] = [*updated.get('task_history', []), current_task]
        updated['current_task'] = None
        updated['dev_output'] = None
        updated['qc_result'] = None
        updated['status'] = WorkflowStatus.DONE.value
        updated['current_agent'] = AgentName.ORCHESTRATOR.value
        self._log_node_exit('mark_task_done', updated)
        return updated

    def max_retry_exceeded(self, state: WorkflowState) -> WorkflowState:
        """Mark the current task as BLOCKED due to max retry or loop detection."""
        self._log_node_enter('max_retry_exceeded', state)
        self._check_cancelled(state['workflow_id'])
        updated = deepcopy(state)
        updated['visited_nodes'] = [*updated.get('visited_nodes', []), 'max_retry_exceeded']
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
        self._log_node_exit('max_retry_exceeded', updated)
        return updated
