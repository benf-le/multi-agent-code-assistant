import logging
from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from app.agents.openai_agents import POAgent, POReviewAgent, DevAgent, QCAgent, FinalProjectQAAgent
from app.core.config import get_settings
from app.core.enums import AgentName, WorkflowStatus, TaskStatus
from app.graph.nodes import WorkflowNodes
from app.graph.state import WorkflowState
from app.graph.workflow import WorkflowGraphFactory
from app.repositories.audit_repository import AuditRepository
from app.repositories.brd_repository import BRDRepository
from app.repositories.task_repository import TaskRepository
from app.repositories.workflow_repository import WorkflowRepository
from app.services.event_bus import DatabaseEventBus
from app.services.orchestrator_service import OrchestratorContext, OrchestratorService
from app.core.logging_helper import WorkflowLogger

logger = logging.getLogger(__name__)

# ── Sentinel status sets ────────────────────────────────────────────────
_TERMINAL_WORKFLOW_STATUSES = frozenset({
    WorkflowStatus.DONE.value,
    WorkflowStatus.BLOCKED.value,
    WorkflowStatus.CANCELLED.value,
    WorkflowStatus.PO_REVIEW_FAILED.value,
    WorkflowStatus.FINAL_QA_FAILED.value,
})

_RUNNABLE_TASK_STATUSES = frozenset({
    TaskStatus.NEW.value,
    TaskStatus.READY.value,
    TaskStatus.REOPENED.value,
})

_IN_PROGRESS_TASK_STATUSES = frozenset({
    TaskStatus.DEV_IN_PROGRESS.value,
    TaskStatus.QC_IN_PROGRESS.value,
    TaskStatus.REOPENED.value,
    TaskStatus.READY.value,
})


class WorkflowService:
    """Orchestrates multi-task workflows using a 1-task-per-graph-run design.

    Architecture:
    - PO phase: graph run with review gate (po_review validates before persistence)
    - Task phase: service-level loop, each iteration invokes a bounded
      single-task graph (DEV→QC retry cycle only)
    - DB is source of truth between graph runs
    - GraphRecursionError is caught and handled gracefully
    """

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory
        self.settings = get_settings()

    # ──────────────────────────────────────────────────────────────────────
    #  Context helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_context(self, session: Session) -> OrchestratorService:
        brd_repo = BRDRepository(session)
        workflow_repo = WorkflowRepository(session)
        task_repo = TaskRepository(session)
        audit_repo = AuditRepository(session)
        event_bus = DatabaseEventBus(audit_repo)
        return OrchestratorService(OrchestratorContext(session, brd_repo, workflow_repo, task_repo, audit_repo, event_bus))

    def _build_agents(self):
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment. Real agents are required.")
        
        kwargs = {
            "api_key": self.settings.openai_api_key,
            "base_url": self.settings.openai_base_url,
            "model": self.settings.openai_model,
        }
        
        return (
            POAgent(**kwargs),
            POReviewAgent(**kwargs),
            DevAgent(**kwargs),
            QCAgent(**kwargs),
            FinalProjectQAAgent(**kwargs),
        )

    def _build_graph_factory(self, orchestrator: OrchestratorService) -> WorkflowGraphFactory:
        po, po_review, dev, qc, final_qa = self._build_agents()
        nodes = WorkflowNodes(orchestrator, po, po_review, dev, qc, final_qa)
        return WorkflowGraphFactory(nodes)

    def _graph_config(self) -> dict:
        """Centralized graph invocation config — single source of truth."""
        return {"recursion_limit": self.settings.graph_recursion_limit}

    # ──────────────────────────────────────────────────────────────────────
    #  Workflow status recomputation (DB as source of truth)
    # ──────────────────────────────────────────────────────────────────────

    def _recompute_workflow_status(self, orchestrator: OrchestratorService, workflow_id: int) -> str:
        """Derive workflow-level status from the aggregate state of all tasks.

        Rules:
        - All tasks DONE → DONE
        - All tasks DONE or BLOCKED (at least one BLOCKED) → BLOCKED
        - Any task still runnable or in-progress → IN_PROGRESS equivalent
        - No tasks at all → keep current status (PO phase not done)
        """
        tasks = orchestrator.ctx.task_repo.list_tasks(workflow_id)
        if not tasks:
            # No tasks created yet — PO phase may not be complete
            workflow = orchestrator.ctx.workflow_repo.get(workflow_id)
            return workflow.status if workflow else WorkflowStatus.NEW.value

        statuses = [t.status for t in tasks]
        done_count = statuses.count(TaskStatus.DONE.value)
        blocked_count = statuses.count(TaskStatus.BLOCKED.value)
        total = len(statuses)

        if done_count == total:
            workflow = orchestrator.ctx.workflow_repo.get(workflow_id)
            if workflow and workflow.status == WorkflowStatus.FINAL_QA_PASSED.value:
                return WorkflowStatus.DONE.value
            if workflow and workflow.status in (
                WorkflowStatus.FINAL_QA_IN_PROGRESS.value,
                WorkflowStatus.FINAL_QA_FAILED.value,
                WorkflowStatus.FINAL_QA_RETRY_REQUIRED.value
            ):
                return workflow.status
            return WorkflowStatus.TASKS_COMPLETED.value
        if done_count + blocked_count == total:
            return WorkflowStatus.BLOCKED.value
        return WorkflowStatus.BACKLOG_CREATED.value  # still has runnable tasks

    def _finalize_workflow(self, orchestrator: OrchestratorService, workflow_id: int) -> str:
        """Recompute and persist the final workflow status after all tasks are processed."""
        final_status = self._recompute_workflow_status(orchestrator, workflow_id)
        WorkflowLogger.info("workflow.finalized", 
            workflow_id=workflow_id, 
            final_status=final_status,
            message=f"Workflow finalized with status: {final_status}"
        )
        if final_status in (WorkflowStatus.DONE.value, WorkflowStatus.BLOCKED.value, WorkflowStatus.FINAL_QA_FAILED.value):
            orchestrator.mark_workflow_finished(workflow_id, final_status, f'Workflow finished: {final_status}.')
        return final_status

    # ──────────────────────────────────────────────────────────────────────
    #  Task selection (DB-driven)
    # ──────────────────────────────────────────────────────────────────────

    def _pick_next_task(self, orchestrator: OrchestratorService, workflow_id: int) -> dict | None:
        """Select the next runnable task from the DB.

        Priority:
        1. Tasks currently in-progress (resume scenario)
        2. Tasks with status NEW/READY/REOPENED, ordered by task_number
        """
        tasks = orchestrator.ctx.task_repo.list_tasks(workflow_id)

        # First priority: an in-progress task (crashed/interrupted mid-execution)
        for t in tasks:
            if t.status in _IN_PROGRESS_TASK_STATUSES:
                return orchestrator.serialize_task(t.id)

        # Second priority: next runnable task by task_number
        runnable = [t for t in tasks if t.status in _RUNNABLE_TASK_STATUSES]
        runnable.sort(key=lambda t: t.task_number)
        if runnable:
            task = runnable[0]
            WorkflowLogger.info("workflow.task.selected",
                workflow_id=workflow_id,
                task_id=task.id,
                task_number=task.task_number,
                status=task.status
            )
            return orchestrator.serialize_task(task.id)

        WorkflowLogger.info("workflow.task.none_left", workflow_id=workflow_id)
        return None

    # ──────────────────────────────────────────────────────────────────────
    #  State builders
    # ──────────────────────────────────────────────────────────────────────

    def _build_po_state(self, workflow, brd) -> WorkflowState:
        """Build initial state for the PO analysis phase graph."""
        return {
            'workflow_id': workflow.id,
            'brd_id': brd.id,
            'brd_content': brd.content,
            'feature_summary': '',
            'user_stories': [],
            'acceptance_criteria': [],
            'backlog': [],
            'po_review_result': None,
            'po_review_retries': 0,
            'current_task': None,
            'task_queue': [],
            'task_history': [],
            'dev_output': None,
            'qc_result': None,
            'bug_reports': [],
            'retry_count': 0,
            'max_retry': workflow.max_retry,
            'current_agent': AgentName.ORCHESTRATOR.value,
            'status': WorkflowStatus.NEW.value,
            'event_logs': [],
            'timestamps': {'started_at': None},
            'blocked_tasks': [],
            'completed_tasks': [],
            'loop_signatures': [],
            'visited_nodes': [],
            'route_decisions': [],
            'current_project': {},
            'candidate_project': {},
        }

    def _build_task_state(self, orchestrator: OrchestratorService, workflow, brd, task_dict: dict) -> WorkflowState:
        """Build state for a single-task execution graph, reconstructed from DB."""
        bug_reports = orchestrator.ctx.task_repo.list_bugs_for_task(task_dict['id'])
        current_project = self._load_project_from_disk(workflow.id)
        return {
            'workflow_id': workflow.id,
            'brd_id': brd.id,
            'brd_content': brd.content,
            'feature_summary': '',  # Not needed for task execution
            'user_stories': [],
            'acceptance_criteria': [],
            'backlog': [],
            'current_task': task_dict,
            'task_queue': [],  # Not used — service layer manages task sequencing
            'task_history': [],
            'dev_output': task_dict.get('output_context') if task_dict.get('output_context') else None,
            'qc_result': None,
            'bug_reports': [
                {'id': b.id, 'title': b.title, 'description': b.description,
                 'severity': b.severity, 'failed_criteria': b.failed_criteria, 'task_id': task_dict['id']}
                for b in bug_reports
            ],
            'retry_count': task_dict.get('retry_count', 0),
            'max_retry': workflow.max_retry,
            'current_agent': AgentName.ORCHESTRATOR.value,
            'status': WorkflowStatus.TASK_READY_FOR_DEV.value,
            'event_logs': [],
            'timestamps': {'task_started_at': orchestrator.now_iso()},
            'blocked_tasks': [],
            'completed_tasks': [],
            'loop_signatures': [],
            'visited_nodes': [],
            'route_decisions': [],
            'current_project': current_project,
            'candidate_project': {},
        }

    def _load_project_from_disk(self, workflow_id: int) -> dict[str, str]:
        """Load the accepted generated project so later task graph runs keep context."""
        project_dir = Path("generated_code") / f"wf_{workflow_id}" / "project"
        if not project_dir.exists():
            return {}

        project: dict[str, str] = {}
        for path in project_dir.rglob("*"):
            if not path.is_file():
                continue
            rel_path = path.relative_to(project_dir).as_posix()
            try:
                project[rel_path] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                WorkflowLogger.warning(
                    "workflow.project_state.skip_binary_file",
                    workflow_id=workflow_id,
                    file_path=rel_path,
                )
            except Exception as exc:
                WorkflowLogger.warning(
                    "workflow.project_state.read_file_failed",
                    workflow_id=workflow_id,
                    file_path=rel_path,
                    error=str(exc),
                )
        return project

    # ──────────────────────────────────────────────────────────────────────
    #  PO Phase execution
    # ──────────────────────────────────────────────────────────────────────

    def _run_po_phase(self, session: Session, orchestrator: OrchestratorService, workflow, brd) -> dict:
        """Run the PO analysis phase as a single linear graph invocation.

        This graph has no cycles so GraphRecursionError is extremely unlikely,
        but we still handle it defensively.
        """
        graph_factory = self._build_graph_factory(orchestrator)
        po_graph = graph_factory.build_po_graph()
        state = self._build_po_state(workflow, brd)
        workflow_id = workflow.id

        try:
            WorkflowLogger.info("workflow.graph.start", 
                workflow_id=workflow_id, 
                graph_name="po_graph",
                recursion_limit=self._graph_config().get("recursion_limit")
            )
            session.commit() # End transaction before long-running graph invocation
            result = po_graph.invoke(state, config=self._graph_config())
            WorkflowLogger.info("workflow.graph.end", 
                workflow_id=workflow_id, 
                graph_name="po_graph",
                visited_nodes=result.get("visited_nodes", [])
            )

            # Check if PO review failed — abort before task phase
            if result.get('status') == WorkflowStatus.PO_REVIEW_FAILED.value:
                logger.warning('PO review failed for workflow %s — aborting before task phase', workflow_id)
                return result

            logger.info('PO phase completed for workflow %s', workflow_id)
            return result
        except Exception as e:
            self._handle_graph_error(orchestrator, workflow_id, None, e, phase='po_phase')
            raise

    # ──────────────────────────────────────────────────────────────────────
    #  Single-task execution
    # ──────────────────────────────────────────────────────────────────────

    def _run_single_task(self, session: Session, orchestrator: OrchestratorService, workflow, brd, task_dict: dict) -> dict:
        """Run exactly one task lifecycle in a bounded graph invocation.

        Returns the final graph state for this task.
        Catches GraphRecursionError and marks the task as BLOCKED.
        """
        task_display = f"t-{task_dict['task_number']:03d}"
        logger.info('Starting task %s for workflow %s', task_display, workflow.id)

        graph_factory = self._build_graph_factory(orchestrator)
        task_graph = graph_factory.build_task_graph()
        state = self._build_task_state(orchestrator, workflow, brd, task_dict)
        workflow_id = workflow.id
        task_id = task_dict['id']

        try:
            config = self._graph_config()
            WorkflowLogger.info("workflow.graph.start", 
                workflow_id=workflow_id, 
                task_id=task_id,
                task_number=task_dict['task_number'],
                graph_name="task_graph",
                recursion_limit=config.get("recursion_limit")
            )
            session.commit() # End transaction before long-running graph invocation
            result = task_graph.invoke(state, config=config)
            WorkflowLogger.info("workflow.graph.end", 
                workflow_id=workflow_id, 
                task_id=task_id,
                graph_name="task_graph",
                status=result.get('status'),
                visited_nodes=result.get("visited_nodes", [])
            )
            logger.info('Task %s finished with status=%s', task_display, result.get('status'))
            return result
        except InterruptedError:
            logger.info('Task %s interrupted (workflow cancelled)', task_display)
            raise
        except Exception as e:
            self._handle_graph_error(orchestrator, workflow_id, task_id, e, phase=f'task_{task_display}')
            # Return a synthetic "blocked" result so the service loop can continue
            return {
                'workflow_id': workflow_id,
                'current_task': None,
                'status': WorkflowStatus.MAX_RETRY_EXCEEDED.value,
                'error': str(e),
                'visited_nodes': state.get('visited_nodes', []),
            }

    # ──────────────────────────────────────────────────────────────────────
    #  Final QA Phase execution
    # ──────────────────────────────────────────────────────────────────────

    def _run_final_qa_phase(self, session: Session, orchestrator: OrchestratorService, workflow, brd) -> dict:
        """Run the Final QA phase as a single graph invocation."""
        graph_factory = self._build_graph_factory(orchestrator)
        final_qa_graph = graph_factory.build_final_qa_graph()
        
        state = {
            'workflow_id': workflow.id,
            'brd_id': brd.id,
            'brd_content': brd.content,
            'status': workflow.status,
            'max_retry': workflow.max_retry,
            'current_agent': AgentName.ORCHESTRATOR.value,
            'current_project': {},
            'final_project_qa_attempts': 0,
        }
        
        workflow_id = workflow.id

        try:
            WorkflowLogger.info("workflow.graph.start", 
                workflow_id=workflow_id, 
                graph_name="final_qa_graph",
                recursion_limit=self._graph_config().get("recursion_limit")
            )
            session.commit()
            result = final_qa_graph.invoke(state, config=self._graph_config())
            WorkflowLogger.info("workflow.graph.end", 
                workflow_id=workflow_id, 
                graph_name="final_qa_graph",
                visited_nodes=result.get("visited_nodes", [])
            )
            return result
        except Exception as e:
            self._handle_graph_error(orchestrator, workflow_id, None, e, phase='final_qa_phase')
            raise

    # ──────────────────────────────────────────────────────────────────────
    #  Error handling
    # ──────────────────────────────────────────────────────────────────────

    def _handle_graph_error(self, orchestrator: OrchestratorService, workflow_id: int, task_id: int | None, error: Exception, phase: str = 'unknown') -> None:
        """Gracefully handle any graph execution error.

        - Catches GraphRecursionError specifically
        - Updates task to BLOCKED if task_id is provided
        - Records audit trail
        - Never re-raises (caller decides whether to propagate)
        """
        from langgraph.errors import GraphRecursionError

        # 1. Rollback the current session immediately to allow new transactions
        try:
            orchestrator.ctx.session.rollback()
        except Exception as rb_error:
            logger.error('Failed to rollback session for workflow %s: %s', workflow_id, rb_error)

        error_type = type(error).__name__
        error_msg = str(error)

        if isinstance(error, GraphRecursionError):
            logger.error('GraphRecursionError in %s for workflow %s (task_id=%s): %s', phase, workflow_id, task_id, error_msg)
        else:
            logger.error('Unexpected error in %s for workflow %s (task_id=%s): %s: %s', phase, workflow_id, task_id, error_type, error_msg)

        # 2. Use a NEW session for audit/status updates if the previous one is corrupted
        # Since orchestrator.update_* calls commit(), it will start a new transaction 
        # on the same session if the session is still valid after rollback.
        try:
            # Mark task as BLOCKED if applicable
            if task_id is not None:
                # We use the repo directly with the existing session (which was just rolled back)
                task = orchestrator.ctx.task_repo.get_task(task_id)
                if task and task.status not in (TaskStatus.DONE.value, TaskStatus.BLOCKED.value):
                    orchestrator.update_task_status(
                        workflow_id, task_id, TaskStatus.BLOCKED.value,
                        AgentName.ORCHESTRATOR.value,
                        f'Task blocked due to {error_type} during {phase}: {error_msg[:200]}',
                    )

            # Record the error in audit trail
            orchestrator.record_agent_run(
                workflow_id, AgentName.ORCHESTRATOR.value, 'ERROR',
                task_id=task_id,
                error_message=f'[{phase}] {error_type}: {error_msg[:500]}',
            )

            # Update workflow status to indicate error
            orchestrator.update_workflow_status(
                workflow_id, WorkflowStatus.MAX_RETRY_EXCEEDED.value,
                AgentName.ORCHESTRATOR.value,
                f'Error during {phase}: {error_type}',
                task_id=task_id,
                metadata={'error_type': error_type, 'error_message': error_msg[:500]},
            )
        except Exception as audit_error:
            # Don't let audit failures mask the original error
            logger.error('Failed to record error audit for workflow %s: %s', workflow_id, audit_error)

    # ──────────────────────────────────────────────────────────────────────
    #  Public API: run_workflow
    # ──────────────────────────────────────────────────────────────────────

    def create_workflow_from_brd(self, title: str, brd_content: str, max_retry: int | None = None) -> int:
        with self.session_factory() as session:
            brd_repo = BRDRepository(session)
            workflow_repo = WorkflowRepository(session)
            brd = brd_repo.create_brd(title, brd_content)
            workflow = workflow_repo.create_workflow(brd.id, self.settings.default_max_retry if max_retry is None else max_retry, {'title': title})
            session.commit()
            return workflow.id

    def run_workflow(self, workflow_id: int) -> dict:
        """Run a full workflow: PO phase + all tasks, one task per graph invocation.

        This is the main entry point for new workflow execution.
        """
        with self.session_factory() as session:
            orchestrator = self._build_context(session)
            workflow = orchestrator.ctx.workflow_repo.get(workflow_id)
            if workflow is None:
                raise ValueError(f'Workflow {workflow_id} not found')
            brd = orchestrator.ctx.brd_repo.get_brd(workflow.brd_id)
            if brd is None:
                raise ValueError(f'BRD {workflow.brd_id} not found')

            logger.info('Starting workflow execution %s', workflow.id)

            try:
                # ── Phase 1: PO Analysis ─────────────────────────────────
                # Skip if already past PO phase (e.g., tasks already exist)
                existing_tasks = orchestrator.ctx.task_repo.list_tasks(workflow_id)
                if not existing_tasks:
                    if workflow.status in (WorkflowStatus.NEW.value, WorkflowStatus.PO_ANALYZING.value):
                        po_result = self._run_po_phase(session, orchestrator, workflow, brd)
                        # Refresh workflow after PO phase
                        session.refresh(workflow)
                        # Abort if PO review failed — do NOT proceed to task phase
                        if workflow.status == WorkflowStatus.PO_REVIEW_FAILED.value:
                            logger.warning('Workflow %s aborted — PO review failed', workflow_id)
                            return {
                                'workflow_id': workflow_id,
                                'status': WorkflowStatus.PO_REVIEW_FAILED.value,
                                'po_review_result': po_result.get('po_review_result'),
                                'final_status': WorkflowStatus.PO_REVIEW_FAILED.value,
                            }
                    else:
                        logger.info('Skipping PO phase — workflow %s already at status %s', workflow_id, workflow.status)

                # ── Phase 2 & 3: Task execution & Final QA loop ──────────
                while True:
                    result = self._run_task_loop(session, orchestrator, workflow, brd)
                    
                    if result.get('final_status') == WorkflowStatus.TASKS_COMPLETED.value:
                        qa_result = self._run_final_qa_phase(session, orchestrator, workflow, brd)
                        session.expire_all()
                        workflow = orchestrator.ctx.workflow_repo.get(workflow_id)
                        if workflow.status == WorkflowStatus.FINAL_QA_RETRY_REQUIRED.value:
                            continue # Run task loop again for the new repair task
                        else:
                            result = qa_result
                            result['final_status'] = self._finalize_workflow(orchestrator, workflow_id)
                            break
                    else:
                        break

                logger.info('Workflow execution %s finished with status=%s', workflow.id, result.get('final_status'))
                return result

            except InterruptedError:
                logger.info('Workflow %s was cancelled during execution', workflow_id)
                return {'workflow_id': workflow_id, 'status': WorkflowStatus.CANCELLED.value}
            except ValueError:
                raise
            except Exception as e:
                logger.error('Unhandled error in run_workflow %s: %s', workflow_id, e, exc_info=True)
                # Ensure workflow doesn't get stuck in a non-terminal state
                try:
                    self._finalize_workflow(orchestrator, workflow_id)
                except Exception:
                    pass
                return {'workflow_id': workflow_id, 'status': 'ERROR', 'error': str(e)}

    def resume_workflow(self, workflow_id: int) -> dict:
        """Resume a workflow from its current DB state.

        Handles:
        - CANCELLED workflows (restores previous status)
        - Workflows stuck in PO phase
        - Workflows with pending tasks
        """
        with self.session_factory() as session:
            orchestrator = self._build_context(session)
            workflow = orchestrator.ctx.workflow_repo.get(workflow_id)
            if workflow is None:
                raise ValueError(f'Workflow {workflow_id} not found')

            # Handle CANCELLED state — restore previous status
            if workflow.status == WorkflowStatus.CANCELLED.value:
                transitions = orchestrator.ctx.audit_repo.list_transitions(workflow_id=workflow_id)
                prev_status = WorkflowStatus.NEW.value
                for t in reversed(transitions):
                    if t.to_status != WorkflowStatus.CANCELLED.value:
                        prev_status = t.to_status
                        break
                orchestrator.ctx.workflow_repo.update_status(workflow, status=prev_status, current_agent=workflow.current_agent)
                session.commit()
                session.refresh(workflow)

            # Handle terminal states
            if workflow.status in (WorkflowStatus.DONE.value, WorkflowStatus.BLOCKED.value, WorkflowStatus.PO_REVIEW_FAILED.value):
                logger.info('Workflow %s is already in terminal state %s', workflow_id, workflow.status)
                return {'workflow_id': workflow_id, 'status': workflow.status, 'message': 'Workflow already completed.'}

            brd = orchestrator.ctx.brd_repo.get_brd(workflow.brd_id)
            if brd is None:
                raise ValueError(f'BRD {workflow.brd_id} not found')

            logger.info('Resuming workflow %s from status=%s', workflow_id, workflow.status)

            try:
                # Check if PO phase needs to be run/re-run
                existing_tasks = orchestrator.ctx.task_repo.list_tasks(workflow_id)
                if not existing_tasks:
                    # PO phase not complete — re-run it
                    po_result = self._run_po_phase(session, orchestrator, workflow, brd)
                    session.refresh(workflow)
                    # Abort if PO review failed
                    if workflow.status == WorkflowStatus.PO_REVIEW_FAILED.value:
                        logger.warning('Workflow %s resume aborted — PO review failed', workflow_id)
                        return {
                            'workflow_id': workflow_id,
                            'status': WorkflowStatus.PO_REVIEW_FAILED.value,
                            'po_review_result': po_result.get('po_review_result'),
                            'final_status': WorkflowStatus.PO_REVIEW_FAILED.value,
                        }

                # Run task loop & final QA loop from current state
                while True:
                    result = self._run_task_loop(session, orchestrator, workflow, brd)
                    
                    if result.get('final_status') == WorkflowStatus.TASKS_COMPLETED.value:
                        qa_result = self._run_final_qa_phase(session, orchestrator, workflow, brd)
                        session.expire_all()
                        workflow = orchestrator.ctx.workflow_repo.get(workflow_id)
                        if workflow.status == WorkflowStatus.FINAL_QA_RETRY_REQUIRED.value:
                            continue # Run task loop again for the new repair task
                        else:
                            result = qa_result
                            result['final_status'] = self._finalize_workflow(orchestrator, workflow_id)
                            break
                    else:
                        break
                        
                logger.info('Resume of workflow %s finished with status=%s', workflow_id, result.get('final_status'))
                return result

            except InterruptedError:
                logger.info('Workflow %s was cancelled during resume', workflow_id)
                return {'workflow_id': workflow_id, 'status': WorkflowStatus.CANCELLED.value}
            except ValueError:
                raise
            except Exception as e:
                logger.error('Unhandled error in resume_workflow %s: %s', workflow_id, e, exc_info=True)
                try:
                    self._finalize_workflow(orchestrator, workflow_id)
                except Exception:
                    pass
                return {'workflow_id': workflow_id, 'status': 'ERROR', 'error': str(e)}

    def run_next_task(self, workflow_id: int) -> dict | None:
        """Run exactly one task for a workflow. Returns None if no runnable tasks remain.

        This is the granular entry point that external workers or schedulers
        can use to process one task at a time.
        """
        with self.session_factory() as session:
            orchestrator = self._build_context(session)
            workflow = orchestrator.ctx.workflow_repo.get(workflow_id)
            if workflow is None:
                raise ValueError(f'Workflow {workflow_id} not found')

            if workflow.status in _TERMINAL_WORKFLOW_STATUSES:
                logger.info('Workflow %s is in terminal state %s — no tasks to run', workflow_id, workflow.status)
                return None

            brd = orchestrator.ctx.brd_repo.get_brd(workflow.brd_id)
            if brd is None:
                raise ValueError(f'BRD {workflow.brd_id} not found')

            task_dict = self._pick_next_task(orchestrator, workflow_id)
            if task_dict is None:
                # No runnable tasks — finalize
                final_status = self._finalize_workflow(orchestrator, workflow_id)
                return {'workflow_id': workflow_id, 'final_status': final_status, 'task_id': None}

            try:
                result = self._run_single_task(session, orchestrator, workflow, brd, task_dict)

                # Recompute workflow status after this task
                session.expire_all()
                final_status = self._recompute_workflow_status(orchestrator, workflow_id)
                return {
                    'workflow_id': workflow_id,
                    'task_id': task_dict['id'],
                    'task_status': result.get('status'),
                    'workflow_status': final_status,
                }
            except InterruptedError:
                raise
            except Exception as e:
                logger.error('Error running task %s for workflow %s: %s', task_dict['id'], workflow_id, e, exc_info=True)
                return {
                    'workflow_id': workflow_id,
                    'task_id': task_dict['id'],
                    'task_status': 'ERROR',
                    'error': str(e),
                }

    # ──────────────────────────────────────────────────────────────────────
    #  Internal: task loop
    # ──────────────────────────────────────────────────────────────────────

    def _run_task_loop(self, session: Session, orchestrator: OrchestratorService, workflow, brd) -> dict:
        """Service-level loop: pick task → run graph → repeat until no tasks remain.

        Each iteration is a separate graph.invoke() call with bounded recursion.
        """
        last_result = {}
        tasks_processed = 0

        while True:
            # Refresh workflow state from DB
            session.expire_all()
            workflow = orchestrator.ctx.workflow_repo.get(workflow.id)
            if workflow is None or workflow.status in _TERMINAL_WORKFLOW_STATUSES:
                break

            task_dict = self._pick_next_task(orchestrator, workflow.id)
            if task_dict is None:
                logger.info('No more runnable tasks for workflow %s', workflow.id)
                break

            last_result = self._run_single_task(session, orchestrator, workflow, brd, task_dict)
            session.commit()
            tasks_processed += 1
            logger.info('Workflow %s: processed %d tasks so far', workflow.id, tasks_processed)

        # Finalize workflow
        final_status = self._finalize_workflow(orchestrator, workflow.id)
        last_result['final_status'] = final_status
        last_result['tasks_processed'] = tasks_processed
        return last_result

    # ──────────────────────────────────────────────────────────────────────
    #  Export
    # ──────────────────────────────────────────────────────────────────────

    def export_execution_log(self, workflow_id: int) -> dict:
        with self.session_factory() as session:
            audit_repo = AuditRepository(session)
            workflow_repo = WorkflowRepository(session)
            workflow = workflow_repo.get(workflow_id)
            if workflow is None:
                raise ValueError(f'Workflow {workflow_id} not found')
            transitions = audit_repo.list_transitions(workflow_id=workflow_id)
            events = audit_repo.list_events(workflow_id=workflow_id)
            runs = audit_repo.list_agent_runs(workflow_id=workflow_id)
            return {
                'workflow_id': workflow_id,
                'status': workflow.status,
                'transitions': [
                    {'id': t.id, 'task_id': t.task_id, 'from_status': t.from_status, 'to_status': t.to_status, 'agent_name': t.agent_name, 'reason': t.reason, 'created_at': t.created_at.isoformat() if t.created_at else None}
                    for t in transitions
                ],
                'events': [
                    {'id': e.id, 'task_id': e.task_id, 'event_type': e.event_type, 'agent_name': e.agent_name, 'message': e.message, 'payload': e.payload, 'created_at': e.created_at.isoformat() if e.created_at else None}
                    for e in events
                ],
                'agent_runs': [
                    {'id': r.id, 'task_id': r.task_id, 'agent_name': r.agent_name, 'status': r.status, 'input_payload': r.input_payload, 'output_payload': r.output_payload, 'created_at': r.started_at.isoformat() if r.started_at else None}
                    for r in runs
                ],
            }
