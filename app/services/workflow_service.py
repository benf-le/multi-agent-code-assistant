import logging
from typing import Callable

from sqlalchemy.orm import Session

from app.agents.openai_agents import POAgent, DevAgent, QCAgent
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

logger = logging.getLogger(__name__)


class WorkflowService:
    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory
        self.settings = get_settings()

    def _build_context(self, session: Session) -> OrchestratorService:
        brd_repo = BRDRepository(session)
        workflow_repo = WorkflowRepository(session)
        task_repo = TaskRepository(session)
        audit_repo = AuditRepository(session)
        event_bus = DatabaseEventBus(audit_repo)
        return OrchestratorService(OrchestratorContext(session, brd_repo, workflow_repo, task_repo, audit_repo, event_bus))

    def create_workflow_from_brd(self, title: str, brd_content: str, max_retry: int | None = None) -> int:
        with self.session_factory() as session:
            brd_repo = BRDRepository(session)
            workflow_repo = WorkflowRepository(session)
            brd = brd_repo.create_brd(title, brd_content)
            workflow = workflow_repo.create_workflow(brd.id, self.settings.default_max_retry if max_retry is None else max_retry, {'title': title})
            session.commit()
            return workflow.id

    def run_workflow(self, workflow_id: int) -> dict:
        with self.session_factory() as session:
            orchestrator = self._build_context(session)
            workflow = orchestrator.ctx.workflow_repo.get(workflow_id)
            if workflow is None:
                raise ValueError(f'Workflow {workflow_id} not found')
            brd = orchestrator.ctx.brd_repo.get_brd(workflow.brd_id)
            if brd is None:
                raise ValueError(f'BRD {workflow.brd_id} not found')
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment. Real agents are required.")

            logger.info('Using real OpenAI agents')
            po = POAgent(self.settings.openai_api_key)
            dev = DevAgent(self.settings.openai_api_key)
            qc = QCAgent(self.settings.openai_api_key)

            nodes = WorkflowNodes(orchestrator, po, dev, qc)
            graph = WorkflowGraphFactory(nodes).build()
            initial_state = {
                'workflow_id': workflow.id,
                'brd_id': brd.id,
                'brd_content': brd.content,
                'feature_summary': '',
                'user_stories': [],
                'acceptance_criteria': [],
                'backlog': [],
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
                'timestamps': {'started_at': orchestrator.now_iso()},
                'blocked_tasks': [],
                'completed_tasks': [],
            }
            logger.info('Starting workflow execution %s', workflow.id)
            result = graph.invoke(initial_state, config={"recursion_limit": 50})
            logger.info('Workflow execution %s finished with status=%s', workflow.id, result.get('status'))
            return result

    def resume_workflow(self, workflow_id: int) -> dict:
        with self.session_factory() as session:
            orchestrator = self._build_context(session)
            workflow = orchestrator.ctx.workflow_repo.get(workflow_id)
            if workflow is None:
                raise ValueError(f'Workflow {workflow_id} not found')
            
            # Reset cancellation status if needed
            if workflow.status == WorkflowStatus.CANCELLED.value:
                transitions = orchestrator.ctx.audit_repo.list_transitions(workflow_id=workflow_id)
                # Take the status before 'CANCELLED' or default to BACKLOG_CREATED if we can't find one
                prev_status = WorkflowStatus.NEW.value
                for t in reversed(transitions):
                    if t.to_status != WorkflowStatus.CANCELLED.value:
                        prev_status = t.to_status
                        break
                orchestrator.ctx.workflow_repo.update_status(workflow, status=prev_status, current_agent=workflow.current_agent)
                session.commit()

            brd = orchestrator.ctx.brd_repo.get_brd(workflow.brd_id)
            feature = orchestrator.ctx.brd_repo.get_feature(workflow.id)
            user_stories = orchestrator.ctx.brd_repo.list_user_stories(workflow.id)
            
            # Reconstruct State
            tasks = orchestrator.ctx.task_repo.list_tasks(workflow_id)
            serialized_tasks = [orchestrator.serialize_task(t.id) for t in tasks]
            
            completed_tasks = [t for t in serialized_tasks if t['status'] == TaskStatus.DONE.value]
            blocked_tasks = [t for t in serialized_tasks if t['status'] == TaskStatus.BLOCKED.value]
            
            # Identify current task
            current_task = next((t for t in serialized_tasks if t['status'] in [TaskStatus.DEV_IN_PROGRESS.value, TaskStatus.QC_IN_PROGRESS.value, TaskStatus.REOPENED.value, TaskStatus.READY.value]), None)
            
            # Prepare task queue
            task_queue = [t for t in serialized_tasks if t['status'] in [TaskStatus.NEW.value, TaskStatus.READY.value, TaskStatus.REOPENED.value] and (current_task is None or t['id'] != current_task['id'])]
            task_queue.sort(key=lambda x: x['task_number'])

            # Bug reports
            all_bugs = []
            for t in tasks:
                bugs = orchestrator.ctx.task_repo.list_bugs_for_task(t.id)
                for b in bugs:
                    all_bugs.append({'id': b.id, 'title': b.title, 'description': b.description, 'severity': b.severity, 'failed_criteria': b.failed_criteria, 'task_id': t.id})

            backlog_items = orchestrator.ctx.task_repo.list_backlog_items(workflow.id)

            initial_state: WorkflowState = {
                'workflow_id': workflow.id,
                'brd_id': brd.id,
                'brd_content': brd.content,
                'feature_summary': feature.summary if feature else '',
                'user_stories': [{'id': s.id, 'title': s.title, 'description': s.description, 'priority': s.priority} for s in user_stories],
                'acceptance_criteria': [], # We don't strictly need the global list if they are in tasks/stories
                'backlog': [{'id': b.id, 'title': b.title, 'description': b.description, 'team': b.team} for b in backlog_items],
                'current_task': current_task,
                'task_queue': task_queue,
                'task_history': completed_tasks + blocked_tasks,
                'dev_output': current_task.get('output_context') if current_task else None,
                'qc_result': None, # Will be re-run if resuming at QC
                'bug_reports': all_bugs,
                'retry_count': current_task.get('retry_count', 0) if current_task else 0,
                'max_retry': workflow.max_retry,
                'current_agent': workflow.current_agent or AgentName.ORCHESTRATOR.value,
                'status': workflow.status,
                'event_logs': [],
                'timestamps': {'started_at': workflow.started_at.isoformat() if workflow.started_at else orchestrator.now_iso()},
                'blocked_tasks': blocked_tasks,
                'completed_tasks': completed_tasks,
            }

            logger.info('Resuming workflow execution %s from status=%s', workflow.id, workflow.status)
            
            po = POAgent(self.settings.openai_api_key)
            dev = DevAgent(self.settings.openai_api_key)
            qc = QCAgent(self.settings.openai_api_key)
            nodes = WorkflowNodes(orchestrator, po, dev, qc)
            graph = WorkflowGraphFactory(nodes).build()
            
            result = graph.invoke(initial_state, config={"recursion_limit": 50})
            logger.info('Workflow execution %s finished with status=%s', workflow.id, result.get('status'))
            return result

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
