from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import AcceptanceCriteria, BacklogItem, BugReport, Task


class TaskRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_backlog_item(self, workflow_id: int, title: str, description: str, team: str, sequence: int) -> BacklogItem:
        item = BacklogItem(
            workflow_id=workflow_id,
            title=title,
            description=description,
            team=team,
            sequence=sequence,
            status='NEW',
        )
        self.session.add(item)
        self.session.flush()
        return item

    def create_task(self, workflow_id: int, backlog_item_id: int | None, title: str, description: str, assignee_team: str, max_retry: int, required_markers: list[str] | None = None, input_context: dict | None = None) -> Task:
        # Calculate task_number sequentially for this workflow
        stmt = select(func.count(Task.id)).where(Task.workflow_id == workflow_id)
        count = self.session.scalar(stmt) or 0

        task = Task(
            workflow_id=workflow_id,
            backlog_item_id=backlog_item_id,
            task_number=count + 1,
            title=title,
            description=description,
            assignee_team=assignee_team,
            max_retry=max_retry,
            required_markers=required_markers or [],
            input_context=input_context or {},
            status='NEW',
        )
        self.session.add(task)
        self.session.flush()
        return task

    def attach_acceptance_criterion(self, workflow_id: int, task_id: int, text: str) -> AcceptanceCriteria:
        criterion = AcceptanceCriteria(workflow_id=workflow_id, task_id=task_id, text=text)
        self.session.add(criterion)
        self.session.flush()
        return criterion

    def list_tasks(self, workflow_id: int | None = None) -> list[Task]:
        stmt = select(Task)
        if workflow_id is not None:
            stmt = stmt.where(Task.workflow_id == workflow_id)
        stmt = stmt.order_by(Task.id)
        return list(self.session.scalars(stmt).all())

    def get_task(self, task_id: int) -> Task | None:
        return self.session.get(Task, task_id)

    def get_acceptance_criteria_for_task(self, task_id: int) -> list[AcceptanceCriteria]:
        stmt = select(AcceptanceCriteria).where(AcceptanceCriteria.task_id == task_id).order_by(AcceptanceCriteria.id)
        return list(self.session.scalars(stmt).all())

    def update_task_status(self, task: Task, status: str, current_agent: str | None = None) -> Task:
        task.status = status
        task.current_agent = current_agent
        self.session.add(task)
        self.session.flush()
        return task

    def save_task_output(self, task: Task, output_context: dict) -> Task:
        task.output_context = output_context
        self.session.add(task)
        self.session.flush()
        return task

    def increment_retry(self, task: Task) -> Task:
        task.retry_count += 1
        self.session.add(task)
        self.session.flush()
        return task

    def create_bug(self, workflow_id: int, task_id: int, title: str, description: str, severity: str, failed_criteria: list[str]) -> BugReport:
        bug = BugReport(workflow_id=workflow_id, task_id=task_id, title=title, description=description, severity=severity, failed_criteria=failed_criteria)
        self.session.add(bug)
        self.session.flush()
        task = self.get_task(task_id)
        if task is not None:
            task.latest_bug_id = bug.id
            self.session.add(task)
            self.session.flush()
        return bug

    def get_bug(self, bug_id: int) -> BugReport | None:
        return self.session.get(BugReport, bug_id)

    def list_bugs_for_task(self, task_id: int) -> list[BugReport]:
        stmt = select(BugReport).where(BugReport.task_id == task_id).order_by(BugReport.id)
        return list(self.session.scalars(stmt).all())

    def list_backlog_items(self, workflow_id: int) -> list[BacklogItem]:
        stmt = select(BacklogItem).where(BacklogItem.workflow_id == workflow_id).order_by(BacklogItem.sequence)
        return list(self.session.scalars(stmt).all())
