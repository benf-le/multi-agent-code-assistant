from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class BacklogItem(Base):
    __tablename__ = 'backlog_items'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workflow_id: Mapped[int] = mapped_column(ForeignKey('workflow_executions.id'), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    team: Mapped[str] = mapped_column(String(100), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default='NEW')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    tasks = relationship('Task', back_populates='backlog_item')


class Task(Base):
    __tablename__ = 'tasks'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workflow_id: Mapped[int] = mapped_column(ForeignKey('workflow_executions.id'), nullable=False)
    backlog_item_id: Mapped[Optional[int]] = mapped_column(ForeignKey('backlog_items.id'), nullable=True)
    task_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    assignee_team: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default='NEW')
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_retry: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    current_agent: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    required_markers: Mapped[list] = mapped_column(JSON, default=list)
    input_context: Mapped[dict] = mapped_column(JSON, default=dict)
    output_context: Mapped[dict] = mapped_column(JSON, default=dict)
    latest_bug_id: Mapped[Optional[int]] = mapped_column(ForeignKey('bug_reports.id'), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    workflow = relationship('WorkflowExecution', back_populates='tasks')
    backlog_item = relationship('BacklogItem', back_populates='tasks')
    bugs = relationship('BugReport', back_populates='task', foreign_keys='BugReport.task_id')
    acceptance_criteria = relationship('AcceptanceCriteria', back_populates='task')
