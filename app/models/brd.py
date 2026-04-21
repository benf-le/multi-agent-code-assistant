from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class BRD(Base):
    __tablename__ = 'brds'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    workflows = relationship('WorkflowExecution', back_populates='brd')


class Feature(Base):
    __tablename__ = 'features'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workflow_id: Mapped[int] = mapped_column(ForeignKey('workflow_executions.id'), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserStory(Base):
    __tablename__ = 'user_stories'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workflow_id: Mapped[int] = mapped_column(ForeignKey('workflow_executions.id'), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[str] = mapped_column(String(50), default='MEDIUM')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    acceptance_criteria = relationship('AcceptanceCriteria', back_populates='user_story')


class AcceptanceCriteria(Base):
    __tablename__ = 'acceptance_criteria'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workflow_id: Mapped[int] = mapped_column(ForeignKey('workflow_executions.id'), nullable=False)
    user_story_id: Mapped[Optional[int]] = mapped_column(ForeignKey('user_stories.id'), nullable=True)
    task_id: Mapped[Optional[int]] = mapped_column(ForeignKey('tasks.id'), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user_story = relationship('UserStory', back_populates='acceptance_criteria')
    task = relationship('Task', back_populates='acceptance_criteria')
