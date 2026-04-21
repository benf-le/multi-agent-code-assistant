from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class WorkflowExecution(Base):
    __tablename__ = 'workflow_executions'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    brd_id: Mapped[int] = mapped_column(ForeignKey('brds.id'), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default='NEW')
    current_agent: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    max_retry: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    brd = relationship('BRD', back_populates='workflows')
    tasks = relationship('Task', back_populates='workflow')
