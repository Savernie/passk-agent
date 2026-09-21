import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Boolean, Float, Integer, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    domain: Mapped[str] = mapped_column(String)
    agent_llm: Mapped[str] = mapped_column(String)
    user_llm: Mapped[str] = mapped_column(String)
    num_tasks: Mapped[int] = mapped_column(Integer)
    num_trials: Mapped[int] = mapped_column(Integer)
    config: Mapped[dict] = mapped_column(JSONB, default=dict)


class TaskAttempt(Base):
    __tablename__ = "task_attempts"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id"))
    task_id: Mapped[str] = mapped_column(String)

    passed: Mapped[bool] = mapped_column(Boolean)
    db_reward: Mapped[float] = mapped_column(Float)
    communicate_reward: Mapped[float] = mapped_column(Float)

    cost_usd: Mapped[float] = mapped_column(Float)
    agent_cost_usd: Mapped[float] = mapped_column(Float)
    user_cost_usd: Mapped[float] = mapped_column(Float)

    tool_call_count: Mapped[int] = mapped_column(Integer, default=0)
    read_action_count: Mapped[int] = mapped_column(Integer, default=0)
    write_action_count: Mapped[int] = mapped_column(Integer, default=0)

    trajectory: Mapped[dict] = mapped_column(JSONB)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True, default=None)