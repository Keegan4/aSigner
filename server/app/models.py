from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base

Json = JSON().with_variant(JSONB(), "postgresql")


def now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    google_sub: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(320))
    name: Mapped[str] = mapped_column(String(200), default="")
    avatar_url: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    attempts: Mapped[list["Attempt"]] = relationship(cascade="all, delete-orphan")
    progress: Mapped[list["Progress"]] = relationship(cascade="all, delete-orphan")
    translations: Mapped[list["Translation"]] = relationship(cascade="all, delete-orphan")


class Sign(Base):
    __tablename__ = "signs"

    id: Mapped[int] = mapped_column(primary_key=True)
    gloss: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    category: Mapped[str] = mapped_column(String(40), index=True)
    description: Mapped[str | None] = mapped_column(Text)
    demo_path: Mapped[str | None] = mapped_column(String(300))
    sign_bank_url: Mapped[str | None] = mapped_column(String(500))


class Lesson(Base):
    __tablename__ = "lessons"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(60), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(120))
    summary: Mapped[str] = mapped_column(Text, default="")
    position: Mapped[int] = mapped_column(Integer, default=0)

    items: Mapped[list["LessonSign"]] = relationship(
        order_by="LessonSign.position", cascade="all, delete-orphan"
    )


class LessonSign(Base):
    __tablename__ = "lesson_signs"

    lesson_id: Mapped[int] = mapped_column(
        ForeignKey("lessons.id", ondelete="CASCADE"), primary_key=True
    )
    sign_id: Mapped[int] = mapped_column(
        ForeignKey("signs.id", ondelete="CASCADE"), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer, default=0)

    sign: Mapped[Sign] = relationship()


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    sign_id: Mapped[int] = mapped_column(ForeignKey("signs.id", ondelete="CASCADE"))
    score: Mapped[float] = mapped_column(Float)
    passed: Mapped[bool]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Progress(Base):
    __tablename__ = "progress"
    __table_args__ = (UniqueConstraint("user_id", "sign_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    sign_id: Mapped[int] = mapped_column(ForeignKey("signs.id", ondelete="CASCADE"))
    mastery: Mapped[int] = mapped_column(Integer, default=0)
    streak: Mapped[int] = mapped_column(Integer, default=0)
    next_review_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Translation(Base):
    __tablename__ = "translations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    glosses: Mapped[list] = mapped_column(Json, default=list)
    sentence: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
