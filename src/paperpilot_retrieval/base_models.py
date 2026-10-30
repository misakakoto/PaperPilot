"""Typed state and evidence models for the research workflow."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import Enum
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Stage(str, Enum):
    INITIALIZED = "initialized"
    PLANNED = "planned"
    SEARCHED = "searched"
    SYNTHESIZED = "synthesized"
    REFLECTED = "reflected"
    REPORTED = "reported"


class TaskStatus(str, Enum):
    PENDING = "pending"
    SEARCHING = "searching"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchTask(StrictModel):
    id: str = Field(pattern=r"^T\d+$")
    title: str = Field(min_length=2, max_length=120)
    objective: str = Field(min_length=3, max_length=500)
    search_query: str = Field(min_length=2, max_length=300)
    status: TaskStatus = TaskStatus.PENDING
    error: str | None = None


class ResearchPlan(StrictModel):
    rationale: str = Field(min_length=3, max_length=1000)
    tasks: list[ResearchTask] = Field(min_length=1, max_length=8)

    @field_validator("tasks")
    @classmethod
    def task_ids_must_be_unique(cls, tasks: list[ResearchTask]) -> list[ResearchTask]:
        ids = [task.id for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task IDs must be unique")
        return tasks


class RawSource(StrictModel):
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=2000)
    snippet: str = Field(default="", max_length=8000)
    content: str = Field(default="", max_length=30000)
    query: str = Field(default="", max_length=500)
    provider: str = Field(default="unknown", max_length=80)
    fetch_error: str | None = Field(default=None, max_length=500)

    @field_validator("url")
    @classmethod
    def source_url_must_be_public_web_scheme(cls, value: str) -> str:
        parsed = urlsplit(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source URL must be an absolute http(s) URL")
        return value.strip()

    @field_validator("snippet", "content")
    @classmethod
    def redact_contact_email(cls, value: str) -> str:
        """Do not persist third-party contact addresses in portfolio artifacts."""

        return EMAIL_PATTERN.sub("[EMAIL REDACTED]", value)


class Source(RawSource):
    id: str = Field(pattern=r"^S\d+$")
    retrieved_at: datetime = Field(default_factory=utc_now)


class Claim(StrictModel):
    text: str = Field(min_length=3, max_length=3000)
    source_ids: list[str] = Field(min_length=1, max_length=12)
    confidence: Literal["high", "medium", "low"] = "medium"
    limitations: str = Field(default="", max_length=1000)

    @field_validator("source_ids")
    @classmethod
    def source_ids_must_be_valid(cls, ids: list[str]) -> list[str]:
        if any(not item.startswith("S") or not item[1:].isdigit() for item in ids):
            raise ValueError("source IDs must use the S<number> format")
        return list(dict.fromkeys(ids))


class TaskFinding(StrictModel):
    task_id: str
    title: str
    summary: str = Field(min_length=1, max_length=10000)
    claims: list[Claim] = Field(default_factory=list, max_length=20)
    source_ids: list[str] = Field(default_factory=list)


class ReflectionDecision(StrictModel):
    sufficient: bool
    reason: str = Field(min_length=1, max_length=1200)
    follow_up_tasks: list[ResearchTask] = Field(default_factory=list, max_length=3)


class CitationAudit(StrictModel):
    valid: bool
    valid_source_ids: list[str] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)
    invalid_source_ids: list[str] = Field(default_factory=list)
    uncited_source_ids: list[str] = Field(default_factory=list)
    claim_like_sentences: int = 0
    cited_claim_like_sentences: int = 0
    coverage: float = 0.0
    minimum_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    meets_minimum_coverage: bool = True
    structured_claims: int = 0
    validly_bound_structured_claims: int = 0
    claim_binding_coverage: float = 1.0


class UsageStats(StrictModel):
    llm_calls: int = 0
    search_queries: int = 0
    fetched_pages: int = 0
    fetched_chars: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class RunState(StrictModel):
    schema_version: int = 1
    run_id: str
    topic: str
    stage: Stage = Stage.INITIALIZED
    citation_minimum_coverage: float = Field(default=0.65, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    plan: ResearchPlan | None = None
    sources: list[Source] = Field(default_factory=list)
    task_source_ids: dict[str, list[str]] = Field(default_factory=dict)
    findings: list[TaskFinding] = Field(default_factory=list)
    reflection: ReflectionDecision | None = None
    report_markdown: str | None = None
    citation_audit: CitationAudit | None = None
    usage: UsageStats = Field(default_factory=UsageStats)
    warnings: list[str] = Field(default_factory=list)


class DeepResearchResult(StrictModel):
    run_id: str
    topic: str
    report_markdown: str
    report_path: str
    checkpoint_path: str
    event_log_path: str
    citation_audit: CitationAudit
    task_completion_valid: bool
    usage: UsageStats
    warnings: list[str] = Field(default_factory=list)
