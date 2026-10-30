from __future__ import annotations

import dataclasses
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(*parts: Any, prefix: str = "sha256") -> str:
    payload = "\x1f".join(canonical_json(p) for p in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()}"


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class RunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class ResearchOutcome(StrEnum):
    """Research work outcome, separate from process status and factual acceptance.

    ANSWERED means requested work completed under its review contract. It is not
    an independent original-source review or a claim of scientific correctness.
    """

    ANSWERED = "ANSWERED"
    PARTIAL = "PARTIAL"
    UNRESOLVED = "UNRESOLVED"
    DEGRADED = "DEGRADED"


class AgentAction(StrEnum):
    SEARCH = "SEARCH"
    INSPECT = "INSPECT"
    VERIFY = "VERIFY"
    REFORMULATE = "REFORMULATE"
    SYNTHESIZE = "SYNTHESIZE"
    ABSTAIN = "ABSTAIN"


class StepStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ToolCallStatus(StrEnum):
    PENDING = "PENDING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    APPROVED = "APPROVED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


class ApprovalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class ClaimStatus(StrEnum):
    UNKNOWN = "UNKNOWN"
    PARTIAL = "PARTIAL"
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNRESOLVED = "UNRESOLVED"


class EvidenceType(StrEnum):
    TEXT = "TEXT"
    TABLE = "TABLE"
    FIGURE = "FIGURE"
    EQUATION = "EQUATION"
    CODE = "CODE"
    EXPERIMENT = "EXPERIMENT"
    METADATA = "METADATA"


class EvidencePolarity(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"


class RiskLevel(StrEnum):
    READ_ONLY = "READ_ONLY"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return {
            RiskLevel.READ_ONLY: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }[self]


@dataclass(slots=True)
class Budget:
    max_steps: int = 20
    max_tool_calls: int = 20
    max_llm_calls: int = 20
    max_tokens: int = 100_000
    max_seconds: float = 3_600.0
    max_risk: RiskLevel = RiskLevel.HIGH
    synthesis_reserve_tokens: int = 12_000
    repair_reserve_tokens: int = 8_000
    synthesis_reserve_calls: int = 1
    repair_reserve_calls: int = 1
    max_repair_calls: int = 1
    unlimited_llm_usage: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["max_risk"] = self.max_risk.value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "Budget":
        value = dict(value or {})
        if 'unlimited_llm_usage' in value and type(value['unlimited_llm_usage']) is not bool:
            raise ValueError('unlimited_llm_usage must be an explicit boolean')
        if "max_risk" in value and not isinstance(value["max_risk"], RiskLevel):
            value["max_risk"] = RiskLevel(str(value["max_risk"]))
        return cls(**value)


@dataclass(frozen=True, slots=True)
class LLMCallPolicy:
    """Stage-specific model policy.

    ``max_output_tokens`` is a provider-side hard ceiling. The durable runtime also
    exposes a client-side remaining-budget block to the model and enforces the run
    total before starting a new request.
    """

    stage: str
    max_output_tokens: int
    thinking_enabled: bool
    json_mode: bool = False
    max_repair_attempts: int = 0
    reserve_class: str = "general"


LLM_POLICIES: dict[str, LLMCallPolicy] = {
    "planner": LLMCallPolicy("planner", 6_000, False, True, 1, "general"),
    "controller": LLMCallPolicy("controller", 6_000, True, True, 1, "general"),
    "verifier": LLMCallPolicy("verifier", 4_000, False, True, 1, "general"),
    # Final synthesis is a schema-constrained grounding task. In practice
    # Qwen's thinking mode made short answers much longer and increased
    # claim-id/evidence-id swaps, so keep reasoning in the controller and make
    # the final renderer JSON-focused.
    "synthesizer": LLMCallPolicy("synthesizer", 8_000, False, True, 1, "synthesis"),
    "experiment_planner": LLMCallPolicy("experiment_planner", 6_000, False, True, 1, "general"),
    "json_repair": LLMCallPolicy("json_repair", 3_000, False, True, 0, "repair"),
}


@dataclass(slots=True)
class ResearchPlan:
    objective: str
    constraints: list[str] = field(default_factory=list)
    output_requirements: list[str] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    candidate_queries: list[str] = field(default_factory=list)
    planner_version: str = "research-plan-v2"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(slots=True)
class ClaimRecord:
    claim_id: str
    run_id: str
    statement: str
    status: ClaimStatus = ClaimStatus.UNKNOWN
    importance: float = 1.0
    confidence: float = 0.0
    gaps: list[str] = field(default_factory=list)
    required_evidence_types: list[EvidenceType] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["status"] = self.status.value
        data["required_evidence_types"] = [x.value for x in self.required_evidence_types]
        return data


@dataclass(slots=True)
class ClaimEdgeRecord:
    edge_id: str
    run_id: str
    source_claim_id: str
    target_claim_id: str
    relation: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceRecord:
    evidence_id: str
    run_id: str
    evidence_type: EvidenceType
    content: str
    claim_id: str | None = None
    polarity: EvidencePolarity = EvidencePolarity.SUPPORTS
    source_uri: str | None = None
    page: int | None = None
    bbox: list[float] | None = None
    object_type: str | None = None
    artifact_path: str | None = None
    confidence: float = 0.8
    provenance: dict[str, Any] = field(default_factory=dict)
    content_hash: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.content_hash:
            # Tool-call ids, retry counters and retrieval queries describe how an
            # item was obtained, not the identity of the evidence itself.  Keeping
            # them in the deduplication hash allowed the same passage to be counted
            # repeatedly after a retry and artificially inflated claim confidence.
            stable_provenance = {
                key: self.provenance[key]
                for key in (
                    "document_sha256",
                    "layout_object_id",
                    "paper_sha256",
                    "stable_id",
                    "protocol_hash",
                    "source_hash_after",
                    "parser_cache_identity",
                    "structured_extraction",
                )
                if self.provenance.get(key) is not None
            }
            self.content_hash = stable_hash(
                self.evidence_type.value,
                self.content,
                self.source_uri,
                self.page,
                self.bbox,
                self.object_type,
                stable_provenance,
            )

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["evidence_type"] = self.evidence_type.value
        data["polarity"] = self.polarity.value
        return data


@dataclass(slots=True)
class ActionCandidate:
    tool_name: str
    arguments: dict[str, Any]
    claim_id: str | None
    reason: str
    expected_gain: float
    token_cost: float = 0.0
    time_cost: float = 0.0
    risk_cost: float = 0.0
    gap: str | None = None
    # The controller optimizes evidence coverage, not merely the number of
    # retrieved chunks.  These terms are deliberately explicit so a trace can
    # explain why one Claim/tool was selected over another.
    coverage_gain: float = 0.0
    diversity_gain: float = 0.0
    uncertainty_reduction: float = 0.0
    utility_version: str = "expected-gain-per-cost-v1"

    @property
    def score(self) -> float:
        denominator = max(0.05, self.token_cost + self.time_cost + self.risk_cost)
        numerator = (
            self.expected_gain
            + self.coverage_gain
            + self.diversity_gain
            + self.uncertainty_reduction
        )
        return numerator / denominator

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["score"] = self.score
        return data


@dataclass(slots=True)
class ToolResult:
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolResult":
        return cls(**dict(value))


@dataclass(slots=True)
class ToolInvocationOutcome:
    status: ToolCallStatus
    tool_call_id: str
    result: ToolResult | None = None
    approval_id: str | None = None
    reused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "tool_call_id": self.tool_call_id,
            "result": self.result.to_dict() if self.result else None,
            "approval_id": self.approval_id,
            "reused": self.reused,
        }


@dataclass(slots=True)
class SandboxPolicy:
    backend: str = "auto"
    docker_image: str = "python:3.11-slim"
    timeout_seconds: int = 600
    cpu_seconds: int = 600
    memory_mb: int = 2048
    max_output_mb: int = 64
    max_processes: int = 64
    network: str = "none"
    # Unknown research code must fail closed when a real sandbox is unavailable.
    # Trusted development fixtures may explicitly opt in to the local fallback.
    allow_local_fallback: bool = False
    allowed_executables: tuple[str, ...] = (
        "python",
        "python3",
        "python.exe",
        "pytest",
        "pytest.exe",
    )
    env_allowlist: tuple[str, ...] = (
        "PATH",
        "LANG",
        "LC_ALL",
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
    )
    read_only_files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["allowed_executables"] = list(self.allowed_executables)
        data["env_allowlist"] = list(self.env_allowlist)
        if self.read_only_files:
            data['read_only_files'] = list(self.read_only_files)
        else:
            data.pop('read_only_files')  # Keep historical protocol fingerprints unchanged.
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "SandboxPolicy":
        data = dict(value or {})
        for key in ("allowed_executables", "env_allowlist"):
            if key in data:
                data[key] = tuple(data[key])
        if 'read_only_files' in data:
            if not isinstance(data['read_only_files'], (list, tuple)):
                raise ValueError('Sandbox read-only files must be a list')
            data['read_only_files'] = tuple(data['read_only_files'])
        return cls(**data)


@dataclass(slots=True)
class ExperimentProtocol:
    name: str
    command: list[str]
    source_dir: str
    output_dir: str
    metric_files: list[str] = field(default_factory=lambda: ["metrics.json"])
    acceptance_criteria: list[dict[str, Any]] = field(default_factory=list)
    hypothesis: str = ""
    seed: int | None = None
    environment: dict[str, str] = field(default_factory=dict)
    policy: SandboxPolicy = field(default_factory=SandboxPolicy)
    study_design: dict[str, Any] = field(default_factory=dict)
    seed_sweep: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["policy"] = self.policy.to_dict()
        if not self.study_design:
            data.pop('study_design', None)  # Preserve historical protocol fingerprints.
        if not self.seed_sweep:
            data.pop('seed_sweep', None)
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentProtocol":
        data = dict(value)
        data["policy"] = SandboxPolicy.from_dict(data.get("policy"))
        return cls(**data)

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.to_dict())


def ensure_relative_to(path: str | Path, root: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    root_path = Path(root).expanduser().resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"Path escapes allowed root: {candidate} not under {root_path}") from exc
    return candidate


def json_loads(value: str | bytes | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
