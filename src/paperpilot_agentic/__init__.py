"""Durable, evidence-driven research-agent runtime for PaperPilot.

The package is deliberately dependency-light. Optional integrations (OpenTelemetry,
PyMuPDF, Docker) are detected at runtime and degrade explicitly rather than silently.
"""
from .models import (
    ApprovalStatus,
    ClaimStatus,
    EvidenceType,
    RiskLevel,
    RunStatus,
    StepStatus,
    ToolCallStatus,
)
from .orchestrator import ResearchAgentRuntime

__all__ = [
    "ApprovalStatus",
    "ClaimStatus",
    "EvidenceType",
    "ResearchAgentRuntime",
    "RiskLevel",
    "RunStatus",
    "StepStatus",
    "ToolCallStatus",
]

__version__ = "0.1.0"
