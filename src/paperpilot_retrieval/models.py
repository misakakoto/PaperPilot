"""Typed models for local-paper learning, evidence, and reproduction workflows."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .base_models import StrictModel, UsageStats

PAPER_ID_PATTERN = r"^P\d+$"
EVIDENCE_ID_PATTERN = r"^P\d+:p\d+(?:-p\d+)?:c\d+$"
SECTION_STATUS = Literal["present", "not_applicable", "insufficient_evidence"]


class PaperPage(StrictModel):
    number: int = Field(ge=1)
    text: str = Field(default="", max_length=100_000)
    image_count: int = Field(default=0, ge=0)


class PaperDocument(StrictModel):
    id: str = Field(pattern=PAPER_ID_PATTERN)
    file_name: str = Field(min_length=1, max_length=500)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    title: str = Field(min_length=1, max_length=1000)
    authors: list[str] = Field(default_factory=list, max_length=100)
    page_count: int = Field(ge=1)
    processed_pages: int = Field(ge=0)
    pages: list[PaperPage] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def page_inventory_must_be_consistent(self) -> PaperDocument:
        if self.processed_pages > self.page_count:
            raise ValueError("processed_pages cannot exceed page_count")
        if len(self.pages) != self.processed_pages:
            raise ValueError("pages length must equal processed_pages")
        expected = list(range(1, self.processed_pages + 1))
        if [page.number for page in self.pages] != expected:
            raise ValueError("paper pages must be consecutive and one-indexed")
        return self


class PaperChunk(StrictModel):
    id: str = Field(pattern=EVIDENCE_ID_PATTERN)
    paper_id: str = Field(pattern=PAPER_ID_PATTERN)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section: str = Field(default="未识别章节", max_length=300)
    text: str = Field(min_length=1, max_length=12_000)
    token_count_estimate: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def evidence_id_must_match_chunk_location(self) -> PaperChunk:
        import re

        match = re.fullmatch(
            r"(?P<paper>P\d+):p(?P<start>\d+)(?:-p(?P<end>\d+))?:c\d+",
            self.id,
        )
        assert match is not None
        encoded_end = int(match.group("end") or match.group("start"))
        if self.page_end < self.page_start:
            raise ValueError("page_end cannot be smaller than page_start")
        if (
            match.group("paper") != self.paper_id
            or int(match.group("start")) != self.page_start
            or encoded_end != self.page_end
        ):
            raise ValueError("chunk ID must match paper_id and page range")
        return self


class EvidencePoint(StrictModel):
    text: str = Field(min_length=2, max_length=4000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)
    confidence: Literal["high", "medium", "low"] = "medium"
    boundary: str = Field(default="", max_length=1500)

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_are_unique_and_well_formed(cls, values: list[str]) -> list[str]:
        import re

        if any(not re.fullmatch(EVIDENCE_ID_PATTERN, value) for value in values):
            raise ValueError("evidence IDs must use P<number>:p<number>:c<number> format")
        return list(dict.fromkeys(values))


class ModelComponent(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    role: str = Field(min_length=2, max_length=2000)
    inputs: list[str] = Field(default_factory=list, max_length=20)
    outputs: list[str] = Field(default_factory=list, max_length=20)
    operations: list[str] = Field(default_factory=list, max_length=30)
    depends_on: list[str] = Field(default_factory=list, max_length=20)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)

    _validate_evidence = field_validator("evidence_ids")(
        EvidencePoint.evidence_ids_are_unique_and_well_formed.__func__
    )


class ExperimentRecord(StrictModel):
    question: str = Field(min_length=2, max_length=1000)
    setup: str = Field(min_length=2, max_length=2500)
    dataset: str = Field(default="未明确", max_length=1000)
    baselines: list[str] = Field(default_factory=list, max_length=30)
    metrics: list[str] = Field(default_factory=list, max_length=30)
    result: str = Field(default="论文片段中未找到明确结果", max_length=2500)
    interpretation: str = Field(default="", max_length=2500)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)

    _validate_evidence = field_validator("evidence_ids")(
        EvidencePoint.evidence_ids_are_unique_and_well_formed.__func__
    )


class LearningCard(StrictModel):
    question: str = Field(min_length=2, max_length=1000)
    answer: str = Field(min_length=2, max_length=3000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)

    _validate_evidence = field_validator("evidence_ids")(
        EvidencePoint.evidence_ids_are_unique_and_well_formed.__func__
    )


class PaperProfile(StrictModel):
    paper_id: str = Field(pattern=PAPER_ID_PATTERN)
    title: str = Field(min_length=1, max_length=1000)
    one_sentence_summary: str = Field(min_length=2, max_length=2000)
    summary_evidence_ids: list[str] = Field(min_length=1, max_length=8)
    research_problem: EvidencePoint
    prerequisites: list[str] = Field(default_factory=list, max_length=30)
    contributions: list[EvidencePoint] = Field(default_factory=list, max_length=20)
    model_section_status: SECTION_STATUS
    model_section_reason: str = Field(min_length=2, max_length=1500)
    model_components: list[ModelComponent] = Field(default_factory=list, max_length=30)
    training_objectives: list[EvidencePoint] = Field(default_factory=list, max_length=20)
    experiment_section_status: SECTION_STATUS
    experiment_section_reason: str = Field(min_length=2, max_length=1500)
    experiments: list[ExperimentRecord] = Field(default_factory=list, max_length=30)
    limitations: list[EvidencePoint] = Field(default_factory=list, max_length=20)
    learning_cards: list[LearningCard] = Field(default_factory=list, max_length=20)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("model_components")
    @classmethod
    def component_dependencies_must_resolve(
        cls, components: list[ModelComponent]
    ) -> list[ModelComponent]:
        names = [component.name.casefold() for component in components]
        if len(names) != len(set(names)):
            raise ValueError("model component names must be unique")
        known = set(names)
        for component in components:
            dependencies = {item.casefold() for item in component.depends_on}
            unknown = dependencies - known
            if unknown:
                raise ValueError(
                    f"component {component.name!r} has unknown dependencies: {sorted(unknown)}"
                )
            if component.name.casefold() in dependencies:
                raise ValueError(f"component {component.name!r} cannot depend on itself")
        graph = {
            component.name.casefold(): {
                item.casefold() for item in component.depends_on
            }
            for component in components
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError("model component dependencies cannot contain a cycle")
            if name in visited:
                return
            visiting.add(name)
            for dependency in graph[name]:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in graph:
            visit(name)
        return components

    _validate_summary_evidence = field_validator("summary_evidence_ids")(
        EvidencePoint.evidence_ids_are_unique_and_well_formed.__func__
    )

    @model_validator(mode="after")
    def declared_sections_must_match_content(self) -> PaperProfile:
        if self.model_section_status == "present" and not self.model_components:
            raise ValueError("model_section_status=present requires model_components")
        if self.experiment_section_status == "present" and not self.experiments:
            raise ValueError("experiment_section_status=present requires experiments")
        return self


class ReproductionStep(StrictModel):
    id: str = Field(pattern=r"^R\d+$")
    title: str = Field(min_length=2, max_length=300)
    action: str = Field(min_length=2, max_length=3000)
    inputs: list[str] = Field(default_factory=list, max_length=20)
    outputs: list[str] = Field(default_factory=list, max_length=20)
    verification: str = Field(min_length=2, max_length=2000)
    evidence_ids: list[str] = Field(min_length=1, max_length=12)

    _validate_evidence = field_validator("evidence_ids")(
        EvidencePoint.evidence_ids_are_unique_and_well_formed.__func__
    )


class ReproductionPlan(StrictModel):
    readiness: Literal["ready", "partial", "blocked"]
    readiness_reason: str = Field(min_length=2, max_length=2000)
    suggested_project_tree: list[str] = Field(default_factory=list, max_length=50)
    environment: list[str] = Field(default_factory=list, max_length=40)
    data_preparation: list[str] = Field(default_factory=list, max_length=40)
    steps: list[ReproductionStep] = Field(default_factory=list, max_length=30)
    acceptance_checks: list[str] = Field(default_factory=list, max_length=30)
    missing_information: list[str] = Field(default_factory=list, max_length=30)
    risks: list[str] = Field(default_factory=list, max_length=30)

    @field_validator("environment", "data_preparation")
    @classmethod
    def engineering_suggestions_must_be_labeled(cls, values: list[str]) -> list[str]:
        if any(not value.strip().startswith("SUGGESTED:") for value in values):
            raise ValueError(
                "environment and data_preparation items must start with SUGGESTED:"
            )
        return values

    @field_validator("acceptance_checks")
    @classmethod
    def acceptance_checks_must_be_labeled(cls, values: list[str]) -> list[str]:
        if any(not value.strip().startswith("CHECK:") for value in values):
            raise ValueError("acceptance_checks items must start with CHECK:")
        return values

    @field_validator("missing_information")
    @classmethod
    def unknowns_must_be_labeled(cls, values: list[str]) -> list[str]:
        if any(not value.strip().startswith("TODO_UNKNOWN:") for value in values):
            raise ValueError("missing_information items must start with TODO_UNKNOWN:")
        return values

    @field_validator("risks")
    @classmethod
    def risks_must_be_labeled(cls, values: list[str]) -> list[str]:
        if any(not value.strip().startswith("RISK:") for value in values):
            raise ValueError("risks items must start with RISK:")
        return values

    @model_validator(mode="after")
    def readiness_must_match_plan(self) -> ReproductionPlan:
        if self.readiness == "ready" and (not self.steps or not self.acceptance_checks):
            raise ValueError("readiness=ready requires steps and acceptance_checks")
        if self.readiness == "blocked" and not self.missing_information:
            raise ValueError("readiness=blocked requires missing_information")
        return self


class CodeSymbol(StrictModel):
    file: str = Field(min_length=1, max_length=1000)
    file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    qualified_name: str = Field(min_length=1, max_length=500)
    kind: Literal["class", "function", "async_function", "config", "document"]
    synopsis: str = Field(default="", max_length=1500)

    @model_validator(mode="after")
    def line_range_must_be_ordered(self) -> CodeSymbol:
        if self.line_end < self.line_start:
            raise ValueError("line_end cannot be smaller than line_start")
        return self


class ComponentCodeMapping(StrictModel):
    paper_id: str = Field(pattern=PAPER_ID_PATTERN)
    component_name: str = Field(min_length=1, max_length=300)
    status: Literal["exact", "candidate", "missing"]
    code_refs: list[str] = Field(default_factory=list, max_length=20)
    rationale: str = Field(min_length=2, max_length=2000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)

    _validate_evidence = field_validator("evidence_ids")(
        EvidencePoint.evidence_ids_are_unique_and_well_formed.__func__
    )


class PaperAnalysis(StrictModel):
    document_ids: list[str] = Field(min_length=1, max_length=20)
    profile_evidence_scopes: dict[str, list[str]] = Field(default_factory=dict)
    synthesis_evidence_scope: list[str] = Field(default_factory=list)
    profiles: list[PaperProfile] = Field(min_length=1, max_length=20)
    cross_paper_comparison: list[EvidencePoint] = Field(default_factory=list, max_length=30)
    reproduction: ReproductionPlan
    code_mappings: list[ComponentCodeMapping] = Field(default_factory=list, max_length=100)

    @field_validator("document_ids")
    @classmethod
    def document_ids_are_unique(cls, values: list[str]) -> list[str]:
        import re

        if any(not re.fullmatch(PAPER_ID_PATTERN, value) for value in values):
            raise ValueError("document IDs must use P<number> format")
        return list(dict.fromkeys(values))

    @field_validator("profile_evidence_scopes")
    @classmethod
    def profile_scopes_are_well_formed(
        cls, values: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        import re

        for paper_id, evidence_ids in values.items():
            if not re.fullmatch(PAPER_ID_PATTERN, paper_id):
                raise ValueError("profile scope keys must use P<number> format")
            if any(not re.fullmatch(EVIDENCE_ID_PATTERN, item) for item in evidence_ids):
                raise ValueError("profile scopes contain malformed evidence IDs")
            values[paper_id] = list(dict.fromkeys(evidence_ids))
        return values

    _validate_synthesis_scope = field_validator("synthesis_evidence_scope")(
        EvidencePoint.evidence_ids_are_unique_and_well_formed.__func__
    )

    @model_validator(mode="after")
    def provenance_keys_must_be_consistent(self) -> PaperAnalysis:
        known = set(self.document_ids)
        profile_ids = [profile.paper_id for profile in self.profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("paper profiles must have unique paper_id values")
        if not set(profile_ids).issubset(known):
            raise ValueError("paper profiles must belong to document_ids")
        if not set(self.profile_evidence_scopes).issubset(known):
            raise ValueError("profile evidence scopes must belong to document_ids")
        for paper_id, evidence_ids in self.profile_evidence_scopes.items():
            if any(item.split(":", 1)[0] != paper_id for item in evidence_ids):
                raise ValueError("profile evidence scope IDs must belong to their paper")
        if any(
            item.split(":", 1)[0] not in known
            for item in self.synthesis_evidence_scope
        ):
            raise ValueError("synthesis evidence scope references an unknown paper")
        component_keys = {
            (profile.paper_id, component.name.casefold())
            for profile in self.profiles
            for component in profile.model_components
        }
        mapping_key_list = [
            (mapping.paper_id, mapping.component_name.casefold())
            for mapping in self.code_mappings
        ]
        if len(mapping_key_list) != len(set(mapping_key_list)):
            raise ValueError("code mappings must be unique per paper component")
        mapping_keys = set(mapping_key_list)
        if not mapping_keys.issubset(component_keys):
            unexpected = sorted(mapping_keys - component_keys)
            raise ValueError(
                "code mappings must reference declared paper model components: "
                f"{unexpected}"
            )
        return self


class PaperSynthesis(StrictModel):
    cross_paper_comparison: list[EvidencePoint] = Field(default_factory=list, max_length=30)
    reproduction: ReproductionPlan
    code_mappings: list[ComponentCodeMapping] = Field(default_factory=list, max_length=100)


class PaperAnswer(StrictModel):
    answer: str = Field(min_length=2, max_length=8000)
    key_points: list[EvidencePoint] = Field(default_factory=list, max_length=20)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    uncertainties: list[str] = Field(default_factory=list, max_length=20)
    follow_up_questions: list[str] = Field(default_factory=list, max_length=10)

    _validate_evidence = field_validator("evidence_ids")(
        EvidencePoint.evidence_ids_are_unique_and_well_formed.__func__
    )


class PaperRetrievalPlan(StrictModel):
    standalone_question: str = Field(min_length=2, max_length=2_000)
    search_queries: list[str] = Field(min_length=1, max_length=6)

    @field_validator("search_queries")
    @classmethod
    def queries_are_non_empty_and_unique(cls, values: list[str]) -> list[str]:
        cleaned = [re.sub(r"\s+", " ", value).strip()[:500] for value in values]
        cleaned = [value for value in cleaned if value]
        if not cleaned:
            raise ValueError("search_queries must contain at least one non-empty query")
        return list(dict.fromkeys(cleaned))


class PaperResearchDecision(StrictModel):
    """One bounded, auditable action selected after observing retrieved evidence."""

    action: Literal["answer", "search_more"]
    decision_summary: str = Field(min_length=2, max_length=1000)
    unresolved_gaps: list[str] = Field(default_factory=list, max_length=8)
    follow_up_queries: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("unresolved_gaps", "follow_up_queries")
    @classmethod
    def clean_bounded_text_lists(cls, values: list[str]) -> list[str]:
        cleaned = [re.sub(r"\s+", " ", value).strip()[:500] for value in values]
        return list(dict.fromkeys(value for value in cleaned if value))

    @model_validator(mode="after")
    def action_must_match_queries(self) -> PaperResearchDecision:
        if self.action == "search_more" and not self.follow_up_queries:
            raise ValueError("search_more requires at least one follow_up_query")
        if self.action == "answer" and self.follow_up_queries:
            raise ValueError("answer cannot include follow_up_queries")
        return self


class PaperRouteCandidate(StrictModel):
    paper_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    title: str = Field(min_length=1, max_length=1000)
    score: float = Field(ge=0.0)
    matched_query_indexes: list[int] = Field(default_factory=list, max_length=20)
    best_stable_ids: list[str] = Field(default_factory=list, max_length=20)


class PaperQueryObservation(StrictModel):
    query_index: int = Field(ge=1)
    query: str = Field(min_length=1, max_length=500)
    routed_chunk_hits: int = Field(default=0, ge=0)
    workspace_chunk_hits: int = Field(default=0, ge=0)


class PaperResearchStep(StrictModel):
    round: int = Field(ge=1, le=3)
    evidence_count_before: int = Field(ge=0)
    paper_count_before: int = Field(ge=0)
    action: Literal["answer", "search_more", "stop"]
    decision_summary: str = Field(min_length=2, max_length=1000)
    unresolved_gaps: list[str] = Field(default_factory=list, max_length=8)
    follow_up_queries: list[str] = Field(default_factory=list, max_length=4)
    new_evidence_count: int = Field(default=0, ge=0)
    evidence_count_after: int = Field(ge=0)
    stop_reason: Literal[
        "controller_sufficient",
        "controller_disabled",
        "round_limit",
        "duplicate_queries",
        "no_new_evidence",
    ] | None = None


class PaperResearchTrace(StrictModel):
    """Text-free controller trace; excerpts remain in the evidence ledger only."""

    schema_version: Literal[1] = 1
    strategy: Literal["bounded_react_v1"] = "bounded_react_v1"
    max_research_rounds: int = Field(ge=0, le=2)
    initial_queries: list[str] = Field(min_length=1, max_length=8)
    all_queries: list[str] = Field(min_length=1, max_length=16)
    route_candidates: list[PaperRouteCandidate] = Field(default_factory=list, max_length=50)
    query_observations: list[PaperQueryObservation] = Field(
        default_factory=list, max_length=16
    )
    steps: list[PaperResearchStep] = Field(default_factory=list, max_length=3)
    final_evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    final_paper_count: int = Field(default=0, ge=0)
    termination: Literal[
        "controller_sufficient",
        "controller_disabled",
        "round_limit",
        "duplicate_queries",
        "no_new_evidence",
    ]


class PaperCitationAudit(StrictModel):
    valid: bool
    evidence_valid: bool = False
    content_complete: bool = False
    valid_evidence_ids: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    invalid_evidence_ids: list[str] = Field(default_factory=list)
    misbound_evidence_ids: list[str] = Field(default_factory=list)
    out_of_scope_evidence_ids: list[str] = Field(default_factory=list)
    evidence_points: int = 0
    supported_evidence_points: int = 0
    structured_support_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    cross_paper_comparison_valid: bool = True
    component_mapping_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    retrieval_scope_valid: bool = True
    reproduction_complete: bool = False
    paper_profile_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_content_sections: list[str] = Field(default_factory=list)


class PaperAnswerAudit(StrictModel):
    valid: bool
    valid_evidence_ids: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    invalid_evidence_ids: list[str] = Field(default_factory=list)
    evidence_points: int = Field(default=0, ge=0)
    supported_evidence_points: int = Field(default=0, ge=0)
    support_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    has_supported_answer: bool = False


class PaperAnalysisResult(StrictModel):
    run_id: str
    report_path: str
    analysis_path: str
    architecture_diagram_path: str
    mindmap_path: str
    code_map_path: str | None = None
    learning_cards_path: str
    experiment_matrix_path: str
    reproduction_plan_path: str
    paper_count: int
    chunk_count: int
    audit: PaperCitationAudit
    usage: UsageStats
    warnings: list[str] = Field(default_factory=list)


class PaperQuestionResult(StrictModel):
    run_id: str
    answer_path: str
    research_trace_path: str
    paper_count: int
    retrieved_chunks: int
    audit: PaperAnswerAudit
    invalid_evidence_ids: list[str] = Field(default_factory=list)
    usage: UsageStats
    warnings: list[str] = Field(default_factory=list)
