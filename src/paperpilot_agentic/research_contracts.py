"""Shared persisted research state and role permissions, independent of engines."""

from typing import Any, Literal, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

STATE_VERSION = "research-task-state-v1"
RESEARCH_ROLES = {
    "lead": {"read_source", "delegate_patch", "request_experiment", "delegate_analysis", "stop"},
    "engineer": {"read_code", "validate_patch", "submit_patch", "give_up"},
    "analyst": {"inspect_metrics", "report"},
}
DIAGNOSIS_ROLES = {
    "lead": {"inspect_experiment", "delegate_analysis", "finish"},
    "analyst": {"inspect_experiment", "report"},
}
FINISH_DECISION_VERSION = "diagnostic-finish-decision-v1"
FINISH_DECISIONS = {
    "STOP": "STOP",
    "PROPOSE_EXPERIMENT": "PROPOSE_EXPERIMENT",
    "NEED_MORE_EVIDENCE": "NEED_MORE_EVIDENCE",
    "STOP_INSUFFICIENT_EVIDENCE": "NEED_MORE_EVIDENCE",
}


class ResearchTaskState(BaseModel):
    # Engine-specific data stays owned by that engine, not flattened/lost here.
    model_config = ConfigDict(extra="allow", strict=True)
    state_contract: Literal["research-task-state-v1"] = STATE_VERSION
    goal: str = Field(min_length=1)
    phase: Literal[
        "READY",
        "WAITING_APPROVAL",
        "TRAINING",
        "PAUSED",
        "COMPLETED",
        "PARTIAL",
        "UNKNOWN",
        "FAILED",
        "CANCELLED",
    ]
    role: Literal["lead", "engineer", "analyst"]
    turn: int = Field(ge=0)
    completed: bool
    limits: dict[str, Any]
    contexts: dict[str, list[dict]]
    decisions: list[dict]
    candidates: list[dict]
    constraints: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    experiment_refs: list[dict] = Field(default_factory=list)
    pending_action: dict | None = None
    analysis_revisions: int = Field(default=0, ge=0, le=1)


def new_state(value):
    return ResearchTaskState.model_validate(value).model_dump()


def validate_state(value):
    """Legacy states remain readable and are never silently migrated."""
    if "state_contract" not in value:
        return value
    return ResearchTaskState.model_validate(value).model_dump()


def assurance(value):
    """Describe separate assurances; execution never certifies scientific prose."""
    verified = value.get("verification", {}).get("prediction_label_metrics")
    results = [c.get("result", {}) for c in value.get("candidates", [])]
    has_training = any(r.get("status") == "TRAINING_VERIFIED" for r in results)
    return {
        "execution": value["phase"],
        "facts": "HOST_RECOMPUTED"
        if verified == "HOST_RECOMPUTED" or has_training
        else "NOT_YET_VERIFIED",
        "inference": "NOT_INDEPENDENTLY_VERIFIED",
        "content_review": value.get("content_review", {}).get("verdict", "NOT_REVIEWED"),
        "scientific_conclusion": "NOT_ESTABLISHED_BY_WORKFLOW_COMPLETION",
    }


class StrictAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ActionEnvelope(StrictAction):
    action: str
    reason: str = Field(min_length=1)
    arguments: dict

    @field_validator("reason")
    @classmethod
    def nonblank_reason(cls, value):
        if not value.strip():
            raise ValueError("A nonempty public action reason is required")
        return value


class InspectArguments(StrictAction):
    view: Literal["paired", "coverage", "samples"]


class DelegateArguments(StrictAction):
    feedback: str = Field(default="", max_length=2000, pattern=r"^$|\S")


class RevisionArguments(StrictAction):
    feedback: str = Field(min_length=1, max_length=2000, pattern=r"\S")


class Hypothesis(StrictAction):
    text: str = Field(min_length=1)
    observation_ids: list[str] = Field(min_length=1)
    alternatives: list[str] = Field(min_length=1)


class DesignOutcome(StrictAction):
    observation: str = Field(min_length=1, pattern=r"\S")
    effect: Literal["SUPPORTS", "WEAKENS", "INCONCLUSIVE"]
    alternatives_remaining: list[str] = Field(min_length=1)


class ComparisonDesign(StrictAction):
    mode: Literal["SHARED_WEIGHTS_OPERATOR", "SEPARATELY_TRAINED_MODELS", "OBSERVATIONAL_DIAGNOSTIC"]
    target: Literal["baseline", "candidate", "paired_difference"]
    measured_variants: list[Literal["baseline", "candidate"]] = Field(min_length=1, max_length=2)
    weight_control: Literal["SHARED_CHECKPOINT", "PER_VARIANT_CHECKPOINT"]
    outcomes: list[DesignOutcome] = Field(min_length=2, max_length=8)

    @model_validator(mode="after")
    def measurement_covers_declared_comparison(self):
        measured = set(self.measured_variants)
        required = {"baseline", "candidate"} if self.target == "paired_difference" else {self.target}
        if len(measured) != len(self.measured_variants) or not required <= measured:
            raise ValueError("Measurements must cover the declared target with unique variants")
        if self.mode != "OBSERVATIONAL_DIAGNOSTIC" and measured != {"baseline", "candidate"}:
            raise ValueError("An operator or trained-model comparison must measure both variants")
        if ((self.mode == "SHARED_WEIGHTS_OPERATOR" and self.weight_control != "SHARED_CHECKPOINT") or
                (self.mode == "SEPARATELY_TRAINED_MODELS" and self.weight_control != "PER_VARIANT_CHECKPOINT")):
            raise ValueError("Weight control must match the declared comparison mode")
        return self


class AlternativePrediction(StrictAction):
    hypothesis: str = Field(min_length=1, pattern=r"\S")
    expected_outcomes: list[Literal["COMPATIBLE", "EXCLUDED", "UNKNOWN"]] = Field(
        min_length=2, max_length=8, description="One entry for each comparison outcome, in the same order")
    assumptions: list[str]

    @model_validator(mode="after")
    def exclusions_have_declared_assumptions(self):
        if any(not value.strip() for value in self.assumptions):
            raise ValueError("Declared assumptions must be nonempty")
        if "EXCLUDED" in self.expected_outcomes and not self.assumptions:
            raise ValueError("Exclusions require explicit assumptions, not just a label")
        return self


class IdentificationDesign(StrictAction):
    question: str = Field(min_length=1, pattern=r"\S")
    scope: Literal["FIXED_CHECKPOINT_RESPONSE", "TRAINING_INTERVENTION_EFFECT", "DESCRIPTIVE_ASSOCIATION"]
    intervention_checkpoint: Literal["baseline", "candidate", "NOT_APPLICABLE"]
    alternatives: list[AlternativePrediction] = Field(
        min_length=2, max_length=6, description="Primary hypothesis first, then named competing explanations")


class ContrastiveComparisonDesign(ComparisonDesign):
    identification: IdentificationDesign

    @model_validator(mode="after")
    def declared_predictions_support_outcome_labels(self):
        item = self.identification
        scopes = {"SHARED_WEIGHTS_OPERATOR": {"FIXED_CHECKPOINT_RESPONSE"},
                  "SEPARATELY_TRAINED_MODELS": {"TRAINING_INTERVENTION_EFFECT", "DESCRIPTIVE_ASSOCIATION"},
                  "OBSERVATIONAL_DIAGNOSTIC": {"DESCRIPTIVE_ASSOCIATION", "FIXED_CHECKPOINT_RESPONSE"}}[self.mode]
        if item.scope not in scopes:
            raise ValueError("Identification scope must match the declared intervention mode")
        shared = self.mode == "SHARED_WEIGHTS_OPERATOR"
        if shared == (item.intervention_checkpoint == "NOT_APPLICABLE"):
            raise ValueError("Declare one intervention checkpoint only for a shared-weight comparison")
        names = [row.hypothesis.strip().casefold() for row in item.alternatives]
        if len(set(names)) != len(names):
            raise ValueError("Competing hypotheses must have unique names")
        if any(len(row.expected_outcomes) != len(self.outcomes) for row in item.alternatives):
            raise ValueError("Predict every outcome for every hypothesis")
        for index, outcome in enumerate(self.outcomes):
            predictions = [row.expected_outcomes[index] for row in item.alternatives]
            if (outcome.effect == "SUPPORTS" and
                    not (predictions[0] == "COMPATIBLE" and "EXCLUDED" in predictions[1:])):
                raise ValueError("SUPPORTS must distinguish the primary hypothesis from a declared alternative")
            if (outcome.effect == "WEAKENS" and
                    not (predictions[0] == "EXCLUDED" and "COMPATIBLE" in predictions[1:])):
                raise ValueError("WEAKENS must distinguish an alternative from the primary hypothesis")
        return self

    def declaration_assessment(self):
        return {
            "assurance": "DECLARATIONS_CHECKED_NOT_CAUSAL_PROOF",
            "scope": self.identification.scope,
            "intervention_checkpoint": self.identification.intervention_checkpoint,
            "outcomes": [{
                "index": index,
                **{label.lower(): [i for i, row in enumerate(self.identification.alternatives)
                                  if row.expected_outcomes[index] == label]
                   for label in ("COMPATIBLE", "EXCLUDED", "UNKNOWN")},
            } for index in range(len(self.outcomes))],
        }


class ExperimentProposal(StrictAction):
    decision: Literal["PROPOSE_EXPERIMENT", "STOP_INSUFFICIENT_EVIDENCE"]
    hypothesis: str = Field(min_length=1)
    support_refs: list[str] = Field(min_length=1)
    counter_refs: list[str] = Field(min_length=1)
    missing_observations: list[str] = Field(min_length=1)
    falsification: str = Field(min_length=1)
    intervention: str = Field(min_length=1)
    controls: list[str] = Field(min_length=1)
    measurements: list[str] = Field(min_length=1)
    expected_patterns: list[str] = Field(min_length=1)
    alternatives_remaining: list[str] = Field(min_length=1)
    comparison_design: ComparisonDesign | None = None


class ScopedExperimentProposal(ExperimentProposal):
    model_config = ConfigDict(extra="forbid", strict=True, json_schema_extra={
        "if": {"properties": {"decision": {"const": "PROPOSE_EXPERIMENT"}}},
        "then": {"required": ["comparison_design"],
                 "properties": {"comparison_design": {"not": {"type": "null"}}}},
    })

    @model_validator(mode="after")
    def proposal_needs_comparison_design(self):
        if self.decision == "PROPOSE_EXPERIMENT" and self.comparison_design is None:
            raise ValueError("A scoped proposal requires an explicit comparison design")
        return self


class ReportArguments(StrictAction):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "if": {
                "properties": {
                    "next_experiment": {"properties": {"decision": {"const": "PROPOSE_EXPERIMENT"}}}
                }
            },
            "then": {"properties": {"hypotheses": {"minItems": 1}}},
        },
    )
    observation_ids: list[str] = Field(min_length=1)
    hypotheses: list[Hypothesis]
    limitations: list[str] = Field(min_length=1)
    next_experiment: ExperimentProposal
    fact_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def proposal_has_hypothesis(self):
        if self.next_experiment.decision == "PROPOSE_EXPERIMENT" and not self.hypotheses:
            raise ValueError("A proposal requires a hypothesis")
        return self


class ScopedReportArguments(ReportArguments):
    next_experiment: ScopedExperimentProposal


class ContrastiveExperimentProposal(ScopedExperimentProposal):
    comparison_design: ContrastiveComparisonDesign | None = None


class ContrastiveReportArguments(ScopedReportArguments):
    next_experiment: ContrastiveExperimentProposal


class ContentReview(StrictAction):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "if": {"properties": {"verdict": {"const": "NEEDS_REVISION"}}},
            "then": {"properties": {"issues": {"contains": {"type": "string", "pattern": r"\S"}}}},
        },
    )
    verdict: Literal["ACCEPT_TENTATIVE", "NEEDS_REVISION", "INSUFFICIENT_EVIDENCE"]
    issues: list[str]
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def revision_has_actionable_gap(self):
        if self.verdict == "NEEDS_REVISION" and not any(s.strip() for s in self.issues):
            raise ValueError("A revision requires explicit content issues")
        return self


class FinishArguments(StrictAction):
    decision: str = Field(json_schema_extra={"enum": list(FINISH_DECISIONS)})
    rationale: str = Field(min_length=1)
    content_review: ContentReview

    @field_validator("decision")
    @classmethod
    def known_decision(cls, value):
        if value not in FINISH_DECISIONS:
            raise ValueError("Select a registered finish decision")
        return value


SCIENTIFIC_REVIEW_VERSION = "scientific-review-v2"
CONTRASTIVE_REVIEW_VERSION = "scientific-review-v4"
SCOPED_REVIEW_PROTOCOLS = frozenset({"scientific-review-v3", CONTRASTIVE_REVIEW_VERSION})
SCIENTIFIC_REVIEW_PROTOCOLS = SCOPED_REVIEW_PROTOCOLS | {SCIENTIFIC_REVIEW_VERSION}
REVIEW_ASPECTS = ("source_scope", "causal_identification", "discriminating_measurement", "controlled_comparison")


class ScientificReviewCheck(StrictAction):
    aspect: Literal["source_scope", "causal_identification", "discriminating_measurement", "controlled_comparison"]
    status: Literal["SUPPORTED", "NEEDS_REVISION", "NOT_APPLICABLE"]
    statement: str = Field(min_length=1)
    basis_refs: list[str] = Field(min_length=1)


class AuditedContentReview(ContentReview):
    checks: list[ScientificReviewCheck] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def all_aspects_reviewed(self):
        if {item.aspect for item in self.checks} != set(REVIEW_ASPECTS):
            raise ValueError("Review each scientific aspect exactly once")
        return self


class AuditedFinishArguments(FinishArguments):
    content_review: AuditedContentReview


class ScopedReviewCheck(StrictAction):
    aspect: Literal["source_scope", "causal_identification", "discriminating_measurement", "controlled_comparison"]
    finding: Literal["NO_DEFECT", "LIMITATION", "DEFECT"]
    statement: str = Field(min_length=1)
    basis_refs: list[str] = Field(min_length=1)
    claim_path: str = ""
    claim_quote: str = ""
    correction: str = ""

    @model_validator(mode="after")
    def defect_is_actionable(self):
        if self.finding == "DEFECT" and not all(
                value.strip() for value in (self.claim_path, self.correction)):
            raise ValueError("A quoted defect requires claim_path and correction; the host resolves its quote")
        return self


class ScopedContentReview(StrictAction):
    verdict: Literal["ACCEPT_TENTATIVE", "NEEDS_REVISION", "INSUFFICIENT_EVIDENCE"]
    rationale: str = Field(min_length=1)
    checks: list[ScopedReviewCheck] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def all_aspects_reviewed(self):
        if {item.aspect for item in self.checks} != set(REVIEW_ASPECTS):
            raise ValueError("Review each scientific aspect exactly once")
        if self.verdict == "NEEDS_REVISION" and not any(c.finding == "DEFECT" for c in self.checks):
            raise ValueError("Revision requires a quoted defect; limitations alone do not require revision")
        return self


class ScopedFinishArguments(FinishArguments):
    content_review: ScopedContentReview


class RevisionContentReview(ScopedContentReview):
    verdict: Literal["NEEDS_REVISION"]


class ReviewRevisionArguments(StrictAction):
    content_review: RevisionContentReview


DIAGNOSIS_ARGUMENTS = {
    "inspect_experiment": InspectArguments,
    "delegate_analysis": DelegateArguments,
    "report": ReportArguments,
    "finish": FinishArguments,
}


def diagnosis_unread_views(state):
    available = ["paired", "coverage", "samples"] if state["comparison_mode"] == "tools" else ["paired"]
    return [view for view in available if view not in state["observations"]]


def diagnosis_arguments(role, state=None):
    """One state-dependent action set for prompt discovery and execution."""
    models = {action: DIAGNOSIS_ARGUMENTS[action] for action in sorted(DIAGNOSIS_ROLES[role])}
    if state is not None:
        if state.get("content_review_protocol") == SCIENTIFIC_REVIEW_VERSION and "finish" in models:
            models["finish"] = AuditedFinishArguments
        if state.get("content_review_protocol") in SCOPED_REVIEW_PROTOCOLS and "finish" in models:
            models["finish"] = ScopedFinishArguments
            if state.get("analysis"):
                models["review"] = ReviewRevisionArguments
        if state.get("content_review_protocol") in SCOPED_REVIEW_PROTOCOLS and "report" in models:
            models["report"] = (ContrastiveReportArguments
                if state.get("content_review_protocol") == CONTRASTIVE_REVIEW_VERSION else ScopedReportArguments)
        if not state.get("analysis"):
            models.pop("finish", None)
        elif "delegate_analysis" in models:
            models["delegate_analysis"] = RevisionArguments
        if not diagnosis_unread_views(state):
            models.pop("inspect_experiment", None)
    return models


def diagnosis_action_schema(role, state=None):
    """The prompt and parser use the same models, not two hand-written protocols."""
    schemas = {
        action: {"arguments": model.model_json_schema()}
        for action, model in diagnosis_arguments(role, state).items()
    }
    if state is not None and "inspect_experiment" in schemas:
        schemas["inspect_experiment"]["arguments"]["properties"]["view"]["enum"] = diagnosis_unread_views(state)
    return schemas


def diagnosis_envelope_schema(role, state):
    schema = ActionEnvelope.model_json_schema()
    schema["properties"]["action"]["enum"] = list(diagnosis_arguments(role, state))
    return schema


def parse_diagnosis_action(role, value, state=None):
    envelope = ActionEnvelope.model_validate(value)
    models = diagnosis_arguments(role, state)
    if envelope.action not in models:
        raise ValueError("Diagnostic action not available for this role and state")
    arguments = models[envelope.action].model_validate(envelope.arguments)
    if envelope.action in {"finish", "review"} and isinstance(
            arguments, (AuditedFinishArguments, ScopedFinishArguments, ReviewRevisionArguments)):
        from .research_review import scientific_review_context, review_claims
        context = scientific_review_context(state)
        basis = (set(context['review_reference_catalog']) if 'review_reference_catalog' in context else
                 set(context["basis"]) | set(context["registered_observation_ids"]) | set(context["numeric_source_ids"]))
        if any(ref not in basis for item in arguments.content_review.checks for ref in item.basis_refs):
            raise ValueError("Scientific review must cite an existing review basis ID")
        if isinstance(arguments, (ScopedFinishArguments, ReviewRevisionArguments)):
            claims = review_claims(state)
            for item in arguments.content_review.checks:
                if item.finding == "DEFECT" and (
                        item.claim_path not in claims or (item.claim_quote and item.claim_quote not in claims[item.claim_path])):
                    raise ValueError("A quoted defect must match an existing analysis claim")
                if item.finding == "DEFECT":
                    item.claim_quote = claims[item.claim_path]
    if (state is not None and envelope.action == "inspect_experiment" and
            arguments.view not in diagnosis_unread_views(state)):
        raise ValueError("Diagnostic view not available; select an unread permitted view")
    args = arguments.model_dump(exclude_unset=True)
    if envelope.action == "delegate_analysis" and args == {"feedback":""}:
        args = {}  # An optional empty initial instruction is the same as omission.
    return envelope.model_dump() | {"arguments": args}


class EmptyArguments(StrictAction):
    pass


def nonnullable(schema):
    result = {k: v for k, v in schema.items() if k != "default"}
    if "anyOf" in result:
        choices = [s for s in result.pop("anyOf") if s.get("type") != "null"]
        result.update(choices[0] if len(choices) == 1 else {"anyOf": choices})
    return result


class ExclusiveArguments(StrictAction):
    exclusive_fields: ClassVar[tuple[str, ...]]

    @model_validator(mode="after")
    def one_argument_variant(self):
        selected = self.model_fields_set & set(self.exclusive_fields)
        if len(selected) != 1 or getattr(self, next(iter(selected))) is None:
            raise ValueError(
                "Select exactly one non-null argument: " + ", ".join(self.exclusive_fields)
            )
        return self

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        schema["oneOf"] = []
        for field in cls.exclusive_fields:
            schema["properties"][field] = nonnullable(schema["properties"][field])
            schema["oneOf"].append(
                {
                    "required": [field],
                    "not": {
                        "anyOf": [
                            {"required": [other]}
                            for other in cls.exclusive_fields
                            if other != field
                        ]
                    },
                }
            )
        return schema


class ReadSourceArguments(ExclusiveArguments):
    exclusive_fields = ("source_ref", "source_id")
    source_ref: str | None = Field(default=None, min_length=1, pattern=r"\S")
    source_id: str | None = Field(default=None, min_length=1, pattern=r"\S")


class ResearchProposal(StrictAction):
    hypothesis: str = Field(min_length=1)
    prediction: str = Field(min_length=1)
    falsification: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    code_basis: str = Field(min_length=1)


class DelegatePatchArguments(ExclusiveArguments):
    exclusive_fields = ("proposal", "revision_feedback")
    proposal: ResearchProposal | None = None
    revision_feedback: str | None = Field(
        default=None, min_length=1, max_length=2000, pattern=r"\S"
    )


class FusionArguments(StrictAction):
    fusion_body: str = Field(min_length=1)
    review_notes: list[str] = Field(default_factory=list)


class SubmitPatchArguments(FusionArguments):
    review_notes: list[str] = Field(min_length=1)


class Interpretation(StrictAction):
    text: str = Field(min_length=1)
    fact_refs: list[str] = Field(min_length=1)


class ResearchReportArguments(StrictAction):
    facts: list[str]
    interpretations: list[Interpretation]
    limitations: list[str] = Field(min_length=1)
    next_experiment: str = Field(min_length=1)


class StopArguments(StrictAction):
    summary: str = Field(min_length=1)
    content_review: ContentReview | None = None


RESEARCH_ARGUMENTS = {
    "read_source": ReadSourceArguments,
    "delegate_patch": DelegatePatchArguments,
    "request_experiment": EmptyArguments,
    "delegate_analysis": EmptyArguments,
    "stop": StopArguments,
    "read_code": EmptyArguments,
    "validate_patch": FusionArguments,
    "submit_patch": SubmitPatchArguments,
    "give_up": EmptyArguments,
    "inspect_metrics": EmptyArguments,
    "report": ResearchReportArguments,
}


def needs_result_review(state):
    return any(
        c.get("analysis") and c.get("result", {}).get("status") == "TRAINING_VERIFIED"
        for c in state.get("candidates", [])
    )


def research_action_schema(role, *, reviewed_result=False):
    schemas = {
        action: {"arguments": RESEARCH_ARGUMENTS[action].model_json_schema()}
        for action in sorted(RESEARCH_ROLES[role])
    }
    if role == "lead" and reviewed_result:
        schemas["stop"]["arguments"]["required"].append("content_review")
        properties = schemas["stop"]["arguments"]["properties"]
        properties["content_review"] = nonnullable(properties["content_review"])
    return schemas


def parse_research_action(role, value, *, reviewed_result=False):
    envelope = ActionEnvelope.model_validate(value)
    if envelope.action not in RESEARCH_ROLES[role]:
        raise ValueError("Role lacks tool permission: " + envelope.action)
    arguments = RESEARCH_ARGUMENTS[envelope.action].model_validate(envelope.arguments)
    if envelope.action == "stop" and reviewed_result and arguments.content_review is None:
        raise ValueError("Stopping after verified experiments requires content_review")
    return envelope.model_dump() | {"arguments": arguments.model_dump(exclude_unset=True)}
