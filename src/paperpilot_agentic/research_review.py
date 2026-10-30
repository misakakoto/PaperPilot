"""Bounded model content review, separate from deterministic fact verification."""

import json
from .diagnostic_facts import observation_facts
from .research_contracts import SCOPED_REVIEW_PROTOCOLS, CONTRASTIVE_REVIEW_VERSION, ContrastiveComparisonDesign


def review_claims(state):
    """Exact narrative fields and list records; a quote does not validate criticism."""
    result = {}
    def visit(value, path, *, record=False):
        if isinstance(value, dict):
            if record:
                result[path] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            for key, item in value.items():
                if not key.endswith('_refs') and key not in {'observation_ids', 'decision'}:
                    visit(item, path + '/' + key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + '/' + str(index), record=True)
        elif isinstance(value, str):
            result[path] = value
    analysis = state.get('analysis') or {}
    for key in ('hypotheses', 'experiment_contract'):
        visit(analysis.get(key, []), 'analysis/' + key)
    return result


def scientific_review_context(state):
    """Host scope and analysis references, not host verdicts on model reasoning."""
    analysis = state.get("analysis") or {}
    observed = observation_facts(state.get("observations", {}))
    context = {
        "version": state.get('content_review_protocol', 'scientific-review-v2'),
        "registered_observation_ids": list(observed),
        "numeric_source_ids": sorted({ref for item in observed.values() for ref in item['source_refs']}),
        "basis": {
            "scope/validation_ids": {"status": "IDENTICAL_ACROSS_SEEDS",
                "meaning": "Same frozen evaluation samples and labels, not independently verified training fairness"},
            "scope/coverage_definition": {"status": "LABEL_POSITIVE_PIXEL_FRACTION_NOT_OBJECT_SIZE"},
            "scope/training_assurance": state.get("verification", {}).get("training_source"),
            "scope/unobserved": ["activations", "probabilities", "causal attribution", "statistical equivalence"],
            "analysis/hypotheses": analysis.get("hypotheses", []),
            "analysis/experiment_contract": analysis.get("experiment_contract", {}),
        },
        "review_instructions": [
            "basis_refs select exact basis keys, registered_observation_ids or numeric_source_ids; never invent a path",
            "For each aspect cite basis IDs and assess the actual claim; do not just repeat a disclaimer",
            "Reject explanations that contradict identical evaluation samples or equate coverage with object size",
            "Missing correlation, consistent seed signs or a nonsignificant result cannot uniquely identify or exclude a cause",
            "Check that proposed measurements distinguish named alternatives; state residual ambiguity",
            "Distinguish frozen separately trained weights from shared-weight operator interventions; do not conflate the controls",
            "A conditional unproven hypothesis is permitted; SUPPORTED means bounded proposal, not verified mechanism",
            "An unmeasured quantity is an allowed limitation when the proposal explicitly measures it; lack of proof alone is not NEEDS_REVISION",
            "NEEDS_REVISION must identify an actual contradictory or unsupported assertion, or a defective intervention/control/prediction, and state a correction",
            "Any identified issue requires specific revision; checks are model judgments, not host scientific verification",
        ],
    }
    if context['version'] in SCOPED_REVIEW_PROTOCOLS:
        context['assessment_policy'] = {
            'version': 'claim-before-disposition-v1',
            'order': ['check_report_claims', 'request_bounded_revision_or_finish'],
            'stop_exempts_report_from_review': False,
            'unexecuted_measurement_is_not_itself_defect': True,
            'intervention_response_is_not_historical_cause_identification': True,
            'assurance': 'MODEL_REVIEW_GUIDANCE_NOT_HOST_SCIENTIFIC_VERDICT',
        }
        if analysis.get('experiment_contract'):
            context['basis']['scope/proposal_timing'] = {
                'path': 'analysis/experiment_contract',
                'status': 'UNEXECUTED_RESEARCH_PLAN',
                'execution_authority': 'NONE_IN_THIS_DIAGNOSIS',
                'comparison_design_applies_to': 'PROPOSED_EXPERIMENT_NOT_OBSERVATION_PROVENANCE',
                'assurance': 'Field purpose only; does not certify design validity or historical training',
            }
        if context['version'] == CONTRASTIVE_REVIEW_VERSION:
            design = (analysis.get('experiment_contract') or {}).get('comparison_design')
            context['basis']['scope/identification_structure'] = (
                ContrastiveComparisonDesign.model_validate(design).declaration_assessment() if design else {
                    'status': 'NOT_DECLARED', 'assurance': 'DECLARATIONS_CHECKED_NOT_CAUSAL_PROOF'})
            context['review_instructions'] += [
                'identification.alternatives are MODEL PREDICTIONS, not observations; the first is primary. Entries align with comparison_design.outcomes in order',
                'Review each EXCLUDED entry and its assumptions: could the named alternative still produce that outcome? Structural separation does not prove exclusion',
                'Compare the declared question, scope and intervention_checkpoint with every narrative claim. Acute checkpoint response cannot silently become identification of a historical training cause',
                'All-COMPATIBLE or UNKNOWN predictions are allowed as an INCONCLUSIVE diagnostic; do not demand causal discrimination from every useful exploratory measurement',
            ]
        references = {key: 'MODEL_CONTEXT' if key.startswith('analysis/') else 'HOST_SCOPE'
                      for key in context['basis']}
        references.update({key: 'HOST_OBSERVATION' for key in context.pop('registered_observation_ids')})
        references.update({key: 'HOST_NUMERIC' for key in context.pop('numeric_source_ids')})
        for path in review_claims(state):
            references[path] = 'MODEL_CLAIM'
            parts = path.split('/')
            for end in range(2, len(parts)):
                references.setdefault('/'.join(parts[:end]), 'MODEL_CONTEXT')
        context['review_reference_catalog'] = references
        context['review_instructions'][0] = (
            'basis_refs select exact review_reference_catalog keys. MODEL_CLAIM/MODEL_CONTEXT '
            'identify the text being reviewed, not independently measured evidence. Never invent a path')
        context['review_instructions'] += [
            'A PROPOSE_EXPERIMENT must declare comparison_design: what is targeted, which variants are measured, weight control and possible outcomes',
            'Declared target coverage and weight consistency are structure checks, not proof that the intervention identifies a cause',
            'Compare the declared target with the hypothesis and falsification text; measuring a modified variant alone cannot describe the unmeasured original variant',
            'For each outcome check whether it distinguishes named alternatives; preserve INCONCLUSIVE when measurements cannot decide',
            'Shared-weight operator substitution and comparing separately trained checkpoints answer different questions; controls must not conflate them',
            'scope/proposal_timing identifies an unexecuted plan. Its design may differ from the historical observations; that difference alone is not a defect',
            'Check consistency within the proposed intervention, controls and design. A claim that existing observations already came from that intervention still needs execution evidence; the proposal cannot supply its own provenance',
            'Classify each check as NO_DEFECT, LIMITATION or DEFECT. A missing proposed measurement is normally LIMITATION, not a defect',
            'For DEFECT select a MODEL_CLAIM claim_path and give a concrete correction; the host resolves its quote, so claim_quote can be omitted',
            'The claim text is in independent_analysis; the reference catalog is navigation, not additional evidence',
            'LIMITATION can accompany ACCEPT_TENTATIVE and must stay visible in the final report; acceptance never proves a mechanism',
            'Use INSUFFICIENT_EVIDENCE for a justified stop, not to require proof before proposing a conditional experiment',
            'Check the report claims before choosing a final disposition. Stopping does not repair or exempt a defective hypothesis or plan; request the available bounded revision when a concrete defect remains',
            'Distinguish the response to an intervention at fixed trained parameters from the causal contribution of training interventions to historical performance. Specify what each proposed outcome can actually identify',
            'A response or absence of response does not by itself establish a competing cause. If an outcome remains compatible with both named explanations, it cannot distinguish them without additional identifying assumptions or controls',
            'An unexecuted measurement is ordinarily LIMITATION; an invalid implication from its possible outcome is DEFECT even in a conditional future plan. Quote the offending claim and correct its implication, without demanding that the future experiment already be executed',
        ]
    return context


def request_revision(state, feedback):
    if state.get("analysis_revisions", 0) >= 1:
        return False
    state["analysis_revisions"] = 1
    state["analysis_feedback"] = feedback
    state["role"] = "analyst"
    return True


def apply_content_review(state, review, decision):
    review = dict(review)
    submitted_verdict = review["verdict"]
    if state.get('content_review_protocol') in SCOPED_REVIEW_PROTOCOLS:
        defects = [item for item in review['checks'] if item['finding'] == 'DEFECT']
        if submitted_verdict == 'NEEDS_REVISION' and not defects:
            raise ValueError('Revision requires a quoted defect, not only an unmeasured limitation')
        issues = [item['claim_path'] + ': ' + item['claim_quote'] + '\n' + item['correction'] for item in defects]
        review['limitations'] = [item['statement'] for item in review['checks'] if item['finding'] == 'LIMITATION']
    else:
        issues = list(review["issues"])
        issues.extend(item["statement"] for item in review.get("checks", [])
                      if item["status"] == "NEEDS_REVISION")
    if submitted_verdict == "ACCEPT_TENTATIVE" and any(s.strip() for s in issues):
        review["verdict"] = "NEEDS_REVISION"
    review["issues"] = list(dict.fromkeys(issues))
    if review["verdict"] == "INSUFFICIENT_EVIDENCE" and decision == "PROPOSE_EXPERIMENT":
        raise ValueError("Insufficient evidence cannot approve an experiment proposal")
    state["content_review"] = {
        **review,
        "submitted_verdict": submitted_verdict,
        "reviewer": "lead_model_independent_context",
        "assurance": "MODEL_CONTENT_REVIEW_NOT_INDEPENDENT_SCIENTIFIC_PROOF",
    }
    if review["verdict"] == "NEEDS_REVISION":
        return "REVISE" if request_revision(state, "\n".join(review["issues"])) else "FACTS_ONLY"
    return "FINISH"


def record_final_decision(state, decision):
    """Keep the lead's disposition separate from the analyst's recommendation.

    Only structured decisions and host observations are projected into the stop
    summary. Free-text rationales remain in the immutable turn/action receipts;
    publishing them again would create a second, unverified observation summary.
    """
    recommendation = state["analysis"]["experiment_contract"]["decision"]
    state["final_decision"] = {
        "version": "diagnostic-final-decision-v1",
        "decision": decision,
        "analyst_recommendation": recommendation,
        "recommendation_status": (
            "PROPOSED" if decision == "PROPOSE_EXPERIMENT"
            else "DEFERRED" if decision == "NEED_MORE_EVIDENCE" else "NOT_SELECTED"
        ),
        "execution_authorized": False,
        "rationale_assurance": "UNVERIFIED_ACTION_TRACE; original turn receipt only",
    }
    observations = "\n".join(v["text"] for v in state["analysis"]["observations"].values())
    return decision + "\n宿主观测：\n" + observations


def reviewed_answer(state):
    analysis = state.get("analysis")
    observations = (
        analysis.get("observations", {})
        if analysis
        else state.get("retained_host_observations", {})
    )
    text = "科研诊断" + ("流程完成。" if state["completed"] else "部分完成。")
    text += "\n\n宿主生成的观测：\n" + "\n".join(v["text"] for v in observations.values())
    case = state.get('research_case')
    if case:
        text += '\n\n历史来源链（只读关联，不授权新实验）：\n' + json.dumps(
            {key: case.get(key) for key in ('case_id', 'historical_run_id',
             'historical_candidate_id', 'source_ids', 'approval_status')}, ensure_ascii=False)
    final = state.get("final_decision")
    if final:
        text += "\n\n负责人最终决策：" + final["decision"]
        text += "\n分析者原建议：" + final["analyst_recommendation"]
        text += "\n执行授权：无。本任务只有只读诊断权限。"
    review = state.get("content_review", {})
    if review.get('limitations'):
        text += '\n\n审核保留的限制（不是已证实机制）：\n' + '\n'.join(review['limitations'])
    if analysis and state["completed"] and review.get("verdict") == "ACCEPT_TENTATIVE":
        text += "\n\n模型复核的待验证假设（不是实验事实）：\n" + json.dumps(
            analysis["hypotheses"], ensure_ascii=False
        )
        heading = (
            "负责人采纳的实验建议（仍未授权执行）"
            if final and final["decision"] == "PROPOSE_EXPERIMENT"
            else "分析者备选方案（负责人暂缓，未授权执行）"
            if final and final["decision"] == "NEED_MORE_EVIDENCE"
            else "分析者备选方案（负责人未选择继续，未授权执行）"
            if final
            else "下一实验建议（未执行，模型复核不证明因果成立）"
        )
        text += "\n\n" + heading + "：\n" + json.dumps(
            analysis["experiment_contract"], ensure_ascii=False
        )
    elif analysis and state.get('human_review_required'):
        text += "\n\n待人工复核草稿（模型审核未通过，不是科研结论，也未授权执行）：\n"
        text += json.dumps({'hypotheses': analysis['hypotheses'],
                            'proposal': analysis['experiment_contract']}, ensure_ascii=False)
        text += "\n\n审核问题：\n" + "\n".join(review.get('issues', []))
        text += "\n\n执行权限：仅只读诊断；实验建议必须另行审阅和批准。"
    else:
        gaps = list(review.get("issues", []))
        if analysis:
            gaps += analysis["experiment_contract"]["missing_observations"]
            gaps += analysis.get("limitations", [])
        text += "\n\n缺口：分析尚未通过内容复核或证据不足；保留观测，不发布机制结论。\n"
        text += "\n".join(dict.fromkeys(gaps))
    return text
