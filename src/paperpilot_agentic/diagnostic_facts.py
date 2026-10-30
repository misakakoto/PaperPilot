"""Host-derived trend statements; no model-selected metric values or directions."""

from decimal import Decimal
import re

VERSION = "diagnostic-observation-facts-v2"
NARRATIVE_CONTRACT = "host-observation-prose-v3"
METRICS = ("precision", "recall", "f1", "iou")

# This enforces a representation contract, not semantic truth. Both correct and
# incorrect trend restatements belong in the host-projected observation section.
_METRIC_WORD = re.compile(
    # Unicode \b treats Han characters as word characters too. ASCII identifier
    # edges recognise Chinese-adjacent metrics without matching F16 or myF1Block.
    r"(?<![A-Za-z0-9_])(?:f[\s_-]?1|m?iou|precision|recall|accuracy|p\s*/\s*r)(?![A-Za-z0-9_])"
    r"|召回率?|精确率|精准率|准确率|精度|性能",
    re.IGNORECASE,
)
_TREND_WORD = re.compile(
    r"上升|下降|提高|降低|增加|减少|提升|改善|恶化|退化|同升|同降|权衡|方向相反"
    r"|\b(?:up|down|rose|fell|ris[ei]\w*|fall\w*|increas\w*|decreas\w*"
    r"|improv\w*|declin\w*|drop\w*|gain\w*|trade[ -]?offs?)\b",
    re.IGNORECASE,
)


def validate_narrative_contract(action, reason, arguments):
    """Keep recognisable metric trends out of public mechanism/decision prose.

    Expected future patterns and falsification are explicitly predictive fields;
    feedback may quote an error to request correction. None is certified true by
    this lexical check. Synonyms and causal validity still need content review.
    """
    checked = []
    exempt = {
        "fact_refs",
        "support_refs",
        "counter_refs",
        "observation_ids",
        "expected_patterns",
        "falsification",
        "feedback",
    }

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if key not in exempt:
                    visit(child, path + "." + key)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                visit(child, f"{path}[{i}]")
        elif isinstance(value, str):
            checked.append(path)
            for clause in re.split(r"[。！？.!?;；\n]", value):
                if _METRIC_WORD.search(clause) and _TREND_WORD.search(clause):
                    raise ValueError(
                        "OBSERVATION_PROSE_REQUIRES_ID at "
                        + path
                        + ": keep metric trends in selected host observation IDs; "
                        "mechanism/decision prose must not restate them. This is a "
                        "representation rejection, not a factual verdict. Put future "
                        "test predictions in expected_patterns/falsification."
                    )

    # Navigation/delegation reasons describe what the agent wants to check, not
    # observations to publish. Do not pretend a lexical rule can distinguish a
    # question from an assertion. Retain these as explicitly unverified trace.
    unverified = []
    if action in {"inspect_experiment", "delegate_analysis"}:
        unverified.append(action + ".reason")
    else:
        visit(reason, action + ".reason")
    visit(arguments, action + ".arguments")
    return {
        "version": NARRATIVE_CONTRACT,
        "checked_paths": checked,
        "unverified_trace_paths": unverified,
        "assurance": "LEXICAL_REPRESENTATION_ONLY; not semantic or causal verification",
    }


def direction(value):
    if value is None:
        return "UNDEFINED"
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("Non-finite diagnostic delta")
    return "UP" if number > 0 else "DOWN" if number < 0 else "UNCHANGED"


def observation_facts(observations):
    facts = {}
    for view in ("paired", "coverage"):
        for i, row in enumerate(observations.get(view, {}).get("rows", [])):
            group = row.get("group")
            scope = f"种子{row['seed']}" + (f"，标签正像素占比{group}" if group else "，整体")
            for metric in METRICS:
                ref = f"{view}/{i}/delta/{metric}"
                value = row["delta"][metric]
                status = direction(value)
                display = (
                    "无定义，不能记为零"
                    if status == "UNDEFINED"
                    else {"UP": "上升", "DOWN": "下降", "UNCHANGED": "不变"}[status]
                    + f"，差值{Decimal(str(value)) * 100:+.6g}个百分点"
                )
                facts[f"pattern/{view}/{i}/{metric}"] = {
                    "seed": row["seed"],
                    "group": group,
                    "metric": metric,
                    "direction": status,
                    "source_refs": [ref],
                    "unit": "fraction; rendered delta in percentage points",
                    "text": f"{scope}，{metric}：{display}。",
                }
            p, r = direction(row["delta"]["precision"]), direction(row["delta"]["recall"])
            relation = (
                "UNDEFINED"
                if "UNDEFINED" in (p, r)
                else "BOTH_UP"
                if p == r == "UP"
                else "BOTH_DOWN"
                if p == r == "DOWN"
                else "OPPOSITE"
                if {p, r} == {"UP", "DOWN"}
                else "HAS_UNCHANGED"
            )
            text = {
                "UNDEFINED": "至少一项无定义，不能判定权衡",
                "BOTH_UP": "precision与recall均上升",
                "BOTH_DOWN": "precision与recall均下降",
                "OPPOSITE": "precision与recall方向相反",
                "HAS_UNCHANGED": "precision或recall不变",
            }[relation]
            facts[f"pattern/{view}/{i}/precision_recall"] = {
                "seed": row["seed"],
                "group": group,
                "metric": "precision_recall",
                "relation": relation,
                "source_refs": [f"{view}/{i}/delta/precision", f"{view}/{i}/delta/recall"],
                "text": f"{scope}，{text}。",
            }
    for i, group in enumerate(observations.get("samples", {}).get("rows", [])):
        for j, row in enumerate(group["rows"]):
            facts[f"pattern/samples/{i}/{j}"] = {
                "seed": group["seed"],
                "group": row.get("group"),
                "sample_id": row["sample_id"],
                "metric": group["error"],
                "source_refs": [f"samples/{i}/rows/{j}/extra_errors"],
                "text": f"种子{group['seed']}，样本{row['sample_id']}，{group['error']}净增加{row['extra_errors']}像素；Top5条目不证明总体误差集中。",
            }
    return facts


def select_observations(ids, catalog, *, label="observation_ids"):
    if (
        not isinstance(ids, list)
        or not ids
        or len(ids) > 256
        or any(not isinstance(k, str) or k not in catalog for k in ids)
    ):
        raise ValueError(label + " must select existing host observation IDs")
    return {key: catalog[key] for key in dict.fromkeys(ids)}


def validate_experiment(plan, *, contrastive=False):
    required = {
        "hypothesis",
        "support_refs",
        "counter_refs",
        "missing_observations",
        "falsification",
        "intervention",
        "controls",
        "measurements",
        "expected_patterns",
        "alternatives_remaining",
        "decision",
    }
    if not isinstance(plan, dict) or set(plan) not in (required, required | {"comparison_design"}):
        raise ValueError("Experiment v2 requires exactly: " + ", ".join(sorted(required)))
    if plan.get('comparison_design') is not None:
        from .research_contracts import ComparisonDesign, ContrastiveComparisonDesign
        design_model = ContrastiveComparisonDesign if contrastive else ComparisonDesign
        design_model.model_validate(plan['comparison_design'])
    if plan["decision"] not in {"PROPOSE_EXPERIMENT", "STOP_INSUFFICIENT_EVIDENCE"}:
        raise ValueError("Unknown experiment decision")
    if not isinstance(plan["intervention"], str) or not plan["intervention"].strip():
        raise ValueError("State the intervention or why further execution should stop")
    for key in ("controls", "measurements", "expected_patterns", "alternatives_remaining"):
        if (
            not isinstance(plan[key], list)
            or not plan[key]
            or any(not isinstance(v, str) or not v.strip() for v in plan[key])
        ):
            raise ValueError("Experiment " + key + " must be an explicit nonempty list")
    # Structure is checkable here. Causal identifiability still needs content review.
    return {
        k: plan[k]
        for k in (
            "hypothesis",
            "support_refs",
            "counter_refs",
            "missing_observations",
            "falsification",
        )
    }
