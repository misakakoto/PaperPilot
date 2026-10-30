"""Bounded exact source navigation; never imports or executes candidate code."""

import ast
import hashlib
import re

from .prototype_workspace import digest

POLICY = "source-code-navigation-v1"


def enabled(state):
    policy = state.get("code_navigation_policy")
    if policy not in (None, POLICY):
        raise ValueError("Unknown code navigation policy")
    return policy == POLICY


def _source(artifact):
    if artifact["kind"] == "implement":
        return artifact["content"]["source"]
    if artifact["kind"] == "read" and artifact["content"].get("evidence_type") == "CODE":
        return artifact["content"]["content"]
    raise ValueError("Code navigation requires an owned implementation or CODE read artifact")


def inspect_code(work, arguments):
    allowed = {"artifact_id", "symbol", "line", "context_lines", "max_characters"}
    if (
        not isinstance(arguments, dict)
        or "artifact_id" not in arguments
        or set(arguments) - allowed
        or ("symbol" in arguments) == ("line" in arguments)
    ):
        raise ValueError(
            "inspect_code requires artifact_id and exactly one qualified symbol or source line"
        )
    identity = arguments["artifact_id"]
    if not isinstance(identity, str) or identity not in work["artifacts"]:
        raise ValueError("Unknown code artifact ID")
    artifact = work["artifacts"][identity]
    if digest(artifact) != identity:
        raise ValueError("Code artifact identity changed")
    source = _source(artifact)
    maximum, context = arguments.get("max_characters", 6000), arguments.get("context_lines", 8)
    if (
        type(maximum) is not int
        or not 128 <= maximum <= 12000
        or type(context) is not int
        or not 0 <= context <= 30
    ):
        raise ValueError("Use max_characters 128..12000 and context_lines 0..30")
    lines = source.splitlines(keepends=True)
    symbols = []

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                start = min([child.lineno] + [n.lineno for n in child.decorator_list])
                symbols.append({"symbol": name, "start_line": start, "end_line": child.end_lineno})
                visit(child, name + ".")
            else:
                visit(child, prefix)

    try:
        visit(ast.parse(source))
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("Cannot navigate invalid source syntax") from exc
    if "symbol" in arguments:
        symbol = arguments["symbol"]
        if not isinstance(symbol, str):
            raise ValueError("Use a qualified symbol name")
        matches = [s for s in symbols if s["symbol"] == symbol]
        if len(matches) != 1:
            raise ValueError(
                "Symbol missing or ambiguous; use the exact qualified class.method name"
            )
        first, last = matches[0]["start_line"], matches[0]["end_line"]
    else:
        line = arguments["line"]
        if type(line) is not int or not 1 <= line <= len(lines):
            raise ValueError("Source line is outside the exact artifact version")
        first, last = max(1, line - context), min(len(lines), line + context)
    kept, size = [], 0
    for line in lines[first - 1 : last]:
        if size + len(line) > maximum:
            break
        kept.append(line)
        size += len(line)
    if not kept:
        raise ValueError("Source line exceeds the character bound; increase max_characters")
    end = first + len(kept) - 1
    fragment = "".join(kept)
    containing = sorted(
        (s for s in symbols if s["start_line"] <= first <= s["end_line"]),
        key=lambda s: s["end_line"] - s["start_line"],
    )
    return {
        "status": "OK",
        "action": "inspect_code",
        "artifact_id": identity,
        "workspace_revision": work["revision"],
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "start_line": first,
        "end_line": end,
        "source": fragment,
        "selected_symbol": arguments.get("symbol")
        or (containing[0]["symbol"] if containing else None),
        "truncated": end < last,
        "next_line": end + 1 if end < last else None,
        "unique_in_source": source.count(fragment) == 1,
        "assurance": "EXACT_ORIGINAL_SOURCE_ONLY_NOT_EXECUTION_OR_DESIGN_VERIFICATION",
    }


def repair_locations(state, base):
    if not enabled(state):
        return []
    feedback = state.get("validation_feedback") or {}
    result = feedback.get("result") or {}
    if (
        feedback.get("code_id") != base
        or result.get("validated_code_id") != base
        or result.get("validation_contract") != state.get("validation_contract")
    ):
        return []
    receipts = [
        i
        for i, a in state["work"]["artifacts"].items()
        if a["kind"] == "validate"
        and base in a["inputs"]
        and a["content"] == result
        and digest(a) == i
    ]
    if not receipts:
        return []
    trace = result.get("execution", {}).get("metrics", {}).get("traceback", "")
    locations = []
    for line in dict.fromkeys(re.findall(r'File "[^"\n]*(?:/|\\)model\.py", line (\d+)', trace)):
        try:
            view = inspect_code(
                state["work"],
                {
                    "artifact_id": base,
                    "line": int(line),
                    "context_lines": 4,
                    "max_characters": 1500,
                },
            )
        except ValueError:
            continue
        locations.append(
            {"receipt_artifact_id": sorted(receipts)[0], "failure_line": int(line), **view}
        )
        if len(locations) == 3:
            break
    return locations


INSTRUCTIONS = (
    " inspect_code arguments={artifact_id,symbol:qualified_class.method} OR "
    "{artifact_id,line:positive_source_line,context_lines?:0..30}, max_characters?:128..12000. "
    "It returns exact original source, line numbers and a source hash without executing code. "
    "Use this for omitted functions instead of paging through JSON. repair_code_locations are "
    "exact snippets bound to the failed candidate and receipt, not a proposed fix. "
    "Existing forward excerpts can already supply a unique literal patch; do not re-read them unnecessarily. "
    "Patches still target candidate_patch.base_artifact_id and exact original text; stale versions cannot be reused."
)
