"""Read-only, zero-call representation checks of saved provider responses.

This does not rescore historical answers or simulate a corrected model response.
Only response files are read; request files/credentials are never loaded.
"""

import argparse
import hashlib
import json
from pathlib import Path

from paperpilot_agentic.diagnostic_facts import (
    NARRATIVE_CONTRACT,
    validate_narrative_contract,
)


def replay(receipts: Path):
    rows = []
    for path in sorted(receipts.glob("response-*.json")):
        raw = path.read_bytes()
        row = {"response": path.name, "source_sha256": hashlib.sha256(raw).hexdigest()}
        try:
            response = json.loads(raw)
            text = response["choices"][0]["message"]["content"]
            value = json.loads(text)
        except (ValueError, KeyError, IndexError, TypeError):
            row["status"] = "NOT_CHECKED_INVALID_JSON_OR_ENVELOPE"
        else:
            if not isinstance(value, dict) or not isinstance(value.get("arguments"), dict):
                row["status"] = "NOT_CHECKED_NON_ACTION"
            elif not isinstance(value.get("action"), str) or value["action"] not in {
                "report",
                "finish",
                "inspect_experiment",
                "delegate_analysis",
            }:
                row["status"] = "NOT_CHECKED_UNKNOWN_ACTION"
            elif not isinstance(value.get("reason"), str) or not value["reason"].strip():
                row["status"] = "NOT_CHECKED_EXISTING_MISSING_REASON_ERROR"
            else:
                row["action"] = value["action"]
                try:
                    row["check"] = validate_narrative_contract(
                        value["action"], value["reason"], value["arguments"]
                    )
                except ValueError as exc:
                    row["status"] = "REJECT_REPRESENTATION"
                    row["error"] = str(exc)
                else:
                    row["status"] = "ACCEPT_REPRESENTATION_ONLY"
        rows.append(row)
    return {
        "contract": NARRATIVE_CONTRACT,
        "responses": rows,
        "new_provider_requests": 0,
        "new_gpu_starts": 0,
        "historical_scores_changed": False,
        "assurance": "COUNTERFACTUAL_LEXICAL_CHECK; accepted prose is not scientifically verified",
        "counts": {
            s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.receipts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # New output only: never overwrite a frozen replay or historical response.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
