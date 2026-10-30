import json

from scripts.paired_synthesis import prepare_pair


def test_both_arms_receive_exact_production_excerpts():
    snapshot = {
        "run": {"run_id": "fixture", "question": "实验结果是什么？"},
        "claim_graph": {"claims": [{"claim_id": "claim_1", "statement": "Result",
                                    "status": "SUPPORTED"}]},
        "evidence": [{"evidence_id": "evidence_1", "claim_id": "claim_1",
                      "polarity": "SUPPORTS", "content": "F1 80.2. " * 800}],
    }
    pair = prepare_pair(snapshot)
    direct = json.loads(pair["arms"]["direct"]["messages"][-1]["content"])
    structured = json.loads(pair["arms"]["structured"]["messages"][-1]["content"])
    assert direct["evidence"] == structured["evidence"]
    assert len(direct["evidence"][0]["content"]) == 4000
    assert pair["provider_attempts"] == 0
    assert pair["semantic_review"] == "NOT_RUN"
