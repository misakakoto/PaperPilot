"""Re-score historical component receipts in a new folder without provider calls."""
import json
from pathlib import Path
import shutil


def main():
    from evals.qasper.audit_candidate_budget import audit as audit_candidates
    from evals.qasper.audit_llm_rerank import audit as audit_selection
    root = Path(__file__).resolve().parents[2]
    folder = root/'outputs/interview-retrieval-recheck-20261001'
    folder.mkdir(parents=True, exist_ok=False)
    candidates, selected = folder/'candidates', folder/'selection'
    candidates.mkdir()
    selected.mkdir()
    for name in ('manifest.json', 'rankings.jsonl'):
        shutil.copyfile(root/'.paperpilot/candidate-budget-confirm-v2'/name, candidates/name)
    source = root/'.paperpilot/evidence-select-confirm-v2'
    for path in source.iterdir():
        if path.is_file() and path.name != 'independent-audit.json':
            shutil.copyfile(path, selected/path.name)
    dataset = root/'.paperpilot/datasets/qasper/v0.3/qasper-test-v0.3.json'
    a = audit_candidates(dataset, candidates)
    b = audit_selection(dataset, candidates, selected)
    print(json.dumps({'candidate_audit': a, 'selection_audit': b,
                      'new_provider_requests': 0}, indent=2))


if __name__ == '__main__':
    main()
