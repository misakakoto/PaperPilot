"""Freeze an existing Reader's source window; compare a generic quantity policy."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from paperpilot_agentic.config import RuntimeSettings  # noqa: E402
from paperpilot_agentic.db import RuntimeDatabase  # noqa: E402
from paperpilot_agentic.durable_reader import DurableResearchModel  # noqa: E402
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime  # noqa: E402
from paperpilot_agentic.providers import OpenAICompatibleLLM  # noqa: E402
from paperpilot_agentic.models import Budget, RunStatus  # noqa: E402
from paperpilot_agentic.answer_audit import answer_then_audit, QUANTITY_POLICY  # noqa: E402
from paperpilot_agentic.direct_answer import answer_from_evidence  # noqa: E402


class PolicyModel:
    def __init__(self, model):
        self.model = model
    def complete(self, messages, **kwargs):
        messages = [dict(m) for m in messages]
        messages[0]['content'] += '\n' + QUANTITY_POLICY
        return self.model.complete(messages, **kwargs)
    def for_repair(self):
        return PolicyModel(self.model.for_repair())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['baseline', 'quantity', 'revision', 'direct', 'direct_concise',
                                             'audit_full', 'audit_cited'], required=True)
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier runs')
    source = json.loads(args.source.read_text(encoding='utf-8'))
    selected = set(source['persisted_run']['metadata']['reader_workset']['selected_evidence_ids'])
    rows = [row for row in source['evidence'] if row['evidence_id'] in selected]
    if not rows or len(rows) != len(selected):
        raise ValueError('Cannot recover exact source window')
    if not args.execute_paid:
        print('No provider calls; pass --execute-paid to run frozen-context diagnostic')
        return
    settings = RuntimeSettings.from_env(require_api_key=True)
    args.output.mkdir(parents=True)
    db = RuntimeDatabase(args.output/'runtime.sqlite3')
    rid = db.create_run(source['question'], Budget())
    db.update_run_status(rid, RunStatus.RUNNING)
    model = DurableResearchModel(PersistentLLMRuntime(db, OpenAICompatibleLLM(settings, max_retries=0)),
                                 rid, stage='answer_first', reserve_class='synthesis')
    if args.variant == 'quantity':
        model = PolicyModel(model)
    feedback = ({k: source['reader'].get(k) for k in ('draft', 'audit', 'limitations')}
                if args.variant == 'revision' else None)
    try:
        report = (answer_from_evidence(source['question'], rows, model, concise_policy=args.variant == 'direct_concise')
                  if args.variant in {'direct', 'direct_concise'} else
                  answer_then_audit(source['question'], rows, model, quantity_policy=False, revision_feedback=feedback,
                      frozen_draft=source['reader']['draft'] if args.variant in {'audit_full', 'audit_cited'} else None,
                      audit_cited_only=args.variant == 'audit_cited'))
    except Exception as exc:
        db.update_run_status(rid, RunStatus.FAILED, error=type(exc).__name__)
        raise
    db.update_run_status(rid, RunStatus.FAILED if report['status'] == 'FAILED' else RunStatus.COMPLETED)
    result = {'scope': 'frozen-source diagnostic; not retrieval or full Agent evaluation',
        'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'variant': args.variant, 'model': settings.model, 'question': source['question'],
        'policy': QUANTITY_POLICY if args.variant == 'quantity' else '',
        'report': report, 'usage': db.get_run(rid)}
    (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'calls': result['usage']['used_llm_calls'],
                      'tokens': result['usage']['used_tokens']}))


if __name__ == '__main__':
    main()
