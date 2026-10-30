"""Offline-first frozen audit experiment. Never used by the production Reader.

Labels stay outside model inputs. Atomic candidates must be exact, annotated
slices of the frozen fact, not a rewrite that silently corrects its attribution.
Every arm receives the complete original workset with identical span IDs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import time

from .answer_audit import answer_then_audit, evidence_spans, validate_draft
from .direct_answer import prepare_evidence


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def frozen_inputs(snapshot):
    saved = snapshot['run']['metadata']['reader_progress']
    spans = saved['source_spans']
    by_id = {}
    for row in snapshot['evidence']:
        eid = row['evidence_id']
        if eid in by_id and by_id[eid] != row:
            raise ValueError('Conflicting frozen evidence IDs')
        by_id[eid] = row
    rows = [by_id[eid] for eid in dict.fromkeys(s['evidence_id'] for s in spans)]
    if evidence_spans(prepare_evidence(rows)) != spans:
        raise ValueError('Saved workset does not exactly reproduce original source spans')
    return rows, deepcopy(saved['draft'])


def build_plan(question, evidence, draft, annotations):
    """Compile and validate ALL treatments before any billable request.

    annotations binds to the original inputs via SHA-256. It contains one
    entry per targeted fact, an expected verdict, and optional exact atomic slices.
    Whole-vs-individual measures fact batching; individual-vs-atomic measures
    granularity. Atomic coverage is NOT original-question completeness.
    """
    rows = prepare_evidence(evidence)
    draft = deepcopy(validate_draft(draft, rows))
    fingerprint = digest({'question': question, 'evidence': rows, 'draft': draft})
    if annotations['input_sha256'] != fingerprint:
        raise ValueError('Annotation input fingerprint mismatch')
    labels = annotations['facts']
    facts = {f['id']: f for f in draft['facts']}
    targets = annotations.get('target_fact_ids', list(facts))
    if (not facts or not isinstance(targets, list) or not targets
            or any(not isinstance(fid, str) for fid in targets)
            or len(set(targets)) != len(targets) or not set(targets) <= set(facts)
            or set(labels) != set(targets)):
        raise ValueError('Annotations must cover every predeclared target exactly once')
    valid = {'SUPPORTED', 'UNSUPPORTED', 'UNCERTAIN'}
    whole_labels = {}
    individual, atomic = [], []
    for fid, fact in facts.items():
        if fid not in targets:
            continue
        annotation = labels[fid]
        expected = annotation['expected']
        if expected not in valid:
            raise ValueError('Invalid expected verdict')
        whole_labels[fid] = expected
        single = {**draft, 'facts': [deepcopy(fact)], 'direct_answer_id': fid}
        individual.append({'arm': 'individual', 'unit': fid, 'draft': single,
                           'expected': {fid: expected}})
        atom_ids = set()
        for atom in annotation.get('atoms', []):
            aid, start, end = atom['id'], atom['start'], atom['end']
            if (not isinstance(aid, str) or not aid or aid in atom_ids
                    or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(fact['text'])
                    or fact['text'][start:end] != atom['text']
                    or atom['expected'] not in valid):
                raise ValueError('Atomic annotation must be a unique exact source-fact slice')
            atom_ids.add(aid)
            af = {**fact, 'id': aid, 'text': atom['text']}
            atomic.append({'arm': 'atomic', 'unit': fid + '/' + aid,
                           'parent_fact_id': fid, 'original_fact': deepcopy(fact),
                           'slice': {'start': start, 'end': end},
                           'draft': {**draft, 'facts': [af], 'direct_answer_id': aid},
                           'expected': {aid: atom['expected']}})
    jobs = [{'arm': 'whole', 'unit': 'all', 'draft': draft, 'expected': whole_labels}]
    jobs += individual + atomic
    control_ids = set()
    for control in annotations.get('controls', []):
        if (control.get('origin') != 'explicitly_edited_control'
                or not control.get('change_description')
                or control.get('expected') not in valid):
            raise ValueError('Controls require an explicit edit description and separate label')
        candidate = deepcopy(validate_draft(control['draft'], rows))
        if len(candidate['facts']) != 1:
            raise ValueError('Each control must have one independently labeled fact')
        fid = candidate['facts'][0]['id']
        if fid in control_ids or fid in facts:
            raise ValueError('Control IDs must be unique and separate from frozen fact IDs')
        control_ids.add(fid)
        jobs.append({'arm': 'edited_control', 'unit': fid, 'draft': candidate,
                     'expected': {fid: control['expected']}, 'origin': control['origin'],
                     'change_description': control['change_description']})
    return {'protocol': 'frozen-audit-granularity-v1', 'question': question,
            'evidence': rows, 'source_spans': evidence_spans(rows), 'draft': draft,
            'input_sha256': fingerprint, 'annotation_sha256': digest(annotations),
            'target_fact_ids': targets,
            'scoring_scope': 'predeclared_targets_only_not_all_answer_facts',
            'jobs': jobs, 'planned_max_requests': len(jobs)}


def score(results):
    """Missing/malformed responses are failures, not correct rejections."""
    metrics = {}
    for job in results:
        counts = metrics.setdefault(job['arm'], dict(
            total=0, correct=0, failed=0, unsupported_total=0, false_approvals=0,
            supported_total=0, false_rejections=0, supported_failures=0, unsupported_failures=0))
        report = job['report']
        assessed = {a['fact_id']: a['verdict'] for a in report.get('audit', {}).get('assessments', [])}
        for fid, expected in job['expected'].items():
            predicted = assessed.get(fid) if report.get('status') != 'FAILED' else None
            counts['total'] += 1
            counts['failed'] += predicted is None
            counts['correct'] += predicted == expected
            counts['unsupported_total'] += expected == 'UNSUPPORTED'
            counts['false_approvals'] += expected == 'UNSUPPORTED' and predicted == 'SUPPORTED'
            counts['supported_total'] += expected == 'SUPPORTED'
            counts['false_rejections'] += expected == 'SUPPORTED' and predicted in {'UNSUPPORTED', 'UNCERTAIN'}
            counts['supported_failures'] += expected == 'SUPPORTED' and predicted is None
            counts['unsupported_failures'] += expected == 'UNSUPPORTED' and predicted is None
    for c in metrics.values():
        c['accuracy'] = c['correct'] / c['total'] if c['total'] else None
        observed_negative = c['unsupported_total'] - c['unsupported_failures']
        observed_positive = c['supported_total'] - c['supported_failures']
        c['false_approval_rate'] = c['false_approvals'] / observed_negative if observed_negative else None
        c['false_rejection_rate'] = c['false_rejections'] / observed_positive if observed_positive else None
    return metrics


def run_plan(plan, model, *, max_requests, on_progress=None):
    if type(max_requests) is not int or max_requests < plan['planned_max_requests']:
        raise ValueError('Budget must cover the complete paired plan before execution')
    report = {'protocol': plan['protocol'], 'plan': deepcopy(plan), 'results': [],
              'status': 'RUNNING', 'provider_attempts': 0, 'calls': [],
              'evaluation_scope': 'FROZEN_AUDIT_NOT_END_TO_END',
              'semantic_correctness': 'NOT_INDEPENDENTLY_REVIEWED'}

    def save():
        report['metrics'] = score(report['results'])
        if on_progress:
            on_progress(deepcopy(report))

    class Meter:
        def complete(self, messages, **kwargs):
            original_guard = kwargs.pop('before_attempt')
            call = {'job_index': len(report['results']), 'prompt_sha256': digest(messages),
                    'status': 'STARTED', 'attempts': 0}
            report['calls'].append(call)

            def before_attempt():
                if report['provider_attempts'] >= max_requests:
                    raise RuntimeError('Experiment request budget exhausted')
                if call['attempts']:
                    raise RuntimeError('Experiment forbids provider retries')
                original_guard()
                call['attempts'] += 1
                report['provider_attempts'] += 1
                save()

            started = time.monotonic()
            try:
                response = model.complete(messages, before_attempt=before_attempt, **kwargs)
                call.update(status='RETURNED', response=asdict(response))
                return response
            except Exception as exc:
                # Do not serialize provider errors: they may contain request bodies or secrets.
                call.update(status='FAILED', error_type=type(exc).__name__, usage_unknown=True)
                raise
            finally:
                call['duration_ms'] = round((time.monotonic() - started) * 1000)
                save()

    save()
    for job in plan['jobs']:
        result = answer_then_audit(plan['question'], plan['evidence'], Meter(),
                                  frozen_draft=job['draft'], allow_repair=False,
                                  numeric_scope_mode='off')
        report['results'].append({**deepcopy(job), 'report': result})
        save()
    failed = sum(job['report']['status'] == 'FAILED' for job in report['results'])
    report['status'] = ('FAILED' if failed == len(report['results']) else
                        'PARTIAL_FAILURE' if failed else 'REVIEW_REQUIRED')
    report['known_tokens'] = sum(c.get('response', {}).get('prompt_tokens', 0)
                                 + c.get('response', {}).get('completion_tokens', 0)
                                 for c in report['calls'])
    save()
    return report
