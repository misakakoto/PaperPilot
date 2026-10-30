"""Quarantine local draft defects without rewriting or approving any claim.

Ambiguous/global contracts still use the existing bounded format repair. The
remaining facts cross exactly the same strict validator and independent audit.
"""
from copy import deepcopy

from .answer_contracts import (validate_draft, validate_fact_shape,
    validate_fact_dependencies, unrequested_structural_formula,
    normalize_inference_citations, normalize_inference_dependencies)
from .table_claims import row_catalog, validate_row_refs

VERSION = 'answer-fact-isolation-v2-aggregated-and-post-audit-targets'


PATCH_VERSION = 'host-keyed-replacements-v1'


def bind_replacements(value, target_ids):
    """Bind explicit keys to host-owned IDs; never infer an ID from text.

    Old facts envelopes remain strict-compatible for retained responses. New
    replacements cannot alter identity fields or introduce additional targets.
    Scientific content still crosses projection, dependency checks and audit.
    """
    if not isinstance(value, dict) or 'replacements' not in value:
        return value, []
    if set(value) != {'replacements', 'limitations'}:
        raise ValueError('Replacement response requires replacements and limitations only')
    replacements = value['replacements']
    if not isinstance(replacements, dict) or set(replacements) != set(target_ids):
        raise ValueError('Replacement keys must match exactly the host target IDs')
    facts, bindings = [], []
    for fid in target_ids:
        replacement = replacements[fid]
        if not isinstance(replacement, dict) or 'id' in replacement:
            raise ValueError('Replacement identity belongs to the host, not its content')
        facts.append({'id': fid, **deepcopy(replacement)})
        bindings.append({'fact_id': fid, 'selection_key': fid, 'policy': PATCH_VERSION})
    return {'facts': facts, 'limitations': deepcopy(value['limitations'])}, bindings


def isolate_repair_targets(value, target_ids):
    """Quarantine additions; retained exact-ID targets still need full audit."""
    if not isinstance(value, dict) or not isinstance(value.get('facts'), list):
        raise ValueError('Expected a local repair fact patch')
    facts = value['facts']
    ids = [fact.get('id') if isinstance(fact, dict) else None for fact in facts]
    if (any(not isinstance(fid, str) or not fid for fid in ids)
            or len(set(ids)) != len(ids) or not set(target_ids) <= set(ids)):
        raise ValueError('Local repair target IDs are missing or ambiguous')
    result = deepcopy(value)
    result['facts'] = [deepcopy(fact) for fact in facts if fact['id'] in target_ids]
    return result, [deepcopy(fact) for fact in facts if fact['id'] not in target_ids]


def _citation_ids(fact):
    value = fact.get('evidence_ids')
    return [eid for eid in value if isinstance(eid, str)] if isinstance(value, list) else []


def repair_workset(original, isolation, evidence):
    """Whole, source-owned passages for the failed claims and source premises.

    No summary/truncation and no new evidence. All original candidates remain
    available to the completeness audit; this view is only for local repair.
    """
    import hashlib
    target_ids = set(isolation['quarantined_fact_ids'])
    targets = [f for f in original['facts'] if f['id'] in target_ids]
    by_id = {f['id']: f for f in original['facts']}
    cited = {eid for f in targets for eid in _citation_ids(f)}
    needed = set()
    pending = list(target_ids)
    while pending:
        fid = pending.pop()
        if fid in needed or fid not in by_id:
            continue
        needed.add(fid)
        deps = by_id[fid].get('depends_on', [])
        if isinstance(deps, list):
            pending.extend(d for d in deps if isinstance(d, str))
    premises = [f for f in original['facts'] if f['id'] not in target_ids
                and f.get('kind', 'SOURCE_FACT') == 'SOURCE_FACT'
                and (f['id'] in needed or cited.intersection(f['evidence_ids']))]
    wanted = {eid for f in targets + premises for eid in _citation_ids(f)}
    selected = [deepcopy(e) for e in evidence if e['evidence_id'] in wanted]
    navigation = [{'evidence_id': e['evidence_id'], 'source_uri': e.get('source_uri'),
                   'page': e.get('page'), 'content_sha256': hashlib.sha256(
                       e['content'].encode('utf-8')).hexdigest(),
                   'status': 'READ' if e['evidence_id'] in wanted else 'NOT_READ_IN_REPAIR'}
                  for e in evidence]
    return targets, premises, selected, navigation



def validate_patch_operations(original, patch, isolation):
    """Do not turn a rejected numerical inference into a repeated value row.

    This checks a concrete loss of work, not semantic correctness. A numbers-free
    inference or a host-computed comparison still needs the independent audit.
    Non-table and historical shape repairs retain their existing contracts.
    """
    if not isinstance(patch, dict) or not isinstance(patch.get('facts'), list):
        return  # merge_repair owns the strict envelope/ID checks.
    originals = {f['id']: f for f in original['facts']}
    numerical_errors = {row['fact_id'] for row in isolation['errors']
                        if 'Catalogued table numbers require TABLE_FACT projection' in row['error']}
    for fact in patch['facts']:
        if not isinstance(fact, dict):
            continue
        prior = originals.get(fact.get('id'), {})
        if (fact.get('id') in numerical_errors and prior.get('kind') == 'INFERENCE'
                and fact.get('kind') == 'TABLE_FACT' and fact.get('operation') == 'value'):
            raise ValueError('Local repair lost inference work: fact_id=' + fact['id'] +
                             '; use a numbers-free inference or a source-bound comparison, '
                             'not a repeated value row')

def merge_repair(original, patch, isolation, evidence, *, question, request_history, table_rows,
                 premise_ids=None, repair_sources=None, purpose_projection=(), table_projection=()):
    """A patch may replace only quarantined IDs; immutable eligible facts stay."""
    if not isinstance(patch, dict) or set(patch) != {'facts', 'limitations'}:
        raise ValueError('Local repair requires facts and limitations only')
    facts = patch['facts']
    limitations = patch['limitations']
    if (not isinstance(facts, list) or any(not isinstance(f, dict) for f in facts)
            or not isinstance(limitations, list) or any(not isinstance(x, str) for x in limitations)):
        raise ValueError('Invalid local repair shape')
    ids = [f.get('id') for f in facts]
    targets = set(isolation['quarantined_fact_ids'])
    generated_ids = set()
    original_ids = {f['id'] for f in original['facts']}
    by_patch_id = {f.get('id'): f for f in facts if isinstance(f, dict)}
    for entry in table_projection:
        selection = entry.get('raw_selection') or {}
        parent_id = selection.get('id')
        if parent_id not in targets:
            raise ValueError('Table projection parent is not an owned repair target')
        if selection.get('operation') != 'difference':
            continue
        # Recompute from registered rows. Neither a trace label nor a familiar
        # endpoint suffix is permission to insert model-created facts.
        from .table_projection import project_table_facts
        chinese = any('\u4e00' <= c <= '\u9fff' for c in entry['projected_fact']['text'])
        canonical, _ = project_table_facts({'facts': [selection]}, table_rows, chinese=chinese)
        canonical_by_id = {f['id']: f for f in canonical['facts']}
        if (canonical_by_id.get(parent_id) != entry.get('projected_fact') or
                by_patch_id.get(parent_id) != canonical_by_id[parent_id]):
            raise ValueError('Table calculation trace differs from registered source projection')
        for child_id, child in canonical_by_id.items():
            if child_id == parent_id:
                continue
            if (child_id in original_ids or child_id in generated_ids or by_patch_id.get(child_id) != child):
                raise ValueError('Table endpoint is changed, colliding or not host generated')
            generated_ids.add(child_id)
    for entry in purpose_projection:
        parent_id, child_id = entry['parent_fact_id'], entry['purpose_fact_id']
        if (parent_id not in targets or child_id != parent_id + '__purpose'
                or child_id in original_ids or child_id in generated_ids):
            raise ValueError('Invalid host-projected purpose identity')
        child = next((f for f in facts if f.get('id') == child_id), None)
        parts = entry['original_parts']['purpose']
        if child != {'id': child_id, 'kind': 'SOURCE_FACT', **parts, 'depends_on': []}:
            raise ValueError('Purpose content differs from its explicit original parts')
        generated_ids.add(child_id)
    if (any(not isinstance(fid, str) for fid in ids) or len(set(ids)) != len(ids)
            or set(ids) != targets | generated_ids):
        raise ValueError('Local repair must replace exactly the quarantined fact IDs')
    # Reuse the existing lossless provenance recovery on a view containing only
    # supplied SOURCE_FACT premises and patch targets. Never normalize or edit
    # the already eligible facts, nor borrow a premise the model did not see.
    premises = [f for f in original['facts'] if f['id'] not in ids
                and f.get('kind', 'SOURCE_FACT') == 'SOURCE_FACT'
                and (premise_ids is None or f['id'] in premise_ids)]
    scope = evidence if repair_sources is None else repair_sources
    view, _ = normalize_inference_citations({'facts': premises + facts}, scope)
    view, _ = normalize_inference_dependencies(view)
    facts = [f for f in view['facts'] if f['id'] in ids]
    for fact in facts:
        validate_fact_shape(fact, {e['evidence_id'] for e in scope})
    replacements = {f['id']: f for f in facts}
    if premise_ids is not None:
        allowed_dependencies = set(premise_ids) | set(ids) | generated_ids
        for fact in facts:
            deps = fact.get('depends_on', [])
            if (not isinstance(deps, list) or any(not isinstance(d, str) or
                    d not in allowed_dependencies for d in deps)):
                raise ValueError('Local repair dependency outside supplied premises')
    merged = deepcopy(original)
    merged['facts'] = [deepcopy(replacements.get(f['id'], f)) for f in original['facts']]
    merged['facts'].extend(deepcopy(f) for f in facts if f['id'] in generated_ids)
    merged['limitations'] = list(dict.fromkeys(original.get('limitations', []) + limitations))
    return validate_draft(merged, evidence, question=question, request_history=request_history,
                          table_rows=table_rows)


def isolate_draft(value, evidence, *, question=None, request_history=None, table_rows=None, rejected_facts=None):
    """Return a strict-valid subset and diagnostics, or None for global errors."""
    if not isinstance(value, dict) or not isinstance(value.get('facts'), list):
        return None
    facts = value['facts']
    limitations = value.get('limitations', [])
    if (not 1 <= len(facts) <= 12 or not isinstance(limitations, list)
            or any(not isinstance(x, str) for x in limitations)):
        return None
    # Never resolve duplicate identities, a missing nomination or a malformed
    # envelope by guessing which claim the model intended.
    ids = [f.get('id') if isinstance(f, dict) else None for f in facts]
    if (any(not isinstance(fid, str) or not fid for fid in ids)
            or len(set(ids)) != len(ids) or value.get('direct_answer_id') not in ids):
        return None
    by_id = dict(zip(ids, facts))
    allowed = {e['evidence_id'] for e in evidence}
    catalog = row_catalog(evidence) if table_rows is None else table_rows
    rejected = dict(rejected_facts or {})
    if set(rejected) - set(ids):
        raise ValueError('Quarantine identity outside draft')
    for fact in facts:
        try:
            validate_fact_shape(fact, allowed)
            validate_fact_dependencies(fact, by_id)
            validate_row_refs({'facts': [fact]}, catalog)
            if question and unrequested_structural_formula(question, [fact]):
                raise ValueError('Unrequested structural formula')
        except (ValueError, TypeError) as exc:
            # One error must not hide another in the sole content-patch turn.
            previous = rejected.get(fact['id'])
            rejected[fact['id']] = (previous + '; ' + str(exc)
                                   if previous and str(exc) not in previous else previous or str(exc))
    # Dependencies on a quarantined SOURCE_FACT cannot survive, even when
    # their own shape and original dependency IDs were otherwise legal.
    while True:
        added = {f['id']: 'Dependency on quarantined fact' for f in facts
                 if f['id'] not in rejected and any(d in rejected for d in f.get('depends_on', []))}
        if not added:
            break
        rejected.update(added)
    retained = [deepcopy(f) for f in facts if f['id'] not in rejected]
    if not rejected or not retained:
        return None
    subset = deepcopy(value)
    subset['facts'] = retained
    reassigned = subset['direct_answer_id'] in rejected
    if reassigned:
        # This remains only a writer proposal. The audit must assign relevance.
        subset['direct_answer_id'] = retained[0]['id']
    subset['limitations'] = list(limitations)
    validate_draft(subset, evidence, question=question, request_history=request_history,
                   table_rows=catalog)
    return subset, {'version': VERSION, 'quarantined_fact_ids': sorted(rejected),
        'eligible_fact_ids': [f['id'] for f in retained],
        'errors': [{'fact_id': fid, 'error': rejected[fid]} for fid in ids if fid in rejected],
        'original_direct_answer_id': value['direct_answer_id'],
        'direct_answer_reassigned': reassigned, 'factual_review': 'NOT_RUN',
        'repair_status': 'PENDING'}


def extend_isolation_after_audit(original, isolation, audit):
    """Include rejected source premises in the existing content-patch slot.

    Already supported facts stay immutable. A previously shape-valid but rejected
    mechanism must not remain a read-only premise for repairing its dependents.
    Independent unsupported inferences are not read-only source premises; keep
    their existing verdict without making them mandatory patch replacements.
    This changes repair eligibility, never any support verdict or source allowlist.
    """
    result = deepcopy(isolation)
    facts = {f['id']: f for f in original['facts']}
    targets = set(result['quarantined_fact_ids'])
    added = []
    for row in audit.get('assessments', []):
        fid = row.get('fact_id')
        if (fid not in facts or fid in targets or row.get('verdict') == 'SUPPORTED'
                or row.get('relevance') == 'TANGENTIAL'
                or facts[fid].get('kind', 'SOURCE_FACT') != 'SOURCE_FACT'):
            continue
        targets.add(fid)
        added.append(fid)
        detail = '; '.join(str(row[k]) for k in (
            'row_binding_reason', 'claim_binding_status', 'reason') if row.get(k))
        result['errors'].append({'fact_id': fid,
            'error': 'Post-audit rejection: ' + str(row.get('verdict')) + '; ' + detail})
    while True:
        dependents = [f['id'] for f in facts.values() if f['id'] not in targets
                      and any(d in targets for d in f.get('depends_on', []))]
        if not dependents:
            break
        targets.update(dependents)
        result['errors'].extend({'fact_id': fid, 'error': 'Dependency on post-audit rejected fact'}
                                for fid in dependents)
    result['audit_rejected_fact_ids'] = sorted(added)
    result['quarantined_fact_ids'] = sorted(targets)
    result['eligible_fact_ids'] = [fid for fid in facts if fid not in targets]
    return result


def apply_isolation_coverage(audit, isolation):
    """A subset cannot erase the original draft's unresolved obligations."""
    if not isolation or not isolation['quarantined_fact_ids']:
        return audit
    result = deepcopy(audit)
    result['coverage'] = 'PARTIAL'
    gap = '部分候选结论尚未完成审核；以下保留已审核事实，未审核的解释不作为答案。'
    result['missing_aspects'] = list(dict.fromkeys(result.get('missing_aspects', []) + [gap]))
    return result
