"""Engine-aware structural observations, separate from independent source review."""
import re


def structural_observations(case, result, parent=None):
    final = result.get('final_answer') or {}
    answer = str(final.get('answer') or '')
    citations = set(re.findall(r'\[(evidence_[A-Za-z0-9_-]+)\]', answer))
    evidence = {e['evidence_id']: e for e in result.get('evidence', [])}
    checks = case.get('answer_checks', {})
    issues = []
    if result['run']['status'] != 'COMPLETED':
        issues.append('process_not_completed')
    if not answer:
        issues.append('empty_answer')
    if citations - evidence.keys():
        issues.append('unowned_citation')
    if len(citations) < checks.get('minimum_citations', 1):
        issues.append('too_few_citations')
    sources = set()
    for eid in citations & evidence.keys():
        e = evidence[eid]
        provenance = e.get('provenance') or {}
        sha = provenance.get('paper_sha256') or provenance.get('document_sha256')
        if not sha:
            match = re.search(r'paper://([a-f0-9]{64})', e.get('source_uri') or '')
            sha = match.group(1) if match else None
        if sha:
            sources.add(sha)
        else:
            issues.append('citation_source_identity_missing')
    if sources - set(case['paper_ids']):
        issues.append('citation_outside_allowed_papers')
    if set(checks.get('required_citation_source_sha256s', [])) - sources:
        issues.append('required_paper_not_cited')
    for term in checks.get('required_substrings', []):
        if term not in answer:
            issues.append('missing_term:' + term)
    for group in checks.get('required_any_substring_groups', []):
        if not any(term in answer for term in group):
            issues.append('missing_alternative:' + '|'.join(group))
    groups = checks.get('required_one_of_substring_groups', [])
    if groups and not any(all(term in answer for term in group) for group in groups):
        issues.append('missing_complete_numeric_group')
    for term in checks.get('forbidden_substrings', []):
        if term in answer:
            issues.append('forbidden_term:' + term)
    if parent:
        meta, pmeta = result['run']['metadata'], parent['run']['metadata']
        if (meta.get('parent_run_id') != parent['run']['run_id'] or
                meta.get('conversation_id') != pmeta.get('conversation_id') or
                meta.get('conversation_context_is_evidence') is not False):
            issues.append('followup_lineage_invalid')
    # Reader facts have source-bound spans; they are not historical Claim objects.
    report = result['run']['metadata'].get('reader_report') or {}
    audit = report.get('audit') or {}
    supported = sum(a.get('verdict') == 'SUPPORTED' for a in audit.get('assessments', []))
    if supported < checks.get('minimum_supported_claims', 0):
        issues.append('too_few_model_supported_facts')
    return {'structural_issues': list(dict.fromkeys(issues)), 'citation_count': len(citations),
            'cited_papers': sorted(sources), 'model_supported_facts': supported,
            'model_audit_coverage': audit.get('coverage'),
            'independent_source_review': 'PENDING', 'live_demo_pass': False,
            'note': 'Structure and model audit do not establish factual correctness.'}
