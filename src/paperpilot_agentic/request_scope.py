"""Resolve explicit source exclusions and table ownership before retrieval.

Uses the caller's indexed-title resolver; never invents hashes or PDF paths.
"""
import re

from .numeric_scope import table_references

EXCLUSION = re.compile(
    r'(?:不要(?:混入|引用|使用|包含)?|不(?:包括|包含|使用|引用)|排除|除外|'
    r'\b(?:exclude|excluding|without|do not (?:use|include|cite))\b)'
    r'\s*([^，,。；;！？\n]+)', re.I)
POSSESSIVE_OWNER = re.compile(
    r'(?<![A-Za-z0-9_-])(?P<name>[A-Za-z][A-Za-z0-9_-]{2,63})\s*(?:的|[\'\u2019]s)\s*')


def resolve_request_scope(question, resolve):
    excluded = set()
    for match in EXCLUSION.finditer(question):
        excluded.update(resolve(match[1]))
    positive = EXCLUSION.sub(' ', question)
    included = set(resolve(positive)) - excluded
    owners = {sha for match in POSSESSIVE_OWNER.finditer(positive)
              for sha in resolve(match.group('name'))} - excluded
    if owners:
        # ``FlowMamba 的 BRIGHT 实验`` names one paper and one dataset;
        # it must not silently expand into a paper whose title is BRIGHT.
        included = owners
    return {'included': sorted(included), 'excluded': sorted(excluded),
            'positive_query': positive,
            'needs_clarification': bool(excluded and not included)}


def numeric_lookup_query(question):
    """Return a dataset-neutral exact-value retrieval query when present."""
    values = re.findall(r'(?<![A-Za-z0-9_.+-])\d+(?:\.\d+)?\s*[%％]', question)
    return ' '.join(dict.fromkeys(value.replace('％', '%').replace(' ', '')
                                  for value in values)) or None


def visual_policy(metadata):
    policy = metadata.get('visual_policy', 'explicit')
    if policy not in {'auto', 'off', 'explicit'}:
        raise ValueError('Unknown visual_policy')
    if metadata.get('allow_visual') is False:
        return 'off'
    return policy


def automatic_table_targets(question, scope, resolve):
    targets, gaps = [], []
    refs = table_references(question)
    titles = list(re.finditer(r'《([^《》]+)》', question))
    # An explicit "respectively cite" list supplies an order mapping. Without
    # that operator, two selected papers and two tables remain ambiguous.
    if len(refs) == 2 and len(titles) == 2 and titles[-1].end() < refs[0]['start']:
        directive = re.search(r'分别(?:引用|参照|使用|读取|报告)\s*$', question[:refs[0]['start']])
        connector = question[refs[0]['end']:refs[1]['start']].strip()
        if (directive and connector in {'和', '与', '、'}
                and not re.search(r'不要|不必|无需|如果|假如|假设',
                                  question[max(titles[-1].end(), directive.start()-12):directive.start()])):
            ordered = [set(resolve(match.group(1))) & set(scope) for match in titles]
            if (all(len(owners) == 1 for owners in ordered)
                    and ordered[0] != ordered[1] and ordered[0] | ordered[1] == set(scope)):
                return [{'paper_sha256': next(iter(owners)),
                         'label': re.sub(r'^(?:表|table)\s*', 'Table ', ref['text'], flags=re.I)}
                        for owners, ref in zip(ordered, refs, strict=True)], []
    previous = 0
    for ref in refs:
        # Bind each table to the nearest preceding explicit source phrase;
        # no cross-product of all papers with all table numbers.
        prefix = question[previous:ref['start']]
        owners = set(resolve(prefix)) & set(scope)
        if not owners and len(scope) == 1:
            owners = set(scope)
        if len(owners) != 1:
            gaps.append({'label': ref['text'], 'reason': 'ambiguous_table_owner'})
        else:
            label = re.sub(r'^(?:表|table)\s*', 'Table ', ref['text'], flags=re.I)
            target = {'paper_sha256': next(iter(owners)), 'label': label}
            if target not in targets:
                targets.append(target)
        previous = ref['end']
    if len(targets) > 2:
        gaps.extend({**t, 'reason': 'automatic_table_limit'} for t in targets[2:])
        targets = targets[:2]
    return targets, gaps


def table_target_query(original_question, resolved_question):
    """A search rewrite may resolve a requested table, not invent a table request."""
    original_refs = table_references(original_question)
    if original_refs:
        if ([r['table'] for r in original_refs] !=
                [r['table'] for r in table_references(resolved_question)]):
            return original_question
        return resolved_question
    if re.search(
            r'\btables?\b|表格|原表|表中|表内|该表|此表|(?:上|前|这|那).{0,3}(?:张|个)?表',
            original_question, re.I):
        return resolved_question
    return original_question


def prioritize_requested_evidence(evidence, targets):
    wanted = {(t['paper_sha256'], table_references(t['label'])[0]['table']) for t in targets}
    def priority(row):
        uri = str(row.get('source_uri', ''))
        sha = row.get('paper_sha256') or (uri.split('/')[2] if uri.startswith('paper://') else '')
        refs = table_references(row['content'])
        requested = any((sha, r['table']) in wanted for r in refs)
        if requested and 'Named-cell transcription' in row['content']:
            return 0
        if requested and re.search(r'(?m)^\s*TABLE\s+', row['content']):
            return 1
        return 2 if requested else 3
    return sorted(evidence, key=priority)
