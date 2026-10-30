"""Conservative, source-located method-name hints; never scientific proof.

Only explicit first-person naming clauses on the opening pages are used. Mere
mentions (including related work and references) cannot establish ownership.
Ambiguous names remain ambiguous. Raw offsets always refer to registered text.
"""
import re


def method_hints(units):
    result = {}
    pattern = re.compile(
        r'\bwe\s+(?:(?:here|also|further)\s+)?(?:propose|introduce|present)\b'
        r'[^.!?;]{0,220}?\((?P<alias>[A-Za-z][A-Za-z0-9_-]{1,29})\)', re.I)
    for uid,u in units.items():
        if u['page_start'] > 2:
            continue
        for match in pattern.finditer(u['text']):
            alias = match['alias']
            if alias.islower():
                continue
            rows = result.setdefault(u['paper_sha'],[])
            if any(r['name'].casefold()==alias.casefold() for r in rows):
                continue
            rows.append({'name':alias,'unit_id':uid,'range':[match.start(),match.end()],
                'quote':match[0], 'assurance':'EXPLICIT_SELF_NAMING_HINT_NOT_REVIEWED'})
    return result


def check_claim_identity(state, claim):
    """Catch explicit handle/known subject mismatches, not semantic entailment."""
    source = state['read_handles'].get(claim.get('source_id'))
    if not source:
        return  # Ordinary admission explains missing sources.
    fact = claim.get('fact')
    if not isinstance(fact,str):
        raise ValueError('A source fact must be text')
    owner = source['paper_id']
    mentioned = set(re.findall(r'(?<![A-Za-z0-9])P[1-9][0-9]*(?![A-Za-z0-9])',fact))
    if mentioned - {owner}:
        raise ValueError('CLAIM_PAPER_MISMATCH: split claims by source paper; '+source['source_id']+' belongs to '+owner)
    # A quote can legitimately discuss another method. Only reject a named
    # subject owned elsewhere when that name is also absent from the quote.
    hints = state.get('method_identity_hints',{})
    owners = {}
    for pid,rows in hints.items():
        for row in rows:
            owners.setdefault(row['name'].casefold(),set()).add(pid)
    for alias,papers in owners.items():
        boundary = r'(?![A-Za-z0-9_-])'
        if (len(papers)==1 and owner not in papers
                and re.match(r'\s*'+re.escape(alias)+boundary,fact,re.I)
                and not re.search(r'(?<![A-Za-z0-9_-])'+re.escape(alias)+boundary,claim.get('quote',''),re.I)):
            raise ValueError('CLAIM_METHOD_MISMATCH: named subject and cited paper differ; inspect original method identity')
