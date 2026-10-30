"""Experimental Writer shape; canonical facts still cross the same audit seam."""
from copy import deepcopy


FLAT_SCHEMA = (
    '{"facts":[{"id":"f1","kind":"SOURCE_FACT|INFERENCE","text":"one concise answer statement",'
    '"evidence_ids":["exact supplied ID"],"depends_on":[]}],"direct_answer_id":"f1",'
    '"limitations":["missing requested information"]}. '
)
LAYERED_SCHEMA = (
    '{"answer_facts":[{"id":"a1","kind":"SOURCE_FACT|INFERENCE",'
    '"text":"self-contained answer to a requested part",'
    '"evidence_ids":["exact supplied ID"],"depends_on":[]}],'
    '"context_facts":[{"id":"c1","kind":"SOURCE_FACT|INFERENCE",'
    '"text":"indispensable explanation, if any",'
    '"evidence_ids":["exact supplied ID"],"depends_on":[]}],'
    '"limitations":["missing requested information"]}. '
)
LAYERED_INSTRUCTION = (
    'Use answer_facts to answer exactly what was asked, not to summarize retrieved passages. '
    'Keep every required qualifier inside the answer fact so it remains valid on its own. '
    'Use context_facts only for an explanation needed to understand or justify the answer; '
    'this list may be empty. Do not repeat answer_facts there. For a names-only question, '
    'answer_facts should name the requested entities; source statistics and motivation are '
    'not additional requested parts. For a yes/no question explicitly answer Yes or No '
    'with a short self-contained statement ONLY when these sources establish that answer '
    'to the actual proposition. Lack of evidence requires uncertainty, not an invented No. '
    'For how/why, comparisons and multipart research requests preserve every requested '
    'mechanism, condition and distinction; do not replace them with a one-line topic label. '
    'When the question is ambiguous between entity levels, label the supported levels '
    'instead of silently substituting one. Do not invent facts for a section: return an '
    'empty answer_facts list and a precise limitation if no direct answer is supported. '
    'IDs must be unique across both lists; dependencies may refer to either list. '
    'Use at most 12 facts TOTAL. Do not return facts or direct_answer_id in this schema. '
    'These labels propose presentation, not proof: all statements will be independently audited.'
)


def layered_writer_instruction(system):
    if system.count(FLAT_SCHEMA) != 1:
        raise ValueError('Writer schema seam changed')
    return system.replace(FLAT_SCHEMA, LAYERED_SCHEMA) + '\n' + LAYERED_INSTRUCTION


def canonical_draft(value):
    """Change structure only, never wording, citations, kinds or dependencies."""
    if not isinstance(value, dict) or set(value) != {'answer_facts', 'context_facts', 'limitations'}:
        raise ValueError('Expected answer_facts, context_facts and limitations only')
    if any(not isinstance(value[name], list) for name in ('answer_facts', 'context_facts')):
        raise ValueError('Answer/context facts must be lists')
    facts = []
    for name, role in [('answer_facts', 'answer'), ('context_facts', 'context')]:
        for fact in value[name]:
            if not isinstance(fact, dict):
                raise ValueError('Each answer/context fact must be an object')
            facts.append({**deepcopy(fact), 'writer_role': role})
    return {'facts': facts, 'direct_answer_id': facts[0].get('id') if facts else None,
            'limitations': deepcopy(value['limitations'])}
