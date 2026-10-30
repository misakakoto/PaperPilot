"""Experimental answer-first generation and independent source audit.

Not the default runtime. Candidate relevance is not promoted to factual support:
supporting spans must belong to the answer statement's own cited passages.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from .verification import grounded_quote_match_mode

QUANTITY_POLICY = (
    'For a requested count, preserve the exact counted entity and scope. Component existence, '
    'a singular noun, component width/dimensionality, or a hyperparameter search range does not '
    'establish multiplicity or total system depth. Do not substitute a subcomponent count for '
    'a whole-system count. A numerical answer requires an explicit count of the requested entity '
    'or a demonstrably exhaustive enumeration; do not fill it from architectural convention. '
    'If this evidence is absent, state the precise quantity cannot be determined from these '
    'sources. Do not label a plausible default as SOURCE_FACT or approve it in the audit.'
)

RELATION_POLICY = (
    'For every SOURCE_FACT, apply a source-only entailment check to every clause. Identify the '
    'subject, asserted relation, object and qualifying scope, and locate that same '
    'relation in this fact\'s own allowed source spans. Merely mentioning the subject '
    'or a related tool, venue, dataset, software or hardware does not establish its '
    'unstated properties. Familiarity, conventional practice and absence of contrary '
    'evidence are not source evidence. Ask whether these spans could still be true '
    'if the claimed property were false; if yes, the claim is not entailed. Reject '
    'unsupported explanatory clauses even when another clause is true. '
    'For INFERENCE, independently ground every premise and empirical subclause, then '
    'evaluate the stated derivation or explicitly scoped interpretation against those '
    'premises. Do not require the derived relation itself to be stated or measured by '
    'the authors. A labeled mechanism-level interpretation is not a claim of measured '
    'superiority or proof of effectiveness; assess its limited wording, not a stronger '
    'unstated claim. Reject invented properties, causal effects, measurements and '
    'unsupported generalizations regardless of kind. Conversely, accept direct '
    'statements and faithful paraphrases without requiring verbatim wording or '
    'extra conditions not requested. Preserve valid sub-scope answers; do not '
    'require an exhaustive global claim for a question that did not ask for one.'
)

PAPER_SCOPE_POLICY = (
    'Resolve paper scope before answering. A paper title can name a family of the authors\' '
    'methods, not one literal model name. When asked which datasets or tasks the PAPER uses, '
    'include the authors\' own task-specific variants when the supplied introduction, methods, '
    'results or captions establish that ownership. Keep each dataset bound to its actual task '
    'and variant; do not erase a reported experiment solely because its variant name differs '
    'from the paper title. Exclude datasets only mentioned as background or prior work. '
    'If the user instead restricts the question to one variant or stage, retain that restriction. '
    'Do not promote evaluation use into training use, or copy scores between variants.'
)

QUESTION_SCOPE_POLICY = (
    'Question alignment is separate from truth. Resolve the requested entity, experimental '
    'unit, dataset and stage from the question and supplied paper context before classifying '
    'DIRECT. The writer direct_answer_id is only a proposal. A true result for a different '
    'unit is not a direct answer just because it uses the same metric. For example, an '
    'illustrative individual outcome does not supply an aggregate experimental result. '
    'Explicitly labeling that different scope makes the sentence honest, but does not make '
    'it answer the requested scope. When a paper identifies its main method and the question '
    'asks for its performance, preserve that referent; do not substitute examples, intermediate '
    'artifacts, ablations or comparison systems. If the referent genuinely cannot be resolved, '
    'record that specific ambiguity as a gap rather than declaring an arbitrary alternative '
    'DIRECT. A pointer to an unavailable table establishes where results are reported, not '
    'their values. Conversely, when the question explicitly asks for that example, intermediate '
    'stage or comparison system, a faithful answer in that scope IS DIRECT. Preserve supported '
    'answers to requested subparts; missing another requested part only makes coverage partial. '
    'Do not downgrade a fact from SUPPORTED merely because it is not the requested answer; '
    'classify relevance separately. These instructions refine, not weaken, source entailment.'
)

# A normal pass uses Writer and Auditor plus one shared format repair. Local
# isolation has one separate content patch followed by a full audit, even when
# a format repair was used. The physical-attempt cap below is unchanged; provider
# retries and every stage also obey the persistent run/campaign budget.
MAX_PROVIDER_ATTEMPTS = 9


def evidence_spans(evidence: list[dict]) -> list[dict]:
    """Number exact source slices so the model selects, rather than rewrites, quotes."""
    spans = []
    for row in evidence:
        content = row['content']
        start = 0
        # PDF line wrapping is not a semantic boundary. Keep it verbatim within
        # a span; only sentence boundaries and blank paragraphs split spans.
        boundaries = [m.start() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z])|\n\s*\n', content)]
        for end in boundaries + [len(content)]:
            text = content[start:end]
            left = len(text) - len(text.lstrip())
            right = len(text.rstrip())
            if right > left:
                spans.append({'span_id': f"s{len(spans)}",
                              'evidence_id': row['evidence_id'], 'start': start + left,
                              'end': start + right, 'text': text[left:right]})
            start = end
    return spans


def audit_source_view(spans: list[dict]) -> list[dict]:
    """Transmit each source ID once; retain every exact span text and local ID."""
    grouped = {}
    for span in spans:
        group = grouped.setdefault(span['evidence_id'], {'evidence_id': span['evidence_id'], 'spans': {}})
        group['spans'][span['span_id']] = span['text']
    return list(grouped.values())


def audit_facts_view(draft: dict, spans: list[dict]) -> list[dict]:
    return [{**fact, 'allowed_span_ids': [s['span_id'] for s in spans
                                         if s['evidence_id'] in fact['evidence_ids']]}
            for fact in draft['facts']]


def requires_chinese_answer(question, request_history=None):
    """Resolve clear positive language edits from owned requests, not evidence.

    This is a limited language gate, not general instruction understanding.
    Quoted, hypothetical and negative clauses cannot change the language state.
    """
    language = None
    requests = [r.get('user_request', '') for r in (request_history or [])]
    for text in requests + [question or '']:
        text = re.sub(r'"[^"\n]*"|“[^”\n]*”|「[^」\n]*」', '', text)
        for clause in re.split(r'[。！？!?；;\n]', text):
            if re.search(r'如果|假如|是否|不要|不用|无需|不必|不能|\b(?:if|not|never)\b|don.t', clause, re.I):
                continue
            edits = list(re.finditer(
                r'(?:^|[,，:：])\s*(?:请|后续|以后|这次|本次|全程|始终)*(?:改为|改用|使用|用|以)'
                r'(中文|英文|英语|汉语)(?:来)?(?:回答|回复|作答|输出|解释|撰写)'
                r'|(?:^|[,，:：])\s*(?:please\s+)?(?:answer|respond|reply|write)\s+(?:only\s+)?in\s+(English|Chinese)',
                clause, re.I))
            for edit in edits:
                name = (edit.group(1) or edit.group(2)).lower()
                language = 'zh' if name in {'中文', '汉语', 'chinese'} else 'en'
    return language == 'zh' if language else bool(question and re.search(r'[\u4e00-\u9fff]', question))


def validate_fact_shape(fact, allowed):
    if isinstance(fact, dict) and 'purpose' in fact:
        raise ValueError('Explicit purpose must be projected before fact validation')
    """Shared strict per-fact contract; passing it grants no factual support."""
    if not isinstance(fact, dict) or not isinstance(fact.get('text'), str) or not fact['text'].strip():
        raise ValueError('Invalid fact text')
    fid = fact.get('id')
    if not isinstance(fid, str) or not fid:
        raise ValueError('Invalid or duplicate fact ID')
    if fact.get('kind', 'SOURCE_FACT') not in {'SOURCE_FACT', 'INFERENCE'}:
        raise ValueError('Fact kind must distinguish source fact from inference')
    citations = fact.get('evidence_ids')
    if not isinstance(citations, list) or not citations or any(not isinstance(e, str) or e not in allowed for e in citations):
        raise ValueError(f'Fact must cite supplied evidence: fact_id={fid}. '
                         'If this is a user constraint or evidence limitation rather than a '
                         'source-supported fact, move it to limitations; never invent a citation.')


def validate_fact_dependencies(fact, by_id):
    deps = fact.get('depends_on', [])
    if not isinstance(deps, list) or any(not isinstance(d, str) or d not in by_id or d == fact['id'] for d in deps):
        raise ValueError('Inference dependencies must reference other draft facts')
    if fact.get('kind', 'SOURCE_FACT') == 'INFERENCE' and not deps:
        raise ValueError('Inference dependencies must list supporting SOURCE_FACT IDs')
    if any(by_id[dep].get('kind', 'SOURCE_FACT') != 'SOURCE_FACT' for dep in deps):
        raise ValueError('Inference dependencies must reference SOURCE_FACT IDs')


def validate_draft(value: Any, evidence: list[dict], *, question=None, request_history=None, comparison=False,
                   table_rows=None) -> dict:
    allowed = {e['evidence_id'] for e in evidence}
    if not isinstance(value, dict) or not isinstance(value.get('facts'), list):
        raise ValueError('Expected facts list')
    facts = value['facts']
    limitations = value.get('limitations', [])
    if not isinstance(limitations, list) or any(not isinstance(x, str) for x in limitations):
        raise ValueError('Limitations must be text items')
    if len(facts) > 12:
        raise ValueError('Expected at most 12 answer facts')
    if not facts and not value.get('limitations'):
        raise ValueError('Abstention requires an explicit limitation')
    ids = set()
    for fact in facts:
        validate_fact_shape(fact, allowed)
        fid = fact.get('id')
        if fid in ids:
            raise ValueError('Invalid or duplicate fact ID')
        ids.add(fid)
    if facts and value.get('direct_answer_id') not in ids:
        raise ValueError('Draft must identify its direct answer fact')
    by_id = {fact['id']: fact for fact in facts}
    for fact in facts:
        validate_fact_dependencies(fact, by_id)
    from .table_claims import row_catalog, validate_row_refs
    validate_row_refs(value, row_catalog(evidence) if table_rows is None else table_rows)
    if question and unrequested_structural_formula(question, facts):
        raise ValueError('Unrequested structural formula: explain the mechanism in words; '
                         'do not publish OCR equations when the user did not request formulas')
    if (comparison != 'single-value-v2' and requires_chinese_answer(question, request_history) and facts and
            not any(re.search(r'[\u4e00-\u9fff]', fact['text']) for fact in facts)):
        raise ValueError('Answer language must match the Chinese original question')
    if requires_chinese_answer(question, request_history):
        for index, limitation in enumerate(limitations):
            if limitation.strip() and not re.search(r'[\u4e00-\u9fff]', limitation):
                raise ValueError(f'Answer limitation language must match the Chinese original question: '
                                 f'limitation_index={index}. Rewrite the entire caveat in Chinese; '
                                 'preserve its scope and uncertainty, do not delete it.')
    if comparison:
        from .paper_comparison import validate_records, prepare_single_value_records
        if comparison == 'single-value-v2':
            # Structured scientific names may remain in their source language;
            # this is extraction, not the user-facing prose answer. UI labels,
            # gap explanations and report framing remain Chinese.
            value = prepare_single_value_records(value)
        validate_records(value)
    return value


def normalize_duplicate_limitations(value: Any) -> tuple[Any, list[str]]:
    """Remove only an unreferenced, uncited gap duplicated verbatim in limitations.

    This is a lossless schema repair, not a factual verdict. Do not move or
    rewrite source claims, infer citations, or erase a dependency target.
    """
    if (not isinstance(value, dict) or not isinstance(value.get('facts'), list)
            or not isinstance(value.get('limitations'), list)):
        return value, []
    facts = value['facts']
    references = ({value['direct_answer_id']} if isinstance(value.get('direct_answer_id'), str) else set())
    for fact in facts:
        if isinstance(fact, dict) and isinstance(fact.get('depends_on'), list):
            references.update(dep for dep in fact['depends_on'] if isinstance(dep, str))
    removed = [fact['id'] for fact in facts if isinstance(fact, dict)
               and isinstance(fact.get('id'), str) and fact['id'] not in references
               and sum(isinstance(other, dict) and other.get('id') == fact['id'] for other in facts) == 1
               and fact.get('kind') == 'LIMITATION' and fact.get('evidence_ids') == []
               and isinstance(fact.get('text'), str) and fact['text'] in value['limitations']]
    if not removed:
        return value, []
    normalized = json.loads(json.dumps(value))
    normalized['facts'] = [fact for fact in normalized['facts']
                           if not isinstance(fact, dict) or fact.get('id') not in removed]
    return normalized, removed


def normalize_inference_dependencies(value: Any) -> tuple[Any, list[str]]:
    """Attach only structurally provable SOURCE_FACT dependencies.

    Some JSON-capable models omit ``depends_on`` even after a format repair.
    Dependency metadata can be recovered without changing answer semantics
    only when SOURCE_FACT citations collectively cover every evidence source
    cited by the inference. The independent Auditor still decides whether the
    inference follows from those premises; this function never grants support.
    """
    if not isinstance(value, dict) or not isinstance(value.get('facts'), list):
        return value, []
    normalized = json.loads(json.dumps(value))
    facts = normalized['facts']
    changed = []
    for fact in facts:
        if not isinstance(fact, dict) or fact.get('kind') != 'INFERENCE' or fact.get('depends_on'):
            continue
        cited = {eid for eid in fact.get('evidence_ids', []) if isinstance(eid, str)}
        premises = [candidate for candidate in facts
                    if isinstance(candidate, dict) and candidate is not fact
                    and candidate.get('kind', 'SOURCE_FACT') == 'SOURCE_FACT'
                    and cited.intersection(candidate.get('evidence_ids', []))]
        covered = {eid for premise in premises for eid in premise.get('evidence_ids', []) if eid in cited}
        if cited and covered == cited:
            fact['depends_on'] = [premise['id'] for premise in premises]
            changed.append(fact['id'])
    return normalized, changed


def normalize_inference_citations(value: Any, evidence: list[dict]) -> tuple[Any, list[str]]:
    """Recover source references only from explicit, valid SOURCE_FACT premises.

    Missing citations or an exact copy of depends_on are schema defects. Never
    repair an arbitrary unknown source, a mixed namespace, a recursive premise
    or a SOURCE_FACT. This records provenance, not entailment; every candidate
    still crosses the independent auditor and deterministic guards.
    """
    if not isinstance(value, dict) or not isinstance(value.get('facts'), list):
        return value, []
    facts = value['facts']
    ids = [fact.get('id') for fact in facts if isinstance(fact, dict)]
    if (len(ids) != len(facts) or any(not isinstance(fid, str) or not fid for fid in ids)
            or len(ids) != len(set(ids))):
        return value, []
    allowed = {item['evidence_id'] for item in evidence}
    by_id = {fact['id']: fact for fact in facts}
    normalized = json.loads(json.dumps(value))
    changed = []
    for fact in normalized['facts']:
        deps = fact.get('depends_on')
        if (fact.get('kind') != 'INFERENCE' or not isinstance(deps, list) or not deps
                or any(not isinstance(dep, str) or dep == fact['id'] or dep not in by_id
                       for dep in deps)):
            continue
        cited = fact.get('evidence_ids')
        missing = cited is None or cited == []
        copied = (isinstance(cited, list) and all(isinstance(eid, str) for eid in cited)
                  and set(cited) == set(deps) and not (set(cited) & allowed))
        if not (missing or copied):
            continue
        sources = []
        for dep in deps:
            premise = by_id[dep]
            citations = premise.get('evidence_ids')
            if (premise.get('kind', 'SOURCE_FACT') != 'SOURCE_FACT'
                    or not isinstance(citations, list) or not citations
                    or any(not isinstance(eid, str) or eid not in allowed for eid in citations)):
                break
            sources.extend(citations)
        else:
            fact['evidence_ids'] = list(dict.fromkeys(sources))
            changed.append(fact['id'])
    return (normalized, changed) if changed else (value, [])


def formulas_requested(question):
    return bool(re.search(r'公式|方程|数学|推导|表达式|损失函数|函数|formula|equation|mathematical|deriv|function|loss',
                          question or '', re.I))


def unrequested_structural_formula(question, facts):
    if formulas_requested(question):
        return False
    return any(isinstance(fact, dict) and re.search(
        r'=\s*\[|\bHW\s*/|=\s*concat[_a-z]*\s*\(', str(fact.get('text', '')), re.I)
        for fact in facts)


def draft_repair_diagnostics(value: Any, *, question: str, primary: Exception, request_history=None,
                             allow_table_facts=False) -> str:
    """Report independent deterministic contract failures in one repair turn."""
    errors = [str(primary)]
    facts = value.get('facts', []) if isinstance(value, dict) else []
    factual_items = [fact for fact in facts if isinstance(fact, dict)
                     and fact.get('kind', 'SOURCE_FACT') in {'SOURCE_FACT', 'INFERENCE'}
                     and fact.get('evidence_ids')] if isinstance(facts, list) else []
    # An illegal Chinese LIMITATION entry cannot supply the language of the
    # remaining English facts. Report both failures in the one allowed repair.
    language_items = factual_items or [fact for fact in facts if isinstance(fact, dict)] if isinstance(facts, list) else []
    if (requires_chinese_answer(question, request_history) and language_items and
            not any(re.search(r'[\u4e00-\u9fff]', str(fact.get('text', '')))
                    for fact in language_items)):
        errors.append('Answer language must match the Chinese original question')
    limitations = value.get('limitations', []) if isinstance(value, dict) else []
    if requires_chinese_answer(question, request_history) and isinstance(limitations, list):
        for index, limitation in enumerate(limitations):
            if isinstance(limitation, str) and limitation.strip() and not re.search(r'[\u4e00-\u9fff]', limitation):
                errors.append(f'limitation_index={index}: Rewrite the entire caveat in Chinese; '
                              'preserve its scope and uncertainty, do not delete it')
    if isinstance(facts, list) and unrequested_structural_formula(question, facts):
        errors.append('Remove unrequested structural formulas; explain token organization in words')
        for fact in facts:
            if unrequested_structural_formula(question, [fact]):
                errors.append(f"fact_id={fact.get('id')}: Rewrite the entire statement in words; "
                              'preserve the requested token order or channel operation and its citations. '
                              'Remove sequence equations, dimensions, denominators and concat notation; '
                              'do not delete the mechanism or keep the equation as an inference')
    if isinstance(facts, list):
        ids = {fact.get('id') for fact in facts if isinstance(fact, dict)}
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            # Writer selections are validated/projected by the host before the
            # ordinary fact contract. They have row_refs, not evidence_ids/text.
            # This flag describes that input schema; it does not approve a row.
            if allow_table_facts and fact.get('kind') == 'TABLE_FACT':
                continue
            if fact.get('kind', 'SOURCE_FACT') not in {'SOURCE_FACT', 'INFERENCE'}:
                errors.append(f"fact_id={fact.get('id')}: kind must be SOURCE_FACT or INFERENCE; "
                              'put evidence gaps in the top-level limitations string array')
            if not fact.get('evidence_ids'):
                errors.append(f"fact_id={fact.get('id')}: facts require citations; "
                              'move evidence gaps to limitations without inventing citations')
        for fact in facts:
            if not isinstance(fact, dict) or fact.get('kind') != 'INFERENCE':
                continue
            deps = fact.get('depends_on', [])
            if not isinstance(deps, list) or not deps or any(dep not in ids for dep in deps):
                errors.append('INFERENCE facts require valid supporting SOURCE_FACT IDs in depends_on')
                break
    return '; '.join(dict.fromkeys(errors))


def validate_audit(value: Any, draft: dict, evidence: list[dict], *, spans=None, require_relevance=False,
                   question=None, request_history=None) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get('assessments'), list):
        raise ValueError('Expected assessments list')
    facts = {f['id']: f for f in draft['facts']}
    sources = {e['evidence_id']: e['content'] for e in evidence}
    by_span = {s['span_id']: s for s in spans} if spans is not None else None
    rows = value['assessments']
    ids = [r.get('fact_id') for r in rows if isinstance(r, dict)]
    if len(ids) != len(rows) or any(not isinstance(fid, str) for fid in ids):
        raise ValueError('Each assessment requires a string fact_id; expected=' + json.dumps(list(facts)))
    if len(ids) != len(rows) or len(ids) != len(set(ids)) or set(ids) != set(facts):
        details = {'expected': list(facts), 'missing': sorted(set(facts) - set(ids)),
                   'unexpected': sorted(set(ids) - set(facts)),
                   'duplicates': sorted({fid for fid in ids if ids.count(fid) > 1})}
        raise ValueError('Audit must cover each fact exactly once: ' + json.dumps(details))
    for row in rows:
        if (require_relevance or 'relevance' in row) and row.get('relevance') not in {
                'DIRECT', 'NECESSARY_CONTEXT', 'TANGENTIAL'}:
            raise ValueError('Audit requires explicit question relevance')
        if row.get('verdict') not in {'SUPPORTED', 'UNSUPPORTED', 'UNCERTAIN'}:
            raise ValueError('Unknown audit verdict')
        if row['verdict'] == 'SUPPORTED':
            if by_span is not None:
                selected = row.get('span_ids')
                if not isinstance(selected, list) or not selected:
                    raise ValueError('Support requires selected source spans')
                if len(selected) > 12:
                    raise ValueError('Support must select at most 12 minimal source spans per fact')
                quotes = []
                for sid in selected:
                    if not isinstance(sid, str) or sid not in by_span:
                        raise ValueError('Unknown source span')
                    span = by_span[sid]
                    eid = span['evidence_id']
                    if eid not in facts[row['fact_id']]['evidence_ids']:
                        allowed_spans = [s['span_id'] for s in spans
                                         if s['evidence_id'] in facts[row['fact_id']]['evidence_ids']]
                        raise ValueError(f"Span outside fact citations: fact_id={row['fact_id']}, "
                                         f"invalid_span={sid}, allowed_span_ids={allowed_spans}")
                    quote = sources[eid][span['start']:span['end']]
                    if quote != span['text']:
                        raise ValueError('Source span mismatch')
                    quotes.append({'evidence_id': eid, 'quote': quote})
                row = {**row, 'quotes': quotes}
            quotes = row.get('quotes')
            if not isinstance(quotes, list) or not quotes:
                raise ValueError('Support requires source quotes')
            for quote in quotes:
                eid = quote.get('evidence_id') if isinstance(quote, dict) else None
                if eid not in facts[row['fact_id']]['evidence_ids']:
                    raise ValueError('Quote outside fact citations')
                if not grounded_quote_match_mode(str(quote.get('quote') or ''), sources[eid]):
                    raise ValueError('Audit quote not in cited source')
    if value.get('coverage') not in {'COMPLETE', 'PARTIAL', 'UNANSWERABLE'}:
        raise ValueError('Unknown coverage assessment')
    missing = value.get('missing_aspects', [])
    if not isinstance(missing, list) or any(not isinstance(x, str) for x in missing):
        raise ValueError('Missing aspects must be text items')
    omitted = value.get('omitted_answer_spans', [])
    if (not isinstance(omitted, list) or len(omitted) > 12 or
            any(not isinstance(sid, str) or by_span is None or sid not in by_span for sid in omitted)):
        raise ValueError('Omitted answer items must reference at most 12 supplied original spans')
    if omitted:
        chinese = any(re.search(r'[\u4e00-\u9fff]', fact['text']) for fact in facts.values())
        notice = ('当前已取回原文中的部分相关条目尚未覆盖，需要补充核对。' if chinese else
                  'Some relevant items in the available original passages are not yet covered and need review.')
        # Preserve the provider's raw response, but never display COMPLETE when
        # its own source-bound omission list is nonempty.
        value = {**value, 'coverage': 'PARTIAL', 'omitted_answer_spans': list(dict.fromkeys(omitted)),
                 'missing_aspects': list(dict.fromkeys(missing + [notice]))}
    from .answer_integrity import validate_answer_conflicts
    from .answer_coverage import validate_requirement_assessments
    validate_requirement_assessments(value, draft)
    from .answer_gap_display import validate_request_quotes
    validate_request_quotes(value, question, request_history)
    from .answer_completion import validate_limitation_assessments
    validate_limitation_assessments(value, draft)
    validate_answer_conflicts(value, draft)
    if 'comparison_records' in draft:
        from .paper_comparison import validate_record_audit
        validate_record_audit(value, draft)
    return value


def audit_repair_diagnostics(value, draft, spans, *, question, primary, request_history=None):
    """Collect independent format violations before spending the ONE repair.

    No output is repaired or marked supported here. The normal strict validator
    still owns acceptance; this only avoids repairing the first error in isolation.
    """
    errors = [str(primary)]
    if not isinstance(value, dict):
        return '; '.join(errors)
    owned = {f['id']: f for f in draft['facts']}
    by_span = {s['span_id']: s for s in spans}
    rows = value.get('assessments', [])
    for index, row in enumerate(rows if isinstance(rows, list) else []):
        if not isinstance(row, dict) or row.get('verdict') != 'SUPPORTED':
            continue
        selected = row.get('span_ids')
        path = f'assessments[{index}].span_ids'
        if not isinstance(selected, list) or not selected:
            errors.append(path + ': select 1..12 original owned spans or downgrade support')
            continue
        if len(selected) > 12:
            errors.append(f'{path}: {len(selected)} items; maximum 12 minimal owned spans')
        fact = owned.get(row.get('fact_id'), {})
        if any(not isinstance(sid, str) or sid not in by_span or
               by_span[sid]['evidence_id'] not in fact.get('evidence_ids', []) for sid in selected):
            errors.append(path + ': contains spans outside the fact citations')
    omitted = value.get('omitted_answer_spans', [])
    if not isinstance(omitted, list):
        errors.append('omitted_answer_spans: expected a list, maximum 12')
    elif len(omitted) > 12:
        errors.append(f'omitted_answer_spans: {len(omitted)} items; maximum 12; '
                      'select minimal representatives and preserve all genuine gaps in missing_aspects')
    from .answer_gap_display import validate_request_quotes
    requirements = value.get('requirement_assessments', [])
    if isinstance(requirements, list):
        if len(requirements) > 12:
            errors.append(f'requirement_assessments: {len(requirements)} items; maximum 12')
        for index, row in enumerate(requirements):
            if not isinstance(row, dict):
                continue
            try:
                validate_request_quotes({'requirement_assessments': [row]}, question, request_history)
            except ValueError as exc:
                errors.append(f'requirement_assessments[{index}]: {exc}')
    return '; '.join(dict.fromkeys(errors))


class AuditSupportError(ValueError):
    """A completed review retained no supported facts; not a spending error."""


def conservative_audit_contract(value, draft, spans, *, question, request_history=None,
                                allow_rejected_only=False):
    """Turn a still-invalid repaired citation selection into uncertainty.

    This never upgrades support. It is used only after the single model repair
    has failed, so malformed span choices do not crash an otherwise auditable
    run or get silently accepted.

    allow_rejected_only is for a pending isolated content patch: a report with
    no supported facts can proceed as rejection/navigation metadata, with bad
    caveats withheld. It does not authorize publishing a rejected statement.
    """
    if not isinstance(value, dict) or not isinstance(value.get('assessments'), list):
        raise ValueError('Cannot conservatively normalize a missing audit')
    facts = {fact['id']: fact for fact in draft['facts']}
    by_span = {span['span_id']: span for span in spans}
    rows = value['assessments']
    if {row.get('fact_id') for row in rows if isinstance(row, dict)} != set(facts) or len(rows) != len(facts):
        raise ValueError('Cannot normalize an audit that does not cover every fact once')
    normalized = []
    changed = False
    omitted = value.get('omitted_answer_spans', [])
    bad_navigation = (not isinstance(omitted, list) or len(omitted) > 12 or any(
        not isinstance(sid, str) or sid not in by_span for sid in omitted))
    if bad_navigation:
        changed = True
    from .answer_gap_display import validate_request_quotes
    from .answer_coverage import validate_requirement_assessments
    requirements, requirement_errors = [], []
    for row in value.get('requirement_assessments', []):
        try:
            # Display anchors alone do not establish ownership of facts or
            # limitation indices. Withhold the complete invalid mapping;
            # never guess a replacement index or upgrade task coverage.
            validate_requirement_assessments({'requirement_assessments': [row]}, draft)
            validate_request_quotes({'requirement_assessments': [row]}, question, request_history)
        except ValueError as exc:
            requirement_errors.append({'raw_assessment': row, 'error': str(exc)})
            changed = True
        else:
            requirements.append(row)
    # A broken caveat-to-request mapping cannot invalidate separately audited
    # facts. Withhold the unreviewed caveats rather than invent their roles.
    limitation_errors = []
    from .answer_completion import validate_limitation_assessments
    try:
        validate_limitation_assessments({**value, 'requirement_assessments': requirements}, draft)
    except ValueError as exc:
        limitation_errors.append({'raw_assessments': value.get('limitation_assessments'),
                                  'error': str(exc)})
        changed = True
    for row in rows:
        fact = facts[row['fact_id']]
        selected = row.get('span_ids') if isinstance(row.get('span_ids'), list) else []
        invalid = [sid for sid in selected if not isinstance(sid, str) or sid not in by_span or
                   by_span[sid]['evidence_id'] not in fact['evidence_ids']]
        current = dict(row)
        if row.get('verdict') == 'SUPPORTED' and (invalid or not selected or len(selected) > 12):
            changed = True
            current.update(verdict='UNCERTAIN', span_ids=[],
                           reason='The repaired audit citation selection failed structural validation; support remains unverified.')
        normalized.append(current)
    if not changed:
        raise ValueError('Conservative audit normalization found no safe downgrade')
    if not any(row.get('verdict') == 'SUPPORTED' for row in normalized):
        if not any(row.get('verdict') == 'SUPPORTED' for row in rows):
            if not allow_rejected_only:
                raise AuditSupportError('No source-supported facts remain after audit; unreviewed caveats withheld')
            # A pending isolated patch can consume this rejection report as
            # navigation. No fact gains support; malformed caveats stay hidden.
        else:
            raise ValueError('All repaired audit support relied on invalid citation spans')
    chinese = bool(re.search(r'[\u4e00-\u9fff]', question))
    return {**value, 'assessments': normalized, 'coverage': 'PARTIAL',
            **({'limitation_assessments': None, 'limitation_contract_errors': limitation_errors,
                'withheld_limitation_indices': list(range(len(draft.get('limitations', []))))}
               if limitation_errors else {}),
            **({'requirement_assessments': requirements,
                'requirement_contract_errors': requirement_errors} if requirement_errors else {}),
            **({'omitted_answer_spans': list(dict.fromkeys(
                    sid for sid in (omitted if isinstance(omitted, list) else [])
                    if isinstance(sid, str) and sid in by_span))[:12],
                'unvalidated_omitted_answer_spans': deepcopy(omitted),
                'navigation_contract_error': 'invalid_omitted_span_identity_or_cardinality'} if bad_navigation else {}),
            'missing_aspects': list(value.get('missing_aspects', [])) + ([
                '原问题子项映射未通过校验，完整性仍未核验；已审核事实保留。' if chinese else
                'Request subpart mapping failed validation; completeness remains unverified and audited facts are retained.']
                if (requirement_errors or limitation_errors) and not bad_navigation and all(
                    r.get('verdict') == 'SUPPORTED' for r in normalized) else [
                '部分引用选择未通过结构校验，相关结论保留为不确定。' if chinese else
                'Some citation selections failed structural validation; the affected claims remain uncertain.'])}


def audited_presentation(draft: dict, audit: dict, *, question=None, request_history=None,
                         host_presentation=None) -> dict:
    """Keep model-identified gaps visible, without inventing a replacement answer."""
    grounded = {r['fact_id'] for r in audit['assessments'] if r['verdict'] == 'SUPPORTED'}
    irrelevant = {r['fact_id'] for r in audit['assessments'] if r.get('relevance') == 'TANGENTIAL'}
    supported = grounded - irrelevant
    # A writer nomination is not an audit. Explicit reviewed relevance wins;
    # only historical reports without relevance retain the old nomination.
    direct_ids = {row['fact_id'] for row in audit['assessments'] if row['fact_id'] in supported and
                  (row.get('relevance') == 'DIRECT' or
                   ('relevance' not in row and row['fact_id'] == draft.get('direct_answer_id')))}
    direct = bool(direct_ids)
    facts = [f for f in draft['facts'] if f['id'] in supported]
    # Put the designated answer first, not a preceding background statement.
    facts.sort(key=lambda f: (f['id'] not in direct_ids, f['id'] != draft.get('direct_answer_id')))
    from .answer_boundaries import GENERIC_GAPS, safe_limitations
    gap_diagnostics = None
    if question is None:
        # Preserve compatibility for historical diagnostic callers, not runtime publication.
        limitations, withheld_limitations = safe_limitations(list(dict.fromkeys(x.strip() for x in
            draft.get('limitations', []) + audit.get('missing_aspects', []) if x.strip())))
    else:
        from .answer_gap_display import user_visible_gaps
        limitations, withheld_limitations, gap_diagnostics = user_visible_gaps(
            draft, audit, question=question, request_history=request_history,
            chinese=requires_chinese_answer(question, request_history), host_presentation=host_presentation)
    from .answer_completion import reviewed_boundary_only
    boundary_only = reviewed_boundary_only(draft, audit, question=question, request_history=request_history)
    # Quarantine a model-invented paper-wide negative for diagnostics. If all
    # requested facts passed and coverage is complete, the rejected sentence is
    # not itself a runtime retrieval gap and must not downgrade the answer.
    if (withheld_limitations and all(value in GENERIC_GAPS for value in limitations)
            and audit['coverage'] == 'COMPLETE' and all(
            row['verdict'] == 'SUPPORTED' or row.get('relevance') == 'TANGENTIAL'
            for row in audit['assessments'])):
        limitations = []
    distinguish = any(f.get('kind') == 'INFERENCE' for f in draft['facts'])
    def present(fact):
        prefix = ''
        if distinguish:
            chinese = bool(re.search(r'[\u4e00-\u9fff]', fact['text']))
            inference = fact.get('kind') == 'INFERENCE'
            prefix = ('比较推断：' if inference else '原文事实：') if chinese else (
                'Inference: ' if inference else 'Source fact: ')
        return prefix + fact['text'] + ' ' + ' '.join(f'[{eid}]' for eid in fact['evidence_ids'])
    return {
        'answer': '\n\n'.join(present(f) for f in facts),
        'answer_sections': [
            {'kind': kind, 'text': '\n\n'.join(present(f) for f in group)}
            for kind, group in [('direct', [f for f in facts if f['id'] in direct_ids]),
                                ('context', [f for f in facts if f['id'] not in direct_ids])]
            if group],
        'answer_units': [present(f) for f in facts] + limitations,
        'direct_answer_supported': direct,
        'answer_scope': ('BACKGROUND_ONLY' if not direct and facts else
                         'NO_SUPPORTED_ANSWER' if not facts else
                         'PARTIAL' if audit['coverage'] != 'COMPLETE' or
                         audit.get('missing_aspects') or (limitations and not boundary_only) or
                         (gap_diagnostics and gap_diagnostics['unrendered_scope_boundaries']) or
                         any(r['verdict'] != 'SUPPORTED' and r.get('relevance') != 'TANGENTIAL'
                             for r in audit['assessments']) else 'MODEL_REVIEWED'),
        'limitations': limitations,
        'unverified_limitations': withheld_limitations,
        'completion_assurance': 'MODEL_REVIEWED_NOT_FACTUAL_ACCEPTANCE',
        'limitation_roles': audit.get('limitation_assessments'),
        **({'internal_gap_diagnostics': gap_diagnostics} if gap_diagnostics is not None else {}),
        'omitted_tangential_fact_ids': sorted(irrelevant),
        'all_facts_supported': bool(draft['facts']) and len(grounded) == len(draft['facts']),
    }


