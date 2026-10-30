"""Keep incomplete retrieval from establishing paper-wide negative claims.

This guard does not prove semantic entailment. Evidence-relative gaps are
separate from scientific claims and cannot introduce unreviewed assertions.
"""
from copy import deepcopy
import re

ABSENCE = re.compile(
    r'不存在.{0,12}(?:表|Table)|没有\s*(?:Table|表)|原文无.{0,12}(?:表|编号)|'
    r'未(?:提供|报告|给出|说明|进行|开展|包含|提及|评估)|没有(?:提供|报告|给出|实验|进行|比较|包含|提及)|'
    r'缺乏(?:统一|直接|实验)|\b(?:does? not|did not|never|no)\s+'
    r'(?:report|provide|specify|conduct|evaluate|experiment|comparison)|'
    r'\b(?:does|do|did)\s+not\s+(?:(?:explicitly|directly)\s+)?'
    r'(?:report|provide|specify|conduct|evaluate|compare|quantify|describe)|'
    r'\bnot\s+(?:reported|provided|specified|evaluated|quantified|compared)\b|'
    r'\bno\s+(?:further\s+)?(?:details?|comparison|benchmark|protocol)\b', re.I)
GAP = re.compile(r'当前|现有|已读|所读|本轮|尚未|无法确定|无法核实|证据不足|'
                 r'\b(?:current|available|retrieved|missing|insufficient|uncertain|cannot determine|'
                 r'cannot be established|not established|no answer found|no [a-z-]+ details|'
                 r'evidence is supplied)\b', re.I)
PAPER_ABSENCE = re.compile(
    r'不存在.{0,12}(?:表|Table)|没有\s*(?:Table|表)|原文无.{0,12}(?:表|编号)|'
    r'未(?:提供|报告|给出|说明|进行|开展|包含|提及|评估)|'
    r'没有(?:提供|报告|给出|实验|进行|比较)|'
    r'\b(?:does|do|did)\s+not\s+(?:(?:explicitly|directly)\s+)?'
    r'(?:report|provide|specify|conduct|evaluate|compare|quantify|describe)|'
    r'\bnot\s+(?:reported|provided|specified|evaluated|quantified|compared)\b|'
    r'\bno\s+(?:further\s+)?(?:details?|comparison|benchmark|protocol)\b', re.I)
COMPARISON_BOUNDARY = re.compile(
    r'(?:Table|tables?|表)\s*[IVXLCDM\d].{0,100}'
    r'(?:不具可比性|不能直接比较|无法直接比较|not\s+directly\s+comparable)|'
    r'(?:这|上述|两)(?:张)?表.{0,100}(?:不具可比性|不能直接比较|无法直接比较)|'
    r'(?:these|the\s+two)\s+(?:tables?|results?).{0,100}(?:not\s+directly\s+comparable|cannot\s+be\s+directly\s+compared)', re.I)


GENERIC_GAPS = (
    '当前证据尚不足以核实全部回答要求；这不表示论文没有相关内容。',
    'The current evidence does not establish every requested point; this does not establish absence from the paper.',
)
# Preserve only an explicitly evidence-relative uncertainty clause. Do not
# carry a following explanation/claim through on the strength of this prefix.
WHETHER_GAP = re.compile(
    r'^(?:the |these )?(?:(?:supplied|retrieved|available) )?(?:passages|evidence) '
    r'(?:do|does) not (?:explicitly )?(?:state|establish|show) whether '
    r'(?P<scope>(?:\b(?:e\.g\.|i\.e\.)|(?<=\d)\.(?=\d)|[^.;\n])+)(?:[.;]|$)', re.I)
CHINESE_WHETHER_GAP = re.compile(
    r'^(?:当前|现有|所读|已读|本轮|所提供(?:的)?)(?:证据|段落|资料)(?:尚)?(?:无法|不能)(?:确定|确认|核实)'
    # Also preserve nominal targets such as hardware needs, not only 是否.
    # The causal explanation is a separate assertion, even without a comma.
    r'(?P<scope>[^。，,；;：:\n]+?)(?:[。，,；;：:]|(?=因为|由于|因此|所以|$))')

# This narrow form qualifies the current evidence and then denies an inference.
# It asserts neither paper-wide absence nor a measured advantage. A fresh,
# source-owned SCOPE_BOUNDARY assessment is required by the caller. Mixed
# positive/causal tails continue through the existing conservative path.
REVIEWED_EVIDENCE_BOUNDARY = re.compile(
    r'^(?:当前|现有|所读|已读|本轮|所提供(?:的)?)(?:证据|段落|资料)(?:尚)?'
    r'未(?:提供|包含|给出|报告)[^。，,；;：:\n]+[，,]\s*'
    r'(?:因此|所以)?(?:无法|不能)(?:证明|推断|确认|判断)'
    r'(?:(?!因为|由于|因此|所以|但是|不过|然而|但|并且|而且)[^。，,；;：:\n])+[。.]?$')


def reviewed_negative_sequence(value):
    """Preserve a whole reviewed, current-evidence negative sequence.

    No paper subject, positive/causal tail or unreviewed absence is admitted.
    This is a rendering check, not independent proof of factual correctness.
    """
    clauses = re.split(r'[；;]', value.strip().rstrip('。.'))
    if not 2 <= len(clauses) <= 4:
        return False
    if not CHINESE_WHETHER_GAP.fullmatch(clauses[0].strip()):
        return False
    forbidden = re.compile(r'因为|由于|因此|所以|从而|但是|不过|然而|并|而且|且|却')
    return all(re.fullmatch(r'未(?:提供|包含|给出|报告|记录)[^。，,；;：:\n]+', clause.strip())
               and not forbidden.search(clause) for clause in clauses[1:])


def reviewed_epistemic_boundary(value):
    """Keep whole reviewed denials of inference, never extract a positive tail."""
    if re.search(r'因为|由于|因此|所以|从而|但|然而|并且|而且|表明|证明了|'
                 r'\b(?:because|since|therefore|thus|however|moreover)\b', value, re.I):
        return False
    clauses = re.split(r'[，,；;]', value.strip().rstrip('。.'))
    if not 1 <= len(clauses) <= 4:
        return False
    prefix = (r'(?:(?:该|本|此|上述|当前|这)[^。，,；;：:\n]{0,32}'
              r'(?:比较|分析|证据|结论))?')
    head = re.compile(prefix + r'(?:本身|尚)?(?:不能|无法|不足以)'
                      r'(?:(?:证明|断言|确定|判断|推断|保证|推出)[^。，,；;：:\n]+|'
                      r'替代实验(?:结果|验证|对照|测量))$')
    english = re.compile(r'(?:(?:this|the|these)\s+(?:mechanism\s+)?'
                         r'(?:comparison|analysis|evidence|conclusions?)\s+)?'
                         r'(?:cannot|does not|do not)\s+'
                         r'(?:prove|establish|determine|guarantee|demonstrate|imply)\s+[^.;\n]+$', re.I)
    return all(head.fullmatch(c.strip()) or english.fullmatch(c.strip()) for c in clauses)


def safe_limitations(values, *, reviewed_values=()):
    visible, withheld = [], []
    reviewed = set(reviewed_values)
    for value in values:
        if value in reviewed and (REVIEWED_EVIDENCE_BOUNDARY.fullmatch(value.strip())
                                 or reviewed_negative_sequence(value)
                                 or reviewed_epistemic_boundary(value)):
            visible.append(value)
            continue
        match = WHETHER_GAP.match(value.strip())
        chinese_match = CHINESE_WHETHER_GAP.match(value.strip())
        embedded = False
        if not match and not chinese_match:
            # Keep only a self-contained uncertainty clause, not the model's
            # preceding explanation of why evidence is absent. Do not promote
            # a mixed sentence merely because it contains an uncertainty cue.
            for clause in re.split(r'[，；;\n]', value)[1:]:
                clause = re.sub(r'^(?:因此|所以|但|不过|然而)\s*', '', clause.strip())
                chinese_match = CHINESE_WHETHER_GAP.match(clause)
                if chinese_match:
                    embedded = True
                    break
        if match or chinese_match:
            scope = (match or chinese_match).group('scope').strip()
            causal_suffix = False
            if match:
                parts = re.split(r',\s*(?:as|because|since)\b', scope, maxsplit=1, flags=re.I)
                causal_suffix = len(parts) > 1
                scope = parts[0].strip()
            visible.append(f'Cannot determine from the supplied passages whether {scope}.' if match else
                           f'当前证据无法确定{scope}。')
            if embedded or causal_suffix or value.strip()[(match or chinese_match).end():].strip():
                withheld.append(value)  # retain the original mixed statement for diagnostics
            continue
        if COMPARISON_BOUNDARY.search(value) or (GAP.search(value) and not ABSENCE.search(value)):
            visible.append(value)
        else:
            withheld.append(value)
    if withheld and not visible:
        chinese = any(re.search(r'[\u4e00-\u9fff]', v) for v in withheld)
        visible.append(GENERIC_GAPS[0] if chinese else GENERIC_GAPS[1])
    return list(dict.fromkeys(visible)), withheld


def enforce_answer_boundaries(draft, audit, evidence):
    effective = deepcopy(audit)
    sources = {e['evidence_id']: e['content'] for e in evidence}
    facts = {f['id']: f for f in draft['facts']}
    assessments = {a['fact_id']: a for a in effective['assessments']}
    changed = []
    for fid, fact in facts.items():
        row = assessments[fid]
        if row['verdict'] != 'SUPPORTED':
            continue
        if (ABSENCE.search(fact['text']) and
                (fact.get('kind', 'SOURCE_FACT') == 'SOURCE_FACT' or
                 PAPER_ABSENCE.search(fact['text']))):
            # A matching negative phrase alone is not semantic confirmation.
            # Require an exact attributed sentence, already model-audited.
            normalized = ' '.join(fact['text'].split()).casefold()
            exact = any(normalized in ' '.join(sources.get(eid, '').split()).casefold()
                        for eid in fact['evidence_ids'])
            if not exact:
                row.update(verdict='UNCERTAIN', boundary_reason='paper_absence_not_established')
                changed.append(fid)
    # Resolve transitively: a rejected premise must invalidate downstream claims.
    for _ in range(len(facts)):
        updates = 0
        for fid, fact in facts.items():
            row = assessments[fid]
            if fact.get('kind') != 'INFERENCE' or row['verdict'] != 'SUPPORTED':
                continue
            deps = fact.get('depends_on')
            if deps is None:
                deps = [other['id'] for other in facts.values() if other['id'] != fid
                        and other.get('kind', 'SOURCE_FACT') == 'SOURCE_FACT'
                        and set(other['evidence_ids']) & set(fact['evidence_ids'])]
            if not deps or any(d not in assessments or assessments[d]['verdict'] != 'SUPPORTED' for d in deps):
                row.update(verdict='UNCERTAIN', boundary_reason='inference_premises_unverified')
                changed.append(fid)
                updates += 1
        if not updates:
            break
    if changed:
        effective['coverage'] = 'PARTIAL'
        effective['missing_aspects'] = list(effective.get('missing_aspects', [])) + [
            '当前证据尚不足以核实部分否定结论或比较推断的前提。']
    return effective, {'changed_fact_ids': changed, 'version': 'answer-boundaries-v2'}
