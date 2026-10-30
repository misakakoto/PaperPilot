"""Narrow request-type checks, not a semantic planner or factual evaluator.

Original user text is preserved. These helpers recognize only explicit current
mechanism-only scope and pure conditional evidence-disclosure instructions.
Ambiguous clauses remain ordinary requirements for source-based review.
"""
import re
from .numeric_scope import table_references

VERSION = 'request-obligations-v1-preserved-constraints'


def mechanism_only_request(question):
    text = question or ''
    for match in re.finditer(r'只(?:作|做|进行|讨论).{0,8}机制|仅.{0,8}机制层面|'
                            r'(?:only|solely).{0,20}mechanism', text, re.I):
        prefix = text[max(0, match.start() - 24):match.start()]
        if re.search(r'(?:不|非|不要|不必|不是)\s*$|\bnot(?:\s+\w+){0,2}\s+$', prefix, re.I):
            continue
        return True
    return False


def current_request_history(question, history):
    """Retire explicit earlier task clauses when the user narrows to mechanisms.

    Only raw initial/follow-up task records qualify. Persistent/derived state,
    source restrictions, prohibitions, language and resource clauses remain.
    Ambiguous mixed clauses are preserved. Every retained fragment is verbatim;
    the caller retains original history separately, never edits prior records.
    """
    from copy import deepcopy
    if not mechanism_only_request(question):
        return deepcopy(history), []
    active, retired = [], []
    constraints = re.compile(
        r'依据|基于|来源|范围|排除|禁止|不要|不得|不许|不能|不启动|不运行|不训练|'
        r'中文|英文|简洁|详细|显存|预算|\d\s*(?:GB|MB)|'
        r'\b(?:based on|source|scope|exclude|must not|do not|never|Chinese|English|'
        r'concise|detailed|budget|memory|VRAM)\b', re.I)
    task = re.compile(
        r'^(?:请|再|然后|并|分别|继续|同时|进一步|还要|需要|要|仅|只|\s)*'
        r'(?:比较|对比|说明|解释|分析|列出|给出|写出|报告|引用|计算|回答|总结|转录|'
        r'compare|describe|explain|list|report|cite|calculate|summarize|write)', re.I)
    for index, row in enumerate(history):
        if row.get('role') not in {'initial_request', 'follow_up_request'}:
            active.append(deepcopy(row))
            continue
        text = row['user_request']
        for match in re.finditer(r'[^，。；！？\n]+[，。；！？]?', text):
            clause = match.group(0).strip()
            if task.search(clause) and not constraints.search(clause):
                retired.append({'history_index': index, 'user_request': clause,
                                'reason': 'earlier_task_superseded_by_current_mechanism_only_scope'})
            elif clause:
                active.append({**deepcopy(row), 'user_request': clause})
    return active, retired


def conditional_evidence_policy(text):
    if not isinstance(text, str):
        return False
    text = text.strip().strip('。；;.!！?？')
    return bool(re.fullmatch(
        r'(?:若|如果|当)?(?:证据|依据|信息|原文证据)(?:不足|不充分|缺失)'
        r'(?:时|则)?[，, ]*(?:请|应|需|必须|要)?(?:明确|如实)?'
        r'(?:说明|指出|标注|告知)(?:证据不足|不确定性)?'
        r'|if (?:the )?evidence is (?:insufficient|missing|unavailable),? '
        r'(?:please )?(?:state|say|disclose|explain)(?: this| so| it)?(?: explicitly| clearly)?',
        text, re.I))


def superseded_numeric_quote(quote, question):
    """Current explicit mechanism-only work excludes old measurement requests.

    Source restrictions and output language/style remain applicable. This does
    not erase history and does not treat an ordinary continuation as a reset.
    """
    if not isinstance(quote, str):
        return False
    # A mixed clause may carry a still-valid source, style or negative rule.
    # Keep it for review; a numerical keyword alone cannot retire that rule.
    if re.search(r'排除|禁止|不要|不得|不许|范围|来源|依据|基于|只(?:使用|引用)|仅(?:使用|引用)|'
                 r'中文|英文|\b(?:exclude|excluding|must not|do not|only use|only cite|'
                 r'based on|scope|source|Chinese|English)\b', quote, re.I):
        return False
    return bool(quote not in (question or '')
                and mechanism_only_request(question)
                and (table_references(quote) or re.search(
                    r'消融|数值|实验结果|定量结果|\bablation\b|\bnumeric(?:al)?\b|'
                    r'\breported results?\b', quote, re.I)))
