"""Freeze once, then execute bounded acceptance in an isolated copy of the corpus.

python -m evals.interview.run --freeze
python -m evals.interview.run --execute-paid
All subsequent invocations reuse the same spending ledger and frozen case results.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from .budget import CampaignBudget, RecordedProvider, RecordedVision, canonical, save, source_fingerprint
from .cases import acceptance_cases, CHANGE, FLOW
from .checks import structural_observations
from .timeouts import request_timeouts

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'outputs/interview-acceptance-20261001'
CAMPAIGN_FOLDER = FOLDER


def campaign_budget(protocol):
    if protocol.get('remaining_grant'):
        from .remaining import RemainingCampaignBudget
        return RemainingCampaignBudget(CAMPAIGN_FOLDER/'receipts', protocol['identity'],
                                       grant=protocol['remaining_grant'])
    # Protocol revisions share the original ledger and absolute campaign ceilings.
    return CampaignBudget(CAMPAIGN_FOLDER / 'receipts', protocol['identity'],
        max_requests=protocol.get('max_requests', 60), max_tokens=protocol.get('max_tokens', 600000))


def answer_budget(protocol):
    """Frozen per-run allowance; campaign checks remain the shared authority."""
    from paperpilot_agentic.models import Budget
    if protocol.get('identity', {}).get('usage_limit_mode') == 'unlimited':
        return Budget(unlimited_llm_usage=True, max_seconds=900)
    tokens = protocol.get('answer_run_max_tokens', 150000)
    if (type(tokens) is not int or tokens <= 0 or
            tokens > protocol.get('max_tokens', 600000)):
        raise ValueError('Answer-run budget must fit the frozen campaign allowance')
    return Budget(max_llm_calls=16, max_tokens=tokens, max_seconds=900)


def model_preflight(provider, folder):
    """A small, counted readiness request before loading local semantic models."""
    path = Path(folder)/'preflight-result.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if 'campaign_identity' in saved and saved['campaign_identity'] != json.loads(provider.campaign.identity):
            raise ValueError('Preflight identity changed')
        return saved['ready']
    if hasattr(provider.campaign, 'grant'):
        from .selection_preflight import selection_preflight
        return selection_preflight(local_configuration(), provider.campaign, folder)
    if json.loads(provider.campaign.identity).get('capability_preflight_models'):
        from .capabilities import capability_preflight
        return capability_preflight(local_configuration(), provider.campaign, folder)
    try:
        response = provider.complete([{'role': 'user', 'content': '仅输出 ready。'}],
            max_tokens=32, thinking_enabled=False, json_mode=False)
        ready = response.finish_reason == 'stop' and response.text.strip().casefold() == 'ready'
        save(path, {'ready': ready, 'finish_reason': response.finish_reason})
    except Exception as exc:
        save(path, {'ready': False, 'error_type': type(exc).__name__})
        ready = False
    if not ready and not provider.campaign.snapshot()['blocked']:
        save(provider.campaign.folder/'stopped.json', {'reason': 'model_preflight_not_ready'})
    return ready


def local_configuration():
    from paperpilot_agentic.config import RuntimeSettings
    settings = RuntimeSettings.from_env()
    for name in ('PAPERPILOT_ALLOW_MODEL_DOWNLOAD', 'PAPERPILOT_ALLOW_EXTERNAL_SCHOLARLY'):
        os.environ[name] = 'false'
    # Preserve configured retrieval mode and cached models; downloads are disabled.
    protocol_path = FOLDER/'protocol.json'
    frozen = json.loads(protocol_path.read_text(encoding='utf-8')) if protocol_path.exists() else None
    selection = FOLDER/'selected-model.json'
    if frozen and frozen.get('remaining_grant') and selection.exists():
        selected = json.loads(selection.read_text(encoding='utf-8'))
        if selected['identity'] != frozen['identity']:
            raise ValueError('Selected model identity changed')
        settings = replace(settings, model=selected['model'])
    return replace(settings, workspace=FOLDER, runtime_db=FOLDER / 'runtime.sqlite3',
                   memory_db=FOLDER / 'corpus.sqlite3', thinking_enabled=False,
                   timeout_seconds=request_timeouts(frozen, settings.timeout_seconds)['text_seconds'],
                   fallback_models=())


def retained_unknown_bounds(previous):
    """Carry complete requests or prior reviewed bounds, without settling usage."""
    rows = previous.snapshot()['attempts']
    with previous.transaction() as conn:
        approved = [json.loads(r[0]) for r in conn.execute('SELECT authorization FROM continuations')]
    bounds = {}
    for row in rows:
        if row['state'] not in {'RESERVED', 'STARTED', 'UNKNOWN', 'FAILED'}:
            continue
        if row['state'] != 'UNKNOWN':
            raise ValueError('A running or failed request cannot be carried as UNKNOWN')
        request_path = previous.folder / f"request-{row['id']:03}.json"
        earlier = [a['unknown_bounds'][str(row['id'])] for a in approved
                   if str(row['id']) in a['unknown_bounds']]
        if request_path.is_file():
            request = json.loads(request_path.read_text(encoding='utf-8'))
            if (request.get('model') != row['model'] or not request.get('messages')
                    or type(request.get('max_tokens')) is not int or request['max_tokens'] <= 0
                    or request.get('thinking_enabled') is not False):
                raise ValueError('A complete bounded non-thinking text request is required')
            bound = len(canonical(request['messages']).encode('utf-8')) + request['max_tokens'] + 2048
            bounds[row['id']] = max(bound, row['reservation'], *earlier)
        elif earlier:
            bounds[row['id']] = max(earlier)
        else:
            raise ValueError('Missing request needs an existing independently reviewed cost bound')
    return bounds


def supplemental_instruction(parent_campaign, note=None):
    """Keep the original approved CLI compatible; never invent later approvals."""
    if isinstance(note, str) and note.strip():
        return note.strip()
    if (CAMPAIGN_FOLDER.resolve() == (ROOT/'outputs/qa-acceptance-20261005').resolve()
            and Path(parent_campaign).resolve() == (ROOT/'outputs/interview-acceptance-20261001').resolve()):
        return '2026-10-05 user approved implementation with 30 requests / 300000 tokens'
    raise ValueError('A new supplement requires its current human budget authorization')


def authorize_supplemental_revision(previous, identity, note, *, carry_unknown=False, resume_json_rejection=False):
    """Keep ordinary repairs fail-closed; explicit resumes carry retained cost."""
    if not note or not note.strip():
        raise ValueError('A current human continuation instruction is required')
    evidence = {'human_instruction': note, 'allowance_unchanged': True}
    if resume_json_rejection:
        return previous.authorize_json_rejection_revision(identity, evidence=evidence)
    if carry_unknown:
        evidence.update({'old_states_remain': 'UNKNOWN', 'old_results_preserved': True,
                         'bound_basis': 'Retained complete requests, UTF-8 bytes + output + overhead'})
        return previous.authorize_continuation(identity,
            unknown_bounds=retained_unknown_bounds(previous), evidence=evidence)
    return previous.authorize_settled_revision(identity, evidence=evidence)


def validate_supplemental_model(settings, parent, authorized_model, note):
    from paperpilot_agentic.model_policy import FREE_MODELS
    if authorized_model is not None and (
            authorized_model not in FREE_MODELS or settings.model != authorized_model or
            settings.provider != 'qwen' or not note):
        raise ValueError('New snapshot requires explicit matching model and human authorization')
    if settings.provider != parent['identity']['provider'] or (
            settings.model != parent['identity']['model'] and authorized_model is None):
        raise ValueError('A supplemental campaign retains the approved model and provider')


def freeze(*, continuation=False, continuation_note=None, carry_unknown=False, parent_campaign=None, supplemental_note=None,
           max_requests=60, max_tokens=600000, capability_models=None, request_cost_policy=None, remaining_from=None,
           authorized_model=None, resume_json_rejection=False, extra_case_revalidation=None, unlimited_usage=False):
    if unlimited_usage:
        if continuation or remaining_from or not parent_campaign or not supplemental_note:
            raise ValueError('Unlimited usage requires a new explicitly authorized campaign with parent lineage')
        max_requests = max_tokens = None
    grant = None
    if remaining_from:
        from .remaining import grant_remaining
        grant = grant_remaining(remaining_from, FOLDER, supplemental_note)
        parent_campaign = remaining_from
        max_requests, max_tokens = grant['max_requests'], grant['max_tokens']
    approval = supplemental_instruction(parent_campaign, supplemental_note) if parent_campaign else None
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.bootstrap import build_runtime
    settings = RuntimeSettings.from_env()
    if capability_models is not None:
        from paperpilot_agentic.model_policy import PRIMARY_MODEL, BACKUP_MODEL
        if (list(capability_models) != [PRIMARY_MODEL, BACKUP_MODEL] or not parent_campaign or continuation
                or (settings.model, settings.provider) != (PRIMARY_MODEL, 'qwen')
                or settings.base_url != 'https://dashscope.aliyuncs.com/compatible-mode/v1'
                or (max_requests, max_tokens) != (30, 300000) or not supplemental_note):
            raise ValueError('Capability campaign requires the authorized Beijing model pair and fresh 30/300000 allowance')
    parent = None
    historical = None
    if parent_campaign:
        from .campaign import historical_snapshot
        historical = historical_snapshot(parent_campaign)
        parent = historical['protocol']
        if not capability_models:
            validate_supplemental_model(settings, parent, authorized_model, approval)
        if acceptance_cases(ROOT) != parent['cases']:
            raise ValueError('A supplemental campaign preserves all frozen questions')
    if continuation:
        parent = json.loads((CAMPAIGN_FOLDER / 'protocol.json').read_text(encoding='utf-8'))
        # A source repair is a revision of the same remainder grant, never a
        # new supplemental allowance or a conversion to the legacy ledger.
        grant = parent.get('remaining_grant')
        if settings.model != parent['identity']['model'] or settings.provider != parent['identity']['provider']:
            raise ValueError('Authorized continuation must retain the original model')
        settings = replace(settings, memory_db=CAMPAIGN_FOLDER / 'corpus.sqlite3')
        previous = campaign_budget(parent)
        max_requests, max_tokens = parent.get('max_requests', 60), parent.get('max_tokens', 600000)
        historical = parent.get('historical_cost')
        attempts = previous.snapshot()['attempts']
        if not continuation_note and not (len(attempts) == 2 and all(a['state'] == 'UNKNOWN' for a in attempts)):
            raise ValueError('A later continuation requires the current human instruction, not automatic replay')
        if continuation_note and not historical:
            unknown_bounds = retained_unknown_bounds(previous)
    if (FOLDER / 'protocol.json').exists():
        raise ValueError('Protocol already frozen; do not reset the acceptance allowance')
    deadlines = request_timeouts(parent if continuation else None, settings.timeout_seconds)
    if historical and not continuation:
        settings = replace(settings, memory_db=Path(historical['protocol_path']).parent/'corpus.sqlite3')
    elif continuation:
        settings = replace(settings, memory_db=CAMPAIGN_FOLDER/'corpus.sqlite3')
    if not settings.memory_db or not settings.memory_db.is_file():
        raise ValueError('An indexed source corpus is required')
    FOLDER.mkdir(parents=True, exist_ok=True)
    if historical:
        from .campaign import backup_parent
        cache_parent = historical if not continuation else {**historical, 'protocol_path': str(CAMPAIGN_FOLDER/'protocol.json')}
        backup_parent(cache_parent, FOLDER)
    else:
        with sqlite3.connect(settings.memory_db.as_uri() + '?mode=ro', uri=True) as original:
            with sqlite3.connect(FOLDER / 'corpus.sqlite3') as copy:
                original.backup(copy)
    config = replace(local_configuration(), timeout_seconds=deadlines['text_seconds'])
    runtime = build_runtime(config)
    store = runtime.search_adapter.store
    papers = store.paper_metadata({CHANGE, FLOW})
    if {p['sha256'] for p in papers} != {CHANGE, FLOW}:
        raise ValueError('Frozen original demo papers are missing from the index')
    originals = []
    for paper in papers:
        ids = [cid for section in store.paper_outline(paper['sha256']) for cid in section['chunk_ids']]
        chunks = store.get_chunks(ids)
        originals.append({'paper': paper, 'chunks': [
            {'stable_id': c.stable_id, 'text': c.text,
             'sha256': hashlib.sha256(c.text.encode()).hexdigest()} for c in chunks]})
    save(FOLDER / 'original-sources.json', originals)
    cases = acceptance_cases(ROOT)
    identity = {'campaign': CAMPAIGN_FOLDER.name, 'model': settings.model,
                'provider': settings.provider, 'source_sha256': source_fingerprint(ROOT),
                'request_timeout': deadlines}
    if unlimited_usage or (continuation and parent['identity'].get('usage_limit_mode') == 'unlimited'):
        identity['usage_limit_mode'] = 'unlimited'
    # Existing revisions retain their policy. New protocols use the same image
    # admission rule as daily execution; old missing fields mean legacy UTF-8.
    if continuation:
        inherited_policy = parent['identity'].get('request_cost_policy', 'legacy-utf8-v1')
        if request_cost_policy is not None and request_cost_policy != inherited_policy:
            raise ValueError('A revision retains the frozen request cost policy')
        if 'request_cost_policy' in parent['identity']:
            identity['request_cost_policy'] = inherited_policy
    else:
        from paperpilot_agentic.model_policy import PRIMARY_MODEL, BACKUP_MODEL, DAILY_MODEL
        identity['request_cost_policy'] = request_cost_policy or (
            'qwen3-vl-still-image-v1' if settings.provider == 'qwen'
            and settings.model in {PRIMARY_MODEL, BACKUP_MODEL} else
            'qwen35-still-image-v1' if settings.provider == 'qwen' and settings.model == DAILY_MODEL
            else 'legacy-utf8-v1')
        if (identity['request_cost_policy'] == 'qwen3-vl-still-image-v1' and
                (settings.provider != 'qwen' or settings.model not in {PRIMARY_MODEL, BACKUP_MODEL})):
            raise ValueError('Image bound requires the confirmed Qwen3-VL model pair')
    if capability_models:
        identity['capability_preflight_models'] = list(capability_models)
    if grant:
        identity['remaining_protocol'] = grant['version']
        identity['remaining_grant_sha256'] = hashlib.sha256(canonical(grant).encode()).hexdigest()
        if authorized_model:
            # Explicit fixed-model recovery never silently tries the old pair.
            identity['selection_models'] = [authorized_model]
        elif continuation and parent['identity'].get('selection_models'):
            identity['selection_models'] = parent['identity']['selection_models']
    if continuation:
        identity['revision'] = FOLDER.name
        if parent['identity'].get('capability_preflight_models'):
            identity['capability_preflight_models'] = parent['identity']['capability_preflight_models']
        if cases != parent['cases']:
            raise ValueError('Continuation must preserve all frozen questions and checks')
    protocol = {'identity': identity, 'cases': cases, 'engine': 'answer_first_text',
                'request_timeout': deadlines, 'timeout_seconds': deadlines['text_seconds'],
                'visual_policy': 'auto', 'max_requests': max_requests, 'max_tokens': max_tokens,
                'answer_run_max_tokens': None if max_tokens is None else min(max_tokens, 300000),
                'reader_protocol': runtime.text_reader_protocol,
                'temperature': config.temperature, 'thinking_enabled': config.thinking_enabled,
                'max_output': config.max_tokens, 'base_url_sha256': hashlib.sha256(config.base_url.encode()).hexdigest(),
                'memory_namespace': config.memory_namespace,
                'model_preflight': {'question': '仅输出 ready。', 'max_tokens': 32},
                'retrieval_environment': {name: os.getenv(name) for name in (
                    'PAPERPILOT_ENABLE_DENSE', 'PAPERPILOT_ENABLE_RERANKER',
                    'PAPERPILOT_RERANK_STAGE', 'PAPERPILOT_EMBEDDING_MODEL', 'PAPERPILOT_RERANKER_MODEL')},
                'source_snapshot_sha256': hashlib.sha256((FOLDER/'original-sources.json').read_bytes()).hexdigest(),
                'independent_benchmark': False, 'review': 'manual original-source review required'}
    if extra_case_revalidation:
        if not continuation or not continuation_note or extra_case_revalidation not in {c['id'] for c in cases}:
            raise ValueError('One extra frozen case requires a documented continuation')
        from .retry_grant import grant_extra_case
        protocol['extra_case_revalidation'] = grant_extra_case(
            CAMPAIGN_FOLDER, identity, extra_case_revalidation, continuation_note)
    if capability_models:
        protocol['model_preflight'] = {'models': list(capability_models), 'max_requests': 4,
            'text_json': {'ready': True}, 'image_json': {'color': 'red'},
            'image': {'width': 64, 'height': 64, 'fill': 'red', 'background': 'white'},
            'max_tokens_per_request': 128, 'answers_model': settings.model}
    inherited_preflight = None
    if continuation and parent['identity'].get('capability_preflight_models'):
        from .capabilities import inherit_preflight
        inherited_preflight = inherit_preflight(parent, identity, CAMPAIGN_FOLDER, config)
        protocol['model_preflight'] = parent['model_preflight']
    if historical:
        protocol['historical_cost'] = {key: value for key, value in historical.items() if key != 'protocol'}
        protocol['supplemental_authorization'] = {
            'human_instruction': approval if not continuation else parent['supplemental_authorization']['human_instruction'],
            'authorized_limits': {'max_requests': max_requests, 'max_tokens': max_tokens},
            'scope': 'Original nine cases and context trace; Qwen/DashScope/source authorization retained.'}
    if authorized_model:
        protocol['authorized_model_change'] = {'from': parent['identity']['model'],
            'to': authorized_model, 'human_instruction': approval, 'parent_preserved': True}
    if settings.provider == 'qwen' and settings.model == 'qwen3.5-plus-2026-02-15':
        from paperpilot_agentic.reviewed_sources import ReviewedSources
        protocol['reviewed_sources_sha256'] = ReviewedSources.configured().identity()
    if grant:
        protocol['remaining_grant'] = grant
        from paperpilot_agentic.reviewed_sources import ReviewedSources
        protocol['reviewed_sources_sha256'] = ReviewedSources.configured().identity()
        protocol['model_preflight'] = {'kind': 'combined-system-image-json', 'max_requests': 2,
                                     'fallback_only_after_explicit_unavailable': True}
    if continuation:
        if historical:
            protocol['continuation'] = authorize_supplemental_revision(
                previous, identity, continuation_note, carry_unknown=carry_unknown,
                resume_json_rejection=resume_json_rejection)
        elif not continuation_note:
            # Legacy first continuation: this missing visual request was
            # reviewed against the dated provider hard limit before approval.
            if settings.model != 'qwen3-vl-flash-2025-10-15':
                raise ValueError('The original missing-request bound applies only to the frozen model')
            request = json.loads((CAMPAIGN_FOLDER / 'receipts/request-001.json').read_text(encoding='utf-8'))
            if request.get('thinking_enabled') is not False:
                raise ValueError('Text bound requires the retained non-thinking request')
            text_bound = len(canonical(request['messages']).encode()) + request['max_tokens'] + 2048
            unknown_bounds = {1: text_bound, 2: 262144 + 32768}
        if not historical:
            evidence = {
            'human_instruction': continuation_note or '2026-10-02 continuation: user requested 开始真实验收 and 继续',
            'old_results_preserved': True, 'old_states_remain': 'UNKNOWN',
            'text_bound_basis': 'Retained complete request, UTF-8 bytes + output + overhead',
            'visual_bound_basis': 'Previously reviewed continuation bound; never settled as zero usage',
            'provider_spec': 'https://platform.qianwenai.com/docs/developer-guides/getting-started/vision-models',
            'assurance': 'Conservative cost carry, not provider usage reconciliation or zero billing'}
            protocol['continuation'] = previous.authorize_continuation(identity,
                unknown_bounds=unknown_bounds, evidence=evidence)
        protocol['parent_protocol_sha256'] = hashlib.sha256((CAMPAIGN_FOLDER/'protocol.json').read_bytes()).hexdigest()
    save(FOLDER / 'protocol.json', protocol)
    if inherited_preflight:
        save(FOLDER/'preflight-result.json', inherited_preflight)
    campaign_budget(protocol)
    print(json.dumps({'frozen': str(FOLDER), 'model': settings.model, 'cases': len(cases),
                      'actual_requests': 0}, ensure_ascii=True), flush=True)


def execute(*, max_new_cases=None, case_ids=None, stage=None):
    if max_new_cases is not None and (type(max_new_cases) is not int or max_new_cases < 1):
        raise ValueError('max_new_cases must be a positive integer')
    from paperpilot_agentic.bootstrap import build_runtime
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    from paperpilot_agentic.qwen_vision import QwenVisionEvidenceAdapter
    protocol = json.loads((FOLDER / 'protocol.json').read_text(encoding='utf-8'))
    frozen_answer_budget = answer_budget(protocol)
    if protocol.get('historical_cost') and stage != 'context':
        max_new_cases = 1 if max_new_cases is None else max_new_cases
        if max_new_cases != 1:
            raise ValueError('Supplemental acceptance reviews one new answer at a time')
    selected = {case['id'] for case in protocol['cases']} if case_ids is None else set(case_ids)
    if not selected or selected - {case['id'] for case in protocol['cases']}:
        raise ValueError('Select only frozen case IDs')
    if protocol.get('historical_cost') and case_ids is not None and all(
            (FOLDER/(case_id+'-result.json')).exists() for case_id in selected):
        from .campaign import report
        data = report(FOLDER, campaign_budget(protocol))
        print(json.dumps({'completed_responses_reused': sorted(selected), 'budget': data['budget']}))
        return
    for case in protocol['cases']:
        parent_id = case.get('parent_scenario_id')
        if (case['id'] in selected and parent_id and parent_id not in selected
                and not (FOLDER/(parent_id+'-result.json')).exists()):
            raise ValueError('A selected follow-up requires its same-revision parent result')
    if protocol.get('historical_cost') and stage != 'context':
        for case in protocol['cases']:
            if (FOLDER/(case['id']+'-result.json')).exists():
                from .campaign import sha
                review_path = FOLDER/'source-reviews'/(case['id']+'.json')
                review = json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else {}
                if (review.get('case') != case['id'] or
                        review.get('result_sha256') != sha(FOLDER/(case['id']+'-result.json'))):
                    raise ValueError('Review the saved answer against original sources before continuing')
                continue
            if case['id'] not in selected:
                raise ValueError('Supplemental answers retain the frozen case order')
            attempts = [CAMPAIGN_FOLDER/(case['id']+'-submission.json')]
            attempts += list((CAMPAIGN_FOLDER/'revisions').glob('*/'+case['id']+'-submission.json'))
            from .retry_grant import submission_ceiling
            if (protocol['identity'].get('usage_limit_mode') != 'unlimited' and
                    sum(path.exists() for path in attempts) >= submission_ceiling(CAMPAIGN_FOLDER, protocol, case['id'])):
                raise ValueError('A case allows only its initial execution and one repair revalidation')
            if (FOLDER/(case['id']+'-submission.json')).exists():
                raise ValueError('Interrupted submission requires receipt inspection; do not enqueue again')
            break
    campaign = campaign_budget(protocol)
    if campaign.snapshot()['blocked']:
        print(json.dumps({'stopped': 'failed_or_unknown_request', 'budget': campaign.snapshot()}))
        return
    settings = local_configuration()
    if protocol.get('reviewed_sources_sha256'):
        from paperpilot_agentic.reviewed_sources import ReviewedSources
        if ReviewedSources.configured().identity() != protocol['reviewed_sources_sha256']:
            raise ValueError('Reviewed source assets changed after freeze')
    if (protocol['identity']['source_sha256'] != source_fingerprint(ROOT) or
            protocol['identity']['model'] != settings.model and not protocol.get('remaining_grant') or
            protocol['identity']['provider'] != settings.provider or
            protocol['source_snapshot_sha256'] != hashlib.sha256((FOLDER/'original-sources.json').read_bytes()).hexdigest() or
            protocol['temperature'] != settings.temperature or
            protocol['thinking_enabled'] != settings.thinking_enabled or
            protocol['max_output'] != settings.max_tokens or
            protocol['base_url_sha256'] != hashlib.sha256(settings.base_url.encode()).hexdigest()):
        raise ValueError('Frozen source/model changed; preserve results and reconcile before more calls')
    from .transport import transport_preflight
    transport = transport_preflight(settings, FOLDER)
    if not transport['reachable']:
        summary = {'rows': [], 'not_run': [c['id'] for c in protocol['cases']],
                   'budget': campaign.snapshot(), 'live_demo_passes': 0,
                   'independent_source_review': 'NOT_RUN_TRANSPORT_UNAVAILABLE',
                   'transport': transport}
        (FOLDER/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=True), flush=True)
        return
    provider = RecordedProvider(OpenAICompatibleLLM(settings, max_retries=0), campaign, phase='preflight')
    if not model_preflight(provider, FOLDER):
        summary = {'rows': [], 'not_run': [c['id'] for c in protocol['cases']],
                   'budget': campaign.snapshot(), 'live_demo_passes': 0,
                   'independent_source_review': 'NOT_RUN_MODEL_PREFLIGHT_FAILED'}
        (FOLDER/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=True), flush=True)
        return
    if protocol.get('remaining_grant'):
        settings = local_configuration()
        provider = RecordedProvider(OpenAICompatibleLLM(settings, max_retries=0), campaign, phase='answers')
    if settings.provider == 'qwen':
        vision = QwenVisionEvidenceAdapter(api_key=settings.api_key, model=settings.model,
            base_url=settings.base_url,
            timeout_seconds=request_timeouts(protocol, settings.timeout_seconds)['vision_seconds'])
        vision = RecordedVision(vision, campaign, phase='visual-auto')
    else:
        raise ValueError('Auto visual acceptance requires the frozen multimodal Qwen model')
    runtime = build_runtime(settings, llm_factory=lambda config: provider,
                            vision_factory=lambda **kwargs: vision)
    expected_reader = protocol['reader_protocol']
    if protocol.get('remaining_grant'):
        expected_reader = expected_reader.replace(protocol['identity']['model'], settings.model)
    if runtime.text_reader_protocol != expected_reader:
        raise ValueError('Frozen retrieval protocol changed')
    if stage == 'context':
        if any(not (FOLDER/(case['id']+'-result.json')).exists() for case in protocol['cases']):
            raise ValueError('Context follows all nine frozen case results')
        from .context import live_context
        provider.phase = 'context-ablation'
        live_context(runtime, FOLDER, paired=bool(protocol.get('historical_cost')))
        return
    rows, results, executed = [], {}, 0
    for case in protocol['cases']:
        path = FOLDER / (case['id'] + '-result.json')
        if path.exists():
            result = json.loads(path.read_text(encoding='utf-8'))
        else:
            if case['id'] not in selected:
                continue
            if max_new_cases is not None and executed >= max_new_cases:
                break
            if campaign.snapshot()['blocked']:
                break
            parent = results.get(case.get('parent_scenario_id'))
            provider.phase = case['id']
            vision.client.chat.completions.phase = case['id'] + ':vision'
            metadata = {'engine': 'answer_first_text', 'visual_policy': 'auto', 'paper_sha256s': case['paper_ids'],
                        'execution_profile': 'frozen_acceptance'}
            budget = frozen_answer_budget
            if parent:
                queued = runtime.enqueue_follow_up(parent_run_id=parent['run']['run_id'],
                    question=case['question'], budget=budget, metadata=metadata)
            else:
                queued = runtime.enqueue(question=case['question'], budget=budget, metadata=metadata)
            save(FOLDER / (case['id'] + '-submission.json'), {'question': case['question'],
                'run_id': queued['run']['run_id'], 'paper_ids': case['paper_ids']})
            started = time.monotonic()
            executed += 1
            try:
                runtime.resume(queued['run']['run_id'])
            except Exception as exc:
                # Provider receipts contain only sanitized failure fields.
                save(FOLDER / (case['id'] + '-exception.json'), {'type': type(exc).__name__})
            result = runtime.status(queued['run']['run_id'])
            result['acceptance_elapsed_seconds'] = round(time.monotonic() - started, 3)
            save(path, result)
        results[case['id']] = result
        row = {'case': case['id'], 'run_id': result['run']['run_id'],
               'status': result['run']['status'], 'elapsed_seconds': result.get('acceptance_elapsed_seconds'),
               'run_llm_calls': result['run']['used_llm_calls'], 'run_tokens': result['run']['used_tokens'],
               **structural_observations(case, result, results.get(case.get('parent_scenario_id')))}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
        if campaign.snapshot()['blocked']:
            break
    if stage is None and not campaign.snapshot()['blocked'] and len(results) == len(protocol['cases']):
        from .context import live_context
        provider.phase = 'context-ablation'
        live_context(runtime, FOLDER)
    summary = {'rows': rows, 'not_run': [c['id'] for c in protocol['cases'] if c['id'] not in results],
               'budget': campaign.snapshot(), 'live_demo_passes': 0,
               'max_new_cases': max_new_cases, 'new_cases_executed': executed,
               'selected_case_ids': sorted(selected),
               'independent_source_review': 'PENDING'}
    # The summary is a projection; original results and receipts remain exclusive.
    (FOLDER / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'summary': str(FOLDER/'summary.json'), 'budget': campaign.snapshot()}, ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--freeze', action='store_true')
    action.add_argument('--freeze-continuation', action='store_true')
    action.add_argument('--execute-paid', action='store_true')
    parser.add_argument('--revision', type=int, help='Explicit protocol revision within the existing campaign')
    parser.add_argument('--continuation-note', help='Current human instruction authorizing a new frozen protocol')
    parser.add_argument('--carry-unknown', action='store_true',
                        help='Explicit human-authorized supplemental resume retaining UNKNOWN costs')
    parser.add_argument('--resume-json-rejection', action='store_true',
                        help='Retain one HTTP 400 reservation while repairing its missing JSON instruction')
    parser.add_argument('--extra-case-revalidation', help='One explicitly authorized extra execution of a frozen case')
    parser.add_argument('--max-new-cases', type=int, help='Execute at most this many new cases; keep all frozen questions')
    parser.add_argument('--case-ids', nargs='+', help='Schedule these frozen IDs; leave other cases not run')
    parser.add_argument('--campaign-dir', type=Path, help='Host-selected isolated campaign directory under outputs')
    parser.add_argument('--parent-campaign', type=Path, help='Read-only historical campaign for a newly authorized supplement')
    parser.add_argument('--remaining-from', type=Path, help='Unique grant of parent remainder, never independent extra budget')
    parser.add_argument('--supplemental-note', help='Current human instruction granting this new campaign budget')
    parser.add_argument('--capability-models', nargs='+',
                        help='Explicit two-model capability probes; answers retain the frozen primary')
    parser.add_argument('--authorized-model', help='Explicit authorized snapshot change in a new supplemental campaign')
    parser.add_argument('--max-requests', type=int)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--unlimited-usage', action='store_true',
                        help='Explicitly authorized new campaign without cumulative request/token ceilings')
    parser.add_argument('--request-cost-policy', choices=('legacy-utf8-v1', 'qwen3-vl-still-image-v1', 'qwen35-still-image-v1'),
                        help='Frozen input estimate; new Qwen3-VL campaigns default to the production image bound')
    parser.add_argument('--stage', choices=('answers', 'context', 'report', 'connectivity'))
    args = parser.parse_args()
    if args.request_cost_policy and not (args.freeze or args.freeze_continuation):
        parser.error('--request-cost-policy is fixed only when freezing a protocol')
    if args.max_requests is not None and not 1 <= args.max_requests <= 60:
        parser.error('--max-requests must be between 1 and 60')
    if args.max_tokens is not None and not 1 <= args.max_tokens <= 600000:
        parser.error('--max-tokens must be between 1 and 600000')
    global FOLDER, CAMPAIGN_FOLDER
    if args.campaign_dir is not None:
        selected_folder = args.campaign_dir.resolve()
        if not selected_folder.is_relative_to((ROOT/'outputs').resolve()) or selected_folder == (ROOT/'outputs').resolve():
            parser.error('--campaign-dir must be a dedicated directory under this checkout outputs')
        CAMPAIGN_FOLDER = FOLDER = selected_folder
    if args.parent_campaign and (not args.freeze or not args.campaign_dir or
                                args.parent_campaign.resolve() == CAMPAIGN_FOLDER):
        parser.error('--parent-campaign requires a new --freeze campaign directory')
    if args.remaining_from and (not args.freeze or not args.campaign_dir or args.parent_campaign or
                               not args.supplemental_note or args.max_requests or args.max_tokens):
        parser.error('--remaining-from requires new --freeze, campaign directory and instruction; limits derive from parent')
    if (args.max_requests is not None or args.max_tokens is not None) and not args.freeze:
        parser.error('Budget parameters are fixed only at --freeze')
    if args.unlimited_usage and (not args.freeze or not args.parent_campaign or
                                 not args.supplemental_note or args.max_requests is not None or
                                 args.max_tokens is not None or args.capability_models):
        parser.error('--unlimited-usage requires a new authorized --freeze and excludes finite limits')
    if args.parent_campaign and not args.unlimited_usage and (args.max_requests, args.max_tokens) != (30, 300000):
        parser.error('The current supplemental authorization is exactly 30 requests / 300000 tokens')
    if args.supplemental_note is not None and (not args.freeze or not (args.parent_campaign or args.remaining_from)):
        parser.error('--supplemental-note requires a new --freeze with --parent-campaign')
    if args.parent_campaign:
        try:
            supplemental_instruction(args.parent_campaign, args.supplemental_note)
        except ValueError as exc:
            parser.error(str(exc))
    if args.capability_models and (not args.freeze or not args.parent_campaign):
        parser.error('--capability-models requires a newly authorized --freeze with parent lineage')
    if args.authorized_model and (not args.freeze or not (args.parent_campaign or args.remaining_from) or
                                  args.capability_models or not args.supplemental_note):
        parser.error('--authorized-model requires a new supplement and explicit human instruction')
    if args.campaign_dir and args.freeze and not (args.parent_campaign or args.remaining_from):
        parser.error('A new campaign requires explicit parent lineage; do not reset old allowance')
    if args.revision is not None:
        if args.revision < 1:
            parser.error('--revision must be positive')
        FOLDER = CAMPAIGN_FOLDER / 'revisions' / f'{args.revision:04}'
    if args.freeze_continuation and args.revision is None:
        parser.error('--freeze-continuation requires --revision')
    if args.freeze and args.revision is not None:
        parser.error('A revision requires --freeze-continuation, never a fresh allowance')
    if args.continuation_note and not args.freeze_continuation:
        parser.error('--continuation-note requires --freeze-continuation')
    if args.carry_unknown and (not args.freeze_continuation or not args.continuation_note):
        parser.error('--carry-unknown requires --freeze-continuation and current --continuation-note')
    if args.resume_json_rejection and (not args.freeze_continuation or not args.continuation_note or args.carry_unknown):
        parser.error('--resume-json-rejection requires a documented source repair, never UNKNOWN carry')
    if args.extra_case_revalidation and (not args.freeze_continuation or not args.continuation_note):
        parser.error('--extra-case-revalidation requires a current explicit continuation instruction')
    if args.max_new_cases is not None and (not args.execute_paid or args.max_new_cases < 1):
        parser.error('--max-new-cases requires --execute-paid and a positive number')
    if args.case_ids is not None and not args.execute_paid:
        parser.error('--case-ids requires --execute-paid')
    if args.stage in {'answers', 'context'} and not args.execute_paid:
        parser.error('answers/context stages require --execute-paid')
    if args.stage == 'connectivity':
        if args.freeze or args.freeze_continuation or args.execute_paid:
            parser.error('connectivity is a zero-call diagnostic; do not combine with execution')
        from paperpilot_agentic.config import RuntimeSettings
        from .transport import transport_preflight
        FOLDER.mkdir(parents=True, exist_ok=True)
        result = transport_preflight(RuntimeSettings.from_env(require_api_key=False), FOLDER)
        print(json.dumps(result, ensure_ascii=True), flush=True)
        return
    if args.stage == 'report':
        from .campaign import report
        protocol = json.loads((FOLDER/'protocol.json').read_text(encoding='utf-8'))
        data = report(FOLDER, campaign_budget(protocol))
        print(json.dumps({'passes': data['original_demo_passes'], 'cost': data['aggregate_cost']}))
        return
    if args.freeze:
        freeze(parent_campaign=args.parent_campaign, max_requests=args.max_requests or 60,
               max_tokens=args.max_tokens or 600000, supplemental_note=args.supplemental_note,
               capability_models=args.capability_models, request_cost_policy=args.request_cost_policy,
               remaining_from=args.remaining_from, authorized_model=args.authorized_model,
               unlimited_usage=args.unlimited_usage)
    elif args.freeze_continuation:
        freeze(continuation=True, continuation_note=args.continuation_note, carry_unknown=args.carry_unknown,
               request_cost_policy=args.request_cost_policy, resume_json_rejection=args.resume_json_rejection,
               extra_case_revalidation=args.extra_case_revalidation)
    elif args.execute_paid:
        execute(max_new_cases=args.max_new_cases, case_ids=args.case_ids, stage=args.stage)
    else:
        print(json.dumps({'folder': str(FOLDER), 'cases': acceptance_cases(ROOT), 'actual_requests': 0}, ensure_ascii=True))


if __name__ == '__main__':
    main()
