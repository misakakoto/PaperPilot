"""Exactly four frozen text/image JSON probes, all on the answer campaign ledger."""
import base64
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import time

from .budget import RecordedProvider, save
from paperpilot_agentic.model_policy import PRIMARY_MODEL, BACKUP_MODEL


def inherit_preflight(parent, identity, folder, settings):
    """Reuse passed capability evidence after a code repair, never invent a probe."""
    if (parent['identity'].get('capability_preflight_models') != [PRIMARY_MODEL, BACKUP_MODEL]
            or (settings.model, settings.provider) != (parent['identity']['model'], parent['identity']['provider'])
            or parent['base_url_sha256'] != hashlib.sha256(settings.base_url.encode()).hexdigest()
            or (parent['temperature'], parent['thinking_enabled']) !=
               (settings.temperature, settings.thinking_enabled)):
        raise ValueError('Preflight provider configuration changed')
    path = Path(folder)/'preflight-result.json'
    old = json.loads(path.read_text(encoding='utf-8'))
    rows = old.get('capability_results', [])
    expected = [(model, kind) for model in (PRIMARY_MODEL, BACKUP_MODEL) for kind in ('text', 'image')]
    if (old.get('ready') is not True or old.get('campaign_identity') != parent['identity']
            or len(rows) != 4 or any((row.get('model'), row.get('kind')) != pair
                or row.get('ready') is not True or row.get('finish_reason') != 'stop'
                or row.get('parsed_json') != ({'ready': True} if pair[1] == 'text' else {'color': 'red'})
                for row, pair in zip(rows, expected, strict=True))):
        raise ValueError('All four bound capability probes must have passed')
    return {**old, 'campaign_identity': identity,
            'reused_preflight': {'result_path': str(path.resolve()),
                'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'parent_identity': parent['identity'], 'new_sdk_requests': 0}}


def capability_preflight(settings, campaign, folder, *, factory=None):
    from PIL import Image, ImageDraw
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    folder = Path(folder)
    final_path = folder/'preflight-result.json'
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding='utf-8'))
        if final.get('campaign_identity') != json.loads(campaign.identity):
            raise ValueError('Preflight identity changed')
        return final['ready']
    identity = json.loads(campaign.identity)
    models = identity.get('capability_preflight_models')
    if models != [PRIMARY_MODEL, BACKUP_MODEL] or settings.model != PRIMARY_MODEL:
        raise ValueError('Capability probes require the explicitly frozen model pair')
    image_path = folder/'preflight-red-square.png'
    image = Image.new('RGB', (64, 64), 'white')
    ImageDraw.Draw(image).rectangle((8, 8, 55, 55), fill='red')
    encoded = io.BytesIO()
    image.save(encoded, format='PNG')
    image_bytes = encoded.getvalue()
    if image_path.exists() and image_path.read_bytes() != image_bytes:
        raise ValueError('Frozen preflight image changed')
    if not image_path.exists():
        with image_path.open('xb') as stream:
            stream.write(image_bytes)
    expected_image_sha = hashlib.sha256(image_bytes).hexdigest()
    url = 'data:image/png;base64,' + base64.b64encode(image_bytes).decode('ascii')
    prompts = {
        'text': [{'role': 'user', 'content': '仅输出 JSON 对象 {"ready": true}。'}],
        'image': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': url}},
            {'type': 'text', 'text': '识别图片中央方形的填充颜色。仅输出 JSON 对象，color 为英文颜色名称。'}]}],
    }
    rows = []
    ready = True
    for model in models:
        for kind, messages in prompts.items():
            result_path = folder/f'capability-{model}-{kind}.json'
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding='utf-8'))
                if result.get('campaign_identity') != identity or (result['model'], result['kind']) != (model, kind):
                    raise ValueError('Capability receipt identity changed')
            else:
                if campaign.snapshot()['blocked']:
                    ready = False
                    break
                config = replace(settings, model=model, fallback_models=(), thinking_enabled=False)
                provider = RecordedProvider((factory or (lambda s: OpenAICompatibleLLM(s, max_retries=0)))(config),
                                            campaign, phase='capability:' + kind)
                start = time.monotonic()
                try:
                    prior = [row for row in campaign.snapshot()['attempts']
                             if row['model'] == model and row['phase'] == 'capability:' + kind]
                    if prior:
                        if len(prior) != 1 or prior[0]['state'] != 'RETURNED':
                            raise ValueError('Interrupted preflight cannot be resent')
                        number = prior[0]['id']
                        request = json.loads((campaign.folder/f'request-{number:03}.json').read_text(encoding='utf-8'))
                        if request != {'model': model, 'phase': 'capability:' + kind,
                                       'protocol_identity': identity, 'messages': messages, 'max_tokens': 128,
                                       'json_mode': True, 'thinking_enabled': False}:
                            raise ValueError('Retained preflight request changed')
                        raw = json.loads((campaign.folder/f'response-{number:03}.json').read_text(encoding='utf-8'))
                        choice = raw.get('choices', [{}])[0]
                        response_text = raw.get('text', choice.get('message', {}).get('content'))
                        finish = raw.get('finish_reason', choice.get('finish_reason'))
                    else:
                        response = provider.complete(messages, max_tokens=128, json_mode=True, thinking_enabled=False)
                        response_text, finish = response.text, response.finish_reason
                    value = json.loads(response_text)
                    passed = finish == 'stop' and value == (
                        {'ready': True} if kind == 'text' else {'color': 'red'})
                    result = {'model': model, 'kind': kind, 'ready': passed,
                        'finish_reason': finish, 'parsed_json': value,
                        'image_sha256': expected_image_sha if kind == 'image' else None}
                except Exception as exc:
                    result = {'model': model, 'kind': kind, 'ready': False, 'error_type': type(exc).__name__}
                result['elapsed_seconds'] = round(time.monotonic() - start, 3)
                result['campaign_identity'] = identity
                save(result_path, result)
            rows.append(result)
            print(json.dumps(result, ensure_ascii=True), flush=True)
            if not result['ready']:
                ready = False
                break
        if not ready:
            break
    ready = ready and len(rows) == 4
    save(final_path, {'ready': ready, 'campaign_identity': identity, 'capability_results': rows,
                      'actual_requests': campaign.snapshot()['actual_requests']})
    if not ready and not (campaign.folder/'stopped.json').exists():
        save(campaign.folder/'stopped.json', {'reason': 'capability_preflight_failed'})
    return ready
