"""Counted combined probe; one backup only for a confirmed availability error."""
import base64
from dataclasses import replace
import io
import json
from pathlib import Path
from .budget import RecordedProvider, save
from paperpilot_agentic.model_policy import PRIMARY_MODEL, BACKUP_MODEL
from paperpilot_agentic.providers import OpenAICompatibleLLM


def selection_preflight(settings, campaign, folder, *, factory=None):
    from PIL import Image
    folder = Path(folder)
    prior_result = folder/'preflight-result.json'
    if prior_result.exists():
        saved = json.loads(prior_result.read_text(encoding='utf-8'))
        if saved.get('campaign_identity') != json.loads(campaign.identity):
            raise ValueError('Preflight identity changed')
        return saved['ready']
    image = io.BytesIO()
    Image.new('RGB', (64, 64), 'red').save(image, format='PNG')
    url = 'data:image/png;base64,' + base64.b64encode(image.getvalue()).decode()
    messages = [{'role': 'system', 'content': 'Return only JSON. Identify the image color.'},
                {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}},
                                           {'type': 'text', 'text': 'Return {"color":"red"} if red, otherwise its actual English color.'}]}]
    identity = json.loads(campaign.identity)
    models = identity.get('selection_models', [PRIMARY_MODEL, BACKUP_MODEL])
    for model in models:
        previous = [r for r in campaign.snapshot()['attempts'] if r['model'] == model and r['phase'] == 'capability:selection']
        if previous:
            if previous[0]['state'] == 'RETURNED':
                receipt = json.loads((campaign.folder/f"response-{previous[0]['id']:03}.json").read_text(encoding='utf-8'))
                choices = receipt.get('choices') or [{}]
                text = receipt.get('text') or choices[0].get('message', {}).get('content', '')
                finish = receipt.get('finish_reason') or choices[0].get('finish_reason')
            elif len(models) == 2 and model == PRIMARY_MODEL and campaign._switchable(previous[0]):
                continue
            else:
                break  # Interrupted/UNKNOWN is never resent.
        else:
            provider = RecordedProvider((factory or (lambda s: OpenAICompatibleLLM(s, max_retries=0)))(
                replace(settings, model=model, fallback_models=(), thinking_enabled=False)), campaign, phase='capability:selection')
            try:
                response = provider.complete(messages, max_tokens=128, json_mode=True, thinking_enabled=False)
                text, finish = response.text, response.finish_reason
            except Exception:
                rows = campaign.snapshot()['attempts']
                if len(models) == 2 and model == PRIMARY_MODEL and len(rows) == 1 and campaign._switchable(rows[0]):
                    continue
                break
        try:
            ready = finish == 'stop' and json.loads(text) == {'color': 'red'}
        except (ValueError, TypeError):
            ready = False
        if ready:
            selected = {'model': model, 'identity': identity, 'phase': 'capability:selection'}
            if not (folder/'selected-model.json').exists():
                save(folder/'selected-model.json', selected)
            save(folder/'preflight-result.json', {'ready': True, 'model': model, 'campaign_identity': identity,
                                                'budget': campaign.snapshot()})
            return True
        break
    save(folder/'preflight-result.json', {'ready': False, 'campaign_identity': identity, 'budget': campaign.snapshot()})
    if not (campaign.folder/'stopped.json').exists():
        save(campaign.folder/'stopped.json', {'reason': 'model_preflight_not_ready'})
    return False
