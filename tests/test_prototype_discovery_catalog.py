import copy

import pytest

from paperpilot_agentic.prototype_literature import project_artifact, execute, POLICY


def test_excerpt_budget_does_not_hide_other_discovered_papers():
    artifact = {'kind': 'search_literature', 'content': {'evidence': [
        {'content': 'Long fragment '*300, 'page': i+1, 'source_uri': f'paper://{i:064x}/chunk',
         'provenance': {'paper_sha256': f'{i:064x}', 'title': f'Paper {i}'}}
        for i in range(6)]}}
    before = copy.deepcopy(artifact)
    view = project_artifact('source', artifact, max_characters=500)
    catalog = view['content']['discovery_catalog']
    assert len(catalog) == 6 and len(view['content']['evidence']) == 1
    assert {x['paper_sha256'] for x in catalog} == {f'{i:064x}' for i in range(6)}
    assert artifact == before


def test_narrative_read_error_explains_read_and_inspection_actions():
    with pytest.raises(ValueError, match='inspect_artifact'):
        execute(None, 'run', {'literature_policy': POLICY}, {'capability': 'read_literature'},
                'I am inspecting the result now')
