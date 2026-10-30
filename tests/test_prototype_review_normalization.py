import copy

import pytest

from test_prototype_literature_review import reviewed, judgment
from test_prototype_literature import work


@pytest.mark.parametrize('case', ['supported_null', 'unsupported_null', 'legacy_null', 'foreign_quote'])
def test_only_explicit_supported_null_corrections_can_normalize(tmp_path, monkeypatch, case):
    _, h, rid, state, _ = reviewed(tmp_path, monkeypatch)
    state['review_correction_policy'] = 'supported-null-correction-v1'
    content = judgment(state, supported=case != 'unsupported_null')
    content['checks'][0]['required_correction'] = None
    if case == 'legacy_null':
        state.pop('review_correction_policy')
    elif case == 'foreign_quote':
        content['checks'][0]['quote_id'] = 'not-an-original-quote'
    before, raw = copy.deepcopy(state), copy.deepcopy(content)
    if case == 'supported_null':
        work(h, rid, state, 'claims', content)
        result = state['work']['artifacts'][state['work']['tasks']['claims']['output']]['content']
        assert result['passed'] is True and result['checks'][0]['required_correction'] == ''
        assert state['literature_review_normalizations'][0]['changes'][0]['from'] is None
        assert state['literature_review_normalizations'][0]['changes'][0]['to'] == ''
    else:
        with pytest.raises(ValueError):
            work(h, rid, state, 'claims', content)
        assert state == before
    assert content == raw
