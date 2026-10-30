"""Host-rendered scientific labels must survive partial-fact language checks."""
from copy import deepcopy
import re
import pytest
from paperpilot_agentic.answer_contracts import validate_draft
from paperpilot_agentic.table_projection import project_table_facts
from test_table_fact_projection import rows, draft


def test_chinese_projection_can_be_the_only_surviving_auditable_fact():
    catalog, selection = rows(), draft()
    before = deepcopy((catalog, selection))
    projected, trace = project_table_facts(selection, catalog, chinese=True)
    assert re.search(r'[\u4e00-\u9fff]', projected['facts'][0]['text'])
    validate_draft(projected, [{'evidence_id':'t','content':'Original Table IV.'}],
                   question='报告该配置的F1指标。', table_rows=catalog)
    assert projected['facts'][0]['row_refs'] == selection['facts'][0]['row_refs']
    assert trace[0]['raw_selection'] == selection['facts'][0]
    assert (catalog, selection) == before


def test_english_host_projection_stays_english_and_untranslated_model_prose_is_rejected():
    projected, _ = project_table_facts(draft(), rows(), chinese=False)
    assert not re.search(r'[\u4e00-\u9fff]', projected['facts'][0]['text'])
    value = {'facts':[{'id':'a','kind':'SOURCE_FACT','text':'Alpha reshapes tokens.',
                       'evidence_ids':['e']}], 'direct_answer_id':'a','limitations':[]}
    with pytest.raises(ValueError, match='language must match'):
        validate_draft(value, [{'evidence_id':'e','content':'Alpha reshapes tokens.'}], question='请解释机制。')
