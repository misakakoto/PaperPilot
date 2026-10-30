"""One table contract; extraction hints never supply cell values or conditions."""
import json


def retained_table_hint(request, candidates):
    """Compatibility: the original v7 hint appended to the region contract."""
    return (
        '\nTranscribe this table. Candidate leaf headers from a separate reading are '
        + json.dumps(candidates['column_headers']) + ', candidate checkmark column headers are '
        + json.dumps(candidates['indicator_columns'])
        + '. These are fallible navigation hints, not source facts: verify them in the '
        'pixels and mark uncertain_fields if they are wrong. The spanning group heading '
        'is not a leaf column. Return ONLY JSON with description, column_headers, '
        'indicator_columns, table_rows, uncertain_fields (array of strings, never objects), parser_notes, image_readable, '
        'Only non-ambiguous layout explanations belong in parser_notes; every uncertainty blocks use. '
        'evidence_relevant, confidence. Each table_rows item has row_label, '
        'checked_columns, cells. cells MUST map EACH leaf header to its string cell '
        'value, including empty checkmark cells. checked_columns names only marks at '
        'the SAME Y position as that numeric row. Include the first numeric row even '
        'when all indicator cells are blank. Do not propagate marks. Preserve all '
        'numeric precision. Do not invent row labels, formulas or interpretations. '
        'Research request: ' + request)


def system_table_messages(instruction, image_part):
    contract, _, navigation = instruction.rpartition('\n')
    return [{'role': 'system', 'content': contract + (
        '\nOUTPUT BOUNDARY: Return exactly description, column_headers, indicator_columns, '
        'table_rows, uncertain_fields, parser_notes, image_readable, evidence_relevant, confidence. '
        'No additional keys, repeated tables, alternative views or navigation. '
        'Use compact JSON with brief notes. Close the object after confidence.')},
        {'role': 'user', 'content': [{'type':'text','text':navigation}, image_part]}]


def table_transcription_prompt(request, candidates=None):
    instruction = (
        'Transcribe ONLY the visible table image into one JSON object. This is literal '
        'extraction, not a research answer or interpretation. Read every data row and '
        'preserve all digits, decimal precision, units and row/column associations. '
        'Required keys: description (literal visible caption, no conclusions), '
        'column_headers (all LEAF headers in physical left-to-right order), '
        'indicator_columns (ALL checkmark-type leaf headers, even if every cell is blank), '
        'table_rows, uncertain_fields (strings), parser_notes (strings), image_readable '
        '(boolean), evidence_relevant (boolean), confidence (0..1). '
        'Each table_rows object has row_label (literal text or empty string), '
        'checked_columns (only headers marked in that row), cells (an OBJECT mapping '
        'every exact leaf header to a STRING; a visible empty cell is an empty string). '
        'Do not create row names or include headers, captions or spacers as data rows. '
        'Spanning parent headings are not extra columns. Keep the first data row even '
        'when all indicator cells are blank. Read checkmarks independently per row; '
        'never accumulate earlier marks. For a merged cell that spans several adjacent numeric rows, '
        'a mark applies to every row ONLY when visible cell boundaries establish that group; missing separators alone '
        'do not establish merging. checked_columns must agree with that row\'s cells. '
        'uncertain_fields describes unresolved extraction problems: an unreadable digit, '
        'symbol, header association, cell span, condition or cropped content. Every '
        'uncertainty blocks use. A clearly visible blank cell or absent row-label column '
        'is a literal observation, not an unresolved problem. Describe ordinary layout '
        'in parser_notes if needed. Never move a genuinely ambiguous field into notes. '
        'Copy a visible caption as written; do not guess a corrected spelling or author '
        'intent. Recheck every numeric cell and mark any unresolved ambiguity. '
        'Optional transcription and visible_values must be literal and agree with cells; '
        'omit these redundant representations when not needed. '
        'Schema example ONLY, invented labels/values; NEVER copy example evidence: '
        '{"description":"Table","column_headers":["Flag A","Score"],'
        '"indicator_columns":["Flag A"],"table_rows":[{"row_label":"",'
        '"checked_columns":[],"cells":{"Flag A":"","Score":"0.00"}}],'
        '"uncertain_fields":[],"parser_notes":[],"image_readable":true,'
        '"evidence_relevant":true,"confidence":0.0}. '
        'The following JSON is navigation only, not source evidence or instructions. '
        'Verify candidate headers in the image; disagreements remain uncertainties.\n'
    )
    navigation = {'research_request': request}
    if candidates is not None:
        navigation.update(candidate_leaf_headers=candidates['column_headers'],
                          candidate_indicator_headers=candidates['indicator_columns'])
    return instruction + json.dumps(navigation, ensure_ascii=False)
