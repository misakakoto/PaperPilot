"""Lossless multi-tier header normalization; never rewrite data cells."""
import copy


def header_candidates(value: dict) -> dict:
    arrays = {}
    for key in ('column_headers', 'indicator_columns', 'group_headers', 'uncertain_fields'):
        items = value.get(key)
        if (not isinstance(items, list) or any(not isinstance(x, str) or not x.strip() for x in items)
                or len(items) != len(set(items))):
            raise ValueError('Invalid header-only extraction')
        arrays[key] = items
    if arrays['uncertain_fields']:
        raise ValueError('Uncertain header-only extraction')
    notes = value.get('parser_notes', [])
    if (not isinstance(notes, list) or len(notes) > 100 or
            any(not isinstance(x, str) or not x.strip() or len(x) > 1000 for x in notes)):
        raise ValueError('Invalid header parser notes')
    if 'parser_notes' in value:
        arrays['parser_notes'] = list(notes)
    leaves = [h for h in arrays['column_headers'] if h not in arrays['group_headers']]
    if not leaves or not set(arrays['indicator_columns']).issubset(leaves):
        raise ValueError('Invalid leaf/indicator header roles')
    return {**arrays, 'column_headers': leaves}


def normalize_header_tiers(value: dict, candidates: dict) -> tuple[dict, list[dict]]:
    """Use a separately retained header read only when every row has exact keys.

    Parent headings cannot have data. Do not discard populated group cells,
    reorder a conflicting metric, map unknown keys, or invent omitted values.
    Pixel checks and source review still decide whether cells are correct.
    """
    result = copy.deepcopy(value)
    leaves = candidates['column_headers']
    parents = candidates['group_headers']
    rows = result.get('table_rows')
    old = result.get('column_headers')
    indicators = result.get('indicator_columns')
    if (not isinstance(rows, list) or not rows or not isinstance(old, list)
            or any(not isinstance(h, str) for h in old) or len(old) != len(set(old))
            or not set(old).issubset(set(leaves) | set(parents))
            or [h for h in old if h not in parents] != [h for h in leaves if h in old]
            or indicators != candidates['indicator_columns']):
        return value, []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('cells'), dict):
            return value, []
        cells = row['cells']
        # Public rows may keep checkmarks solely in checked_columns; they need
        # not duplicate the same flags as strings inside numeric cells.
        if ((set(cells)-set(parents)) | set(indicators) != set(leaves)
                or any(cells[p] != '' for p in parents if p in cells)):
            return value, []
    changes = []
    if old != leaves:
        result['column_headers'] = list(leaves)
        changes.append({'operation': 'header_tiers_to_retained_leaf_order', 'original_headers': old})
    for i, row in enumerate(rows):
        for parent in parents:
            if parent in row['cells']:
                del row['cells'][parent]
                changes.append({'operation': 'remove_empty_parent_cell', 'row_index': i, 'header': parent})
    return result, changes
