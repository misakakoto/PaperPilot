"""Dataset reference tokens have no implicit mapping to printed PDF labels."""
import re


PLACEHOLDER = re.compile(r'\b(?:TABREF|FIGREF|FLOATREF)\d+\b', re.I)
REFERENCE_IDENTITY_POLICY = (
    'Source tokens such as TABREF35, FIGREF12 or FLOATREF7 are unresolved reference '
    'identifiers, NOT printed table/figure numbers. Never turn their numeric suffix '
    'into Table 35, a page number, a layout object ID, or a result. Preserve the '
    'literal token when mentioning it. reference_placeholders lists only tokens '
    'in the supplied evidence window; it does not resolve them. Without an explicit '
    'source mapping, target a missing aspect by its task/metric/section or state that '
    'the referenced table is unavailable. A discussion of a result is not the table '
    'itself, and a plan to read a table is not a completed read.'
)


def reference_placeholders(rows):
    return [{'evidence_id': row['evidence_id'], 'tokens': tokens}
            for row in rows
            if (tokens := list(dict.fromkeys(PLACEHOLDER.findall(row.get('content', '')))))]
