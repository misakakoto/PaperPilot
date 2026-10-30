"""Source-bound numeric values, including decimals adjacent to Chinese text."""
from decimal import Decimal
import re

NUMBER = re.compile(r'(?<![A-Za-z0-9_.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?![0-9_]|\.(?=\d))')


def numeric_values(text):
    if not isinstance(text, str):
        raise ValueError('A numeric statement must be text')
    # F1 and ResNet18 remain identifiers; 93.5% and 为93.5% are values.
    # Formatting equivalence is allowed, arithmetic/units/entailment are not inferred.
    return {Decimal(m[0].replace(',', '').replace('−', '-')) for m in NUMBER.finditer(text)}
