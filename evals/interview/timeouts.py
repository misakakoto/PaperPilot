"""Freeze transport deadlines without changing legacy execution or retry policy."""
import math

VERSION = 'sdk-deadlines-v1'
LEGACY_SECONDS = 60


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Request timeout must be a finite positive number')
    return value


def request_timeouts(protocol, configured):
    """None means a new freeze; an existing empty protocol means legacy."""
    if protocol is None:
        seconds = _seconds(configured)
        return {'version': VERSION, 'text_seconds': seconds, 'vision_seconds': seconds}
    bound = protocol.get('request_timeout')
    identity_bound = protocol.get('identity', {}).get('request_timeout')
    if bound is not None or identity_bound is not None:
        if (not isinstance(bound, dict) or bound != identity_bound or
                set(bound) != {'version', 'text_seconds', 'vision_seconds'} or
                bound['version'] != VERSION or
                protocol.get('timeout_seconds') != bound['text_seconds']):
            raise ValueError('Frozen request timeout identity changed')
        _seconds(bound['text_seconds'])
        _seconds(bound['vision_seconds'])
        return dict(bound)
    # Historical text probes could explicitly use another deadline, while
    # their auto-vision adapter always used 60. Preserve both on a revision.
    return {'version': VERSION,
            'text_seconds': _seconds(protocol.get('timeout_seconds', LEGACY_SECONDS)),
            'vision_seconds': LEGACY_SECONDS}
