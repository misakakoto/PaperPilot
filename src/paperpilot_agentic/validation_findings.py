"""Observe existing audit transitions without changing their decisions."""
from copy import deepcopy
import importlib
from pathlib import Path

from .reference_methods import sha

VERSION = 'audit-transition-observation-v1'


class AuditTrail:
    def __init__(self, audit, draft):
        self.previous = deepcopy(audit)
        self.facts = {f['id']: f for f in draft['facts']}
        self.findings = []

    def capture(self, validator, audit):
        module = importlib.import_module(validator.__module__)
        identity = sha(Path(module.__file__).read_bytes())
        old = {a['fact_id']: a for a in self.previous['assessments']}
        for assessment in audit['assessments']:
            fid = assessment['fact_id']
            before = old.get(fid)
            if before != assessment:
                self.findings.append({'version': VERSION, 'fact_id': fid,
                    'validator': validator.__module__ + '.' + validator.__name__,
                    'validator_version': getattr(module, 'VERSION', identity),
                    'validator_source_sha256': identity,
                    'evidence_ids': self.facts.get(fid, {}).get('evidence_ids', []),
                    'origin': 'HOST_RULE_TRANSITION_NOT_GROUND_TRUTH',
                    'before': deepcopy(before), 'after': deepcopy(assessment),
                    'action': 'WITHHOLD' if assessment['verdict'] != 'SUPPORTED' else 'RETAIN'})
        self.previous = deepcopy(audit)
