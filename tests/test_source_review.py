import copy
import json

import pytest

from evals.source_review import digest, encode, prepare, read, score, score_review, verified_packet


def save(path, value):
    path.write_bytes(encode(value))


@pytest.fixture
def packet(tmp_path):
    source, destination = tmp_path / 'source', tmp_path / 'packet'
    source.mkdir()
    cases = [{'id': str(i), 'question': f'Question {i}', 'review': 'Check original source'}
             for i in range(1, 5)]
    save(source / 'frozen-cases.json', cases)
    for i, status in ((1, 'COMPLETED'), (2, 'COMPLETED'), (3, 'FAILED')):
        save(source / f'result-{i}.json', {
            'run': {'question': f'Question {i}', 'status': status},
            'final_answer': {'answer': f'Unchanged answer {i}', 'audit': {'verdict': 'SUPPORTED'}},
            'evidence': [{'evidence_id': 'e1', 'content': 'Original source',
                          'source_uri': 'paper://sha', 'confidence': 1.0}]})
    prepare(source, destination, protocol='fixture', scope='synthetic tests, not model evaluation')
    return source, destination


def review(folder, identity='reviewer-a', kind='human'):
    value = read(folder / 'review-template.json')
    value.update(reviewer_id=identity, reviewer_kind=kind)
    return value


def grade(value, index, verdict):
    value['cases'][index].update(verdict=verdict, reason='Checked the specified source',
        source_checks=[{'source_uri': 'paper://sha', 'locator': 'page 1 paragraph 2',
                        'note': 'Matched the original statement'}])


def test_packet_preserves_answers_and_omits_model_judgement(packet):
    source, folder = packet
    value, _ = verified_packet(folder)
    assert len(value['cases']) == 4
    assert value['cases'][0]['original_answer'] == 'Unchanged answer 1'
    assert value['cases'][-1]['run_status'] == 'MISSING'
    assert 'SUPPORTED' not in json.dumps(value)
    assert 'confidence' not in json.dumps(value)
    assert (folder / 'result-1.json').read_bytes() == (source / 'result-1.json').read_bytes()


def test_pending_and_missing_are_not_removed_from_denominator(packet):
    _, folder = packet
    labels = review(folder)
    grade(labels, 0, 'PASS')
    path = folder / 'review-a.json'
    save(path, labels)
    report = score(folder, [path])['reviews'][0]
    assert report['count'] == 4
    assert report['counts']['OPERATIONAL_FAILURE'] == 1
    assert report['counts']['UNRESOLVED_EXECUTION'] == 1
    assert report['strict_pass_rate'] is None
    assert report['pass_rate_lower_bound'] == .25
    assert report['pass_rate_upper_bound'] == .75


def test_fully_reviewed_synthetic_cohort_keeps_failure_in_denominator(packet):
    _, folder = packet
    # A synthetic three-case cohort tests arithmetic, not real model quality.
    value, packet_hash = verified_packet(folder)
    value['cases'] = value['cases'][:3]
    labels = review(folder)
    labels['cases'] = labels['cases'][:3]
    grade(labels, 0, 'PASS')
    grade(labels, 1, 'PARTIAL')
    report = score_review(value, packet_hash, labels)
    assert report['review_complete'] is True
    assert report['strict_pass_rate'] == pytest.approx(1 / 3)
    assert report['counts']['OPERATIONAL_FAILURE'] == 1


@pytest.mark.parametrize('mutation', ['omit', 'duplicate', 'binding', 'empty_reason', 'no_source', 'kind', 'failed_pass'])
def test_invalid_or_selective_review_rejected(packet, mutation):
    _, folder = packet
    value = review(folder)
    grade(value, 0, 'PASS')
    if mutation == 'omit':
        value['cases'].pop()
    elif mutation == 'duplicate':
        value['cases'][-1] = copy.deepcopy(value['cases'][0])
    elif mutation == 'binding':
        value['cases'][0]['binding'] = 'different-answer'
    elif mutation == 'empty_reason':
        value['cases'][0]['reason'] = ''
    elif mutation == 'no_source':
        value['cases'][0]['source_checks'] = []
    elif mutation == 'kind':
        value['reviewer_kind'] = 'UNSET'
    else:
        grade(value, 2, 'PASS')
    path = folder / 'bad.json'
    save(path, value)
    with pytest.raises(ValueError):
        score(folder, [path])


def test_two_reviewers_disagreements_not_averaged_away(packet):
    _, folder = packet
    a, b = review(folder), review(folder, 'reviewer-b', 'ai')
    grade(a, 0, 'PASS')
    grade(b, 0, 'FAIL')
    grade(a, 1, 'PARTIAL')
    grade(b, 1, 'PARTIAL')
    paths = [folder / 'a.json', folder / 'b.json']
    for path, value in zip(paths, (a, b)):
        save(path, value)
    report = score(folder, paths)
    assert report['agreement']['rate'] == .5
    assert report['agreement']['disagreements'] == ['1']
    assert report['reviews'][1]['reviewer_kind'] == 'ai'
    b['reviewer_id'] = a['reviewer_id']
    save(paths[1], b)
    with pytest.raises(ValueError):
        score(folder, paths)


@pytest.mark.parametrize('target', ['packet.json', 'result-1.json'])
def test_changed_snapshot_rejected(packet, target):
    _, folder = packet
    (folder / target).write_bytes(b'{}')
    with pytest.raises(ValueError):
        verified_packet(folder)


def test_changed_evidence_rejected_even_when_packet_hash_updated(packet):
    _, folder = packet
    value = read(folder / 'packet.json')
    value['cases'][0]['evidence'][0]['content'] = 'Favorable rewritten source'
    save(folder / 'packet.json', value)
    manifest = read(folder / 'manifest.json')
    manifest['packet_sha256'] = digest((folder / 'packet.json').read_bytes())
    save(folder / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='evidence'):
        verified_packet(folder)


def test_changed_question_and_existing_destination_rejected(packet, tmp_path):
    source, folder = packet
    with pytest.raises(FileExistsError):
        prepare(source, folder, protocol='test', scope='test')
    result = read(source / 'result-1.json')
    result['run']['question'] = 'Easier question'
    save(source / 'result-1.json', result)
    with pytest.raises(ValueError, match='question'):
        prepare(source, tmp_path / 'new', protocol='test', scope='test')
    assert not (tmp_path / 'new').exists()
