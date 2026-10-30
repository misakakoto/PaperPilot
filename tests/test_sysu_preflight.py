from pathlib import Path

import pytest

from scripts.check_sysu_dataset import inspect_dataset


def fixture(tmp_path):
    for split, name in [('train', '1.png'), ('val', '2.png')]:
        for role in ('time1', 'time2', 'label'):
            folder = tmp_path / split / role
            folder.mkdir(parents=True)
            (folder / name).write_bytes(b'fixture-not-an-image')
        (tmp_path/f'{split}.txt').write_text(' '.join(
            f'/old/linux/path/SYSU-CD/{split}/{role}/{name}' for role in ('time1', 'time2', 'label')))
    return tmp_path


def test_linux_paths_are_rebased_without_modifying_lists_or_opening_test(tmp_path):
    root = fixture(tmp_path)
    before = (root/'train.txt').read_bytes()
    # No test directory exists: preflight must not require it.
    result = inspect_dataset(root)
    assert result['test_split_opened'] is False
    assert result['splits']['train']['count'] == 1
    assert Path(result['splits']['train']['samples'][0]['time1']) == root/'train/time1/1.png'
    assert (root/'train.txt').read_bytes() == before


@pytest.mark.parametrize('case', ['duplicate', 'wrong_role', 'traversal', 'missing', 'overlap'])
def test_invalid_or_leaking_split_is_rejected(tmp_path, case):
    root = fixture(tmp_path)
    path = root/'train.txt'
    original = path.read_text()
    if case == 'duplicate':
        path.write_text(original+'\n'+original)
    elif case == 'wrong_role':
        path.write_text(original.replace('/train/time2/', '/val/time2/'))
    elif case == 'traversal':
        path.write_text(original.replace('/old/', '/old/../'))
    elif case == 'missing':
        (root/'train/label/1.png').unlink()
    else:
        (root/'val.txt').write_text((root/'val.txt').read_text().replace('2.png', '1.png'))
    with pytest.raises(ValueError):
        inspect_dataset(root)
