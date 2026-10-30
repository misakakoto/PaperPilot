from pathlib import Path
import importlib.util
import subprocess


def checker():
    path = Path(__file__).resolve().parents[1]/'scripts/check_release_contents.py'
    spec = importlib.util.spec_from_file_location('release_checker',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root,*args):
    return subprocess.check_output(['git','-c',f'safe.directory={root.as_posix()}',
        '-C',str(root),*args],stderr=subprocess.STDOUT)


def test_ignored_but_tracked_private_file_is_rejected(tmp_path):
    git(tmp_path,'init','-q')
    (tmp_path/'.gitignore').write_text('.env\n')
    (tmp_path/'.env').write_text('SYNTHETIC_FIXTURE=true\n')
    git(tmp_path,'add','-f','.env')
    result = checker().scan(tmp_path)
    assert not result['passed']
    assert {'file':'.env','kind':'private_asset'} in result['findings']


def test_secret_pattern_is_rejected_without_returning_value(tmp_path):
    git(tmp_path,'init','-q')
    value = 'sk-'+'A'*24
    (tmp_path/'accidental.txt').write_text(value)
    result = checker().scan(tmp_path)
    assert not result['passed']
    assert any(r['kind']=='provider_token' for r in result['findings'])
    assert value not in str(result)


def test_empty_template_and_test_fixture_are_allowed(tmp_path):
    git(tmp_path,'init','-q')
    (tmp_path/'.env.example').write_text('DASHSCOPE_API_KEY=\n')
    (tmp_path/'fixture.py').write_text('api_key="not-a-real-network-key"\n')
    assert checker().scan(tmp_path)['passed']
