"""Check the current Git candidate set; never print matched credential values."""
import argparse
import json
from pathlib import Path
import re
import subprocess


SECRET_PATTERNS = {
    'provider_token': re.compile(r'(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}'),
    'github_token': re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})'),
    'aws_access_key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'developer_absolute_path': re.compile(r'(?i)[A-Z]:[/\\](?:Users|work|soft)[/\\]'),
}
CREDENTIAL = re.compile(
    r'''(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)["']?\s*[=:]\s*["']([^"'\n]{12,})["']''')
SYNTHETIC = re.compile(r'(?i)unused|not-a-real|fake|dummy|test|offline')
PRIVATE_SUFFIXES = {'.pdf','.sqlite','.sqlite3','.db','.pem','.key','.p12','.pfx',
    '.pt','.pth','.ckpt','.onnx','.safetensors','.npy','.npz','.pkl','.pickle','.har','.log',
    '.zip','.tar','.gz','.7z'}
PRIVATE_DIRS = {'.paperpilot','outputs','papers','private','local-data','model-cache',
    'agent_output','scholarly-cache','__pycache__','.venv','.pytest_cache','.ruff_cache'}


def scan(root):
    raw = subprocess.check_output(['git','-c',f'safe.directory={root.as_posix()}',
        '-C',str(root),'ls-files','--cached','--others','--exclude-standard','-z'])
    names = sorted(set(filter(None,raw.decode('utf-8').split('\0'))))
    findings = []
    for name in names:
        relative = Path(name)
        path = root / relative
        if (PRIVATE_DIRS & set(relative.parts) or relative.suffix.lower() in PRIVATE_SUFFIXES
                or relative.name == '.env' or relative.name.startswith('.env.') and relative.name != '.env.example'):
            findings.append({'file':name,'kind':'private_asset'})
            continue
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            findings.append({'file':name,'kind':'nonregular_or_external_path'})
            continue
        try:
            content = path.read_bytes().decode('utf-8-sig')
        except UnicodeDecodeError:
            continue  # Font/image bytes require separate provenance/visual review.
        for number,line in enumerate(content.splitlines(),1):
            for kind,pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    findings.append({'file':name,'line':number,'kind':kind})
            for match in CREDENTIAL.finditer(line):
                if not SYNTHETIC.search(match[1]):
                    findings.append({'file':name,'line':number,'kind':'review_credential_literal'})
    return {'files_checked':len(names),'findings':findings,'passed':not findings,
        'scope':'current tracked/nonignored files only; no Git history or binary-content audit'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = scan(args.root.resolve())
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
