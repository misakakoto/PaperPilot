"""Bounded probe of an existing training interpreter; installs and trains nothing."""
import argparse
import json
import subprocess


CODE = '''import json
import torch
print(json.dumps({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
                  'torch_cuda': torch.version.cuda}), flush=True)
if torch.cuda.is_available():
    x = torch.ones(2, device='cuda')
    print(json.dumps({'gpu_sum': x.sum().item(), 'device': torch.cuda.get_device_name()}), flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    try:
        result = subprocess.run([args.python, '-I', '-u', '-c', CODE], capture_output=True,
                                text=True, timeout=40)
    except subprocess.TimeoutExpired:
        print(json.dumps({'status': 'TIMEOUT', 'timeout_seconds': 40}))
        raise SystemExit(1)
    print(json.dumps({'exit_code': result.returncode, 'stdout': result.stdout,
                      'stderr': result.stderr}, ensure_ascii=True))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
