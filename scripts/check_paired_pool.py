"""Synthetic global-pool forward/backward parity including tied maxima; no training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paperpilot_agentic.paired_snunet_worker import global_max_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    import torch
    torch.use_deterministic_algorithms(True)
    original = (torch.arange(96) % 7).reshape(2, 3, 4, 4).float()
    weights = torch.arange(1, 7).reshape(2, 3, 1, 1).float()
    oracle = original.double().requires_grad_()
    expected = torch.nn.functional.adaptive_max_pool2d(oracle, 1)
    (expected*weights).sum().backward()
    value = original.to(args.device).requires_grad_()
    actual = global_max_pool(value)
    (actual*weights.to(args.device)).sum().backward()
    forward_error = (actual.cpu().double()-expected).abs().max().item()
    gradient_error = (value.grad.cpu().double()-oracle.grad).abs().max().item()
    assert forward_error == 0 and gradient_error == 0
    print(json.dumps({'status': 'POOL_AND_GRADIENT_VERIFIED', 'device': args.device,
        'torch': torch.__version__, 'forward_max_abs_error': forward_error,
        'gradient_max_abs_error': gradient_error, 'includes_tied_maxima': True,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}))


if __name__ == '__main__':
    main()
