"""Tiny loss/gradient contract check against CPU double-precision PyTorch CE.

Uses synthetic tensors only; no data, training, downloads or paid API calls.
Run with the selected training interpreter, with --device cpu or cuda.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paperpilot_agentic.paired_snunet_worker import classification_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.use_deterministic_algorithms(True)
    target = (torch.arange(16) % 2).reshape(1, 4, 4)
    original = torch.linspace(-2, 2, 32).reshape(1, 2, 4, 4)
    oracle = original.double().requires_grad_()
    reference = torch.nn.functional.cross_entropy(oracle, target)
    reference.backward()
    logits = original.to(args.device).requires_grad_()
    actual = classification_loss(logits, target.to(args.device))
    actual.backward()
    loss_error = abs(actual.item()-reference.item())
    gradient_error = (logits.grad.cpu().double()-oracle.grad).abs().max().item()
    assert loss_error < 1e-6 and gradient_error < 1e-7
    print(json.dumps({'status': 'LOSS_AND_GRADIENT_VERIFIED', 'device': args.device,
        'torch': torch.__version__, 'loss_abs_error': loss_error, 'gradient_max_abs_error': gradient_error,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}))


if __name__ == '__main__':
    main()
