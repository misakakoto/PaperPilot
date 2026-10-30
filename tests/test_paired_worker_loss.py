"""CPU algebra/gradient regression. The CUDA compatibility repro is a separate CLI."""
import pytest

from paperpilot_agentic.paired_snunet_worker import classification_loss, global_max_pool


@pytest.mark.parametrize('shape', [(1, 2, 4, 4), (2, 3, 3, 5)])
def test_pixel_ce_preserves_loss_and_gradient(shape):
    torch = pytest.importorskip('torch')
    torch.manual_seed(7)
    logits = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    oracle = logits.detach().clone().requires_grad_()
    target = torch.randint(0, shape[1], (shape[0], *shape[2:]))
    actual = classification_loss(logits, target)
    expected = torch.nn.functional.cross_entropy(oracle, target)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(logits.grad, oracle.grad, rtol=0, atol=1e-12)


@pytest.mark.parametrize('shape', [(1, 2, 4), (1, 2, 3, 4)])
def test_wrong_loss_tensor_contract_is_rejected(shape):
    torch = pytest.importorskip('torch')
    with pytest.raises(ValueError, match='NCHW logits'):
        classification_loss(torch.zeros(shape), torch.zeros((1, 4, 4), dtype=torch.int64))


def test_global_pool_preserves_tie_gradient():
    torch = pytest.importorskip('torch')
    value = (torch.arange(96) % 7).reshape(2, 3, 4, 4).double().requires_grad_()
    oracle = value.detach().clone().requires_grad_()
    actual = global_max_pool(value)
    expected = torch.nn.functional.adaptive_max_pool2d(oracle, 1)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(value.grad, oracle.grad, rtol=0, atol=0)
