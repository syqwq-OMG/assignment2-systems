"""Small correctness cases only; no benchmark sweeps or large allocations."""
import importlib.util
import os

import pytest
import torch

from cs336_systems.attention import FlashAttentionPytorch


def dense(q, k, v, causal):
    s = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
    if causal:
        rows = torch.arange(q.shape[-2], device=q.device)
        cols = torch.arange(k.shape[-2], device=q.device)
        s = s.masked_fill(rows[:, None] < cols[None, :], -torch.inf)
    return s.softmax(-1) @ v


@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('nq,nk,d', [(1, 1, 16), (33, 65, 17), (65, 33, 100)])
@pytest.mark.parametrize('impl', ['pytorch', 'triton'])
def test_edges(impl, nq, nk, d, causal):
    if impl == 'triton':
        interpreter = os.environ.get('TRITON_INTERPRET') == '1'
        if importlib.util.find_spec('triton') is None or not (interpreter or torch.cuda.is_available()):
            pytest.skip('Requires Triton interpreter or CUDA')
        from cs336_systems.flash_attention import FlashAttention2
        function = FlashAttention2
        device = 'cpu' if interpreter else 'cuda'
    else:
        function, device = FlashAttentionPytorch, 'cpu'
    torch.manual_seed(42)
    # Sliced inputs exercise the public interface's non-contiguous input handling.
    q = torch.randn(2, nq, d * 2, device=device)[..., ::2].detach().requires_grad_()
    k = torch.randn(2, nk, d * 2, device=device)[..., ::2].detach().requires_grad_()
    v = torch.randn_like(k).requires_grad_()
    do = torch.randn_like(q)
    reference = dense(q, k, v, causal)
    expected = torch.autograd.grad(reference, (q, k, v), do)
    actual = function.apply(q, k, v, causal)
    grads = torch.autograd.grad(actual, (q, k, v), do)
    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-4)
    for grad, ref in zip(grads, expected):
        torch.testing.assert_close(grad, ref, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_triton_sum_gradient(dtype):
    interpreter = os.environ.get('TRITON_INTERPRET') == '1'
    if importlib.util.find_spec('triton') is None or not (interpreter or torch.cuda.is_available()):
        pytest.skip('Requires Triton interpreter or CUDA')
    if interpreter and dtype == torch.bfloat16:
        pytest.skip('Triton interpreter does not support BF16 dot products')
    from cs336_systems.flash_attention import FlashAttention2
    device = 'cpu' if interpreter else 'cuda'
    torch.manual_seed(123)
    inputs = [torch.randn(1, 67, 100, device=device, dtype=dtype).requires_grad_() for _ in range(3)]
    refs = [t.detach().float().requires_grad_() for t in inputs]
    ref = dense(*refs, False)
    ref.sum().backward()
    actual = FlashAttention2.apply(*inputs)
    actual.sum().backward()
    tol = 3e-5 if dtype == torch.float32 else (3e-3 if dtype == torch.float16 else 3e-2)
    torch.testing.assert_close(actual.float(), ref, atol=tol, rtol=tol)
    for t, r in zip(inputs, refs):
        torch.testing.assert_close(t.grad.float(), r.grad, atol=tol, rtol=tol)
