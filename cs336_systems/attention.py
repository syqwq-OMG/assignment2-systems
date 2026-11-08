"""Tiled PyTorch reference for the FlashAttention recurrence and its derivative."""
import math

import torch


def validate_inputs(q, k, v):
    if q.ndim < 2 or k.ndim != q.ndim or v.shape != k.shape:
        raise ValueError("Expected Q (..., Nq, D), K/V (..., Nk, D)")
    if q.shape[:-2] != k.shape[:-2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Batch dimensions and head dimensions must match")
    if min(*q.shape, *k.shape) <= 0:
        raise ValueError("Empty attention dimensions are unsupported")
    if q.device != k.device or q.device != v.device or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q/K/V must share device and dtype")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Expected floating-point inputs")


class FlashAttentionPytorch(torch.autograd.Function):
    """Educational reference: O(ND) saved state, O(block_size**2) score tiles."""

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        validate_inputs(q, k, v)
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        nq, d = q.shape[-2:]
        nk = k.shape[-2]
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=dtype)
        scale = 1 / math.sqrt(d)
        for i in range(0, nq, 32):
            qi = q[..., i:i + 32, :].to(dtype)
            m = torch.full(qi.shape[:-1], -torch.inf, device=q.device, dtype=dtype)
            ell = torch.zeros_like(m)
            acc = torch.zeros_like(qi)
            for j in range(0, nk, 32):
                if is_causal and j > i + qi.shape[-2] - 1:
                    break
                s = qi @ k[..., j:j + 32, :].to(dtype).transpose(-1, -2) * scale
                if is_causal:
                    rows = torch.arange(i, i + qi.shape[-2], device=q.device)
                    cols = torch.arange(j, min(j + 32, nk), device=q.device)
                    s = s.masked_fill(rows[:, None] < cols[None, :], -torch.inf)
                new_m = torch.maximum(m, s.amax(-1))
                alpha = torch.exp(m - new_m)
                p = torch.exp(s - new_m[..., None])
                acc = alpha[..., None] * acc + p @ v[..., j:j + 32, :].to(dtype)
                ell = alpha * ell + p.sum(-1)
                m = new_m
            out[..., i:i + 32, :] = (acc / ell[..., None]).to(q.dtype)
            lse[..., i:i + 32] = m + ell.log()
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors
        dtype = lse.dtype
        dq, dk, dv = [torch.zeros_like(t, dtype=dtype) for t in (q, k, v)]
        delta = (out.to(dtype) * do.to(dtype)).sum(-1)
        nq, d = q.shape[-2:]
        nk = k.shape[-2]
        scale = d ** -0.5
        for i in range(0, nq, 32):
            qi, doi = q[..., i:i + 32, :].to(dtype), do[..., i:i + 32, :].to(dtype)
            for j in range(0, nk, 32):
                if ctx.is_causal and j > i + qi.shape[-2] - 1:
                    break
                kj, vj = k[..., j:j + 32, :].to(dtype), v[..., j:j + 32, :].to(dtype)
                s = qi @ kj.transpose(-1, -2) * scale
                if ctx.is_causal:
                    rows = torch.arange(i, i + qi.shape[-2], device=q.device)
                    cols = torch.arange(j, min(j + 32, nk), device=q.device)
                    s = s.masked_fill(rows[:, None] < cols[None, :], -torch.inf)
                p = torch.exp(s - lse[..., i:i + 32, None])
                ds = p * (doi @ vj.transpose(-1, -2) - delta[..., i:i + 32, None]) * scale
                dq[..., i:i + 32, :] += ds @ kj
                dk[..., j:j + 32, :] += ds.transpose(-1, -2) @ qi
                dv[..., j:j + 32, :] += p.transpose(-1, -2) @ doi
        grads = (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None)
        return grads[:len(ctx.needs_input_grad)]
