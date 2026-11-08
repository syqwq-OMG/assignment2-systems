"""Standalone course submission: exact tiled attention, with Triton forward/backward.

Copy this file alone as sunyuquan.py for the course evaluator. No package imports
or installed FlashAttention/SDPA are needed. FP16 evaluator inputs keep config C
within 80 GB; accumulation and normalization are FP32. FP32 callers use tf32x3.
"""

import math
import os
from contextlib import nullcontext as _nullcontext

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(Q, K, V, O, L, NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, CAUSAL: tl.constexpr, SCALE: tl.constexpr):
    h = tl.program_id(1).to(tl.int64)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    dims = tl.arange(0, BD)
    cols = tl.arange(0, BN)
    q = tl.load(Q + h * NQ * D + rows[:, None] * D + dims[None, :], (rows[:, None] < NQ) & (dims[None, :] < D), 0)
    m = tl.full((BM,), -float("inf"), tl.float32)
    ell = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    end = NK
    if CAUSAL:
        end = tl.minimum(NK, (tl.program_id(0) + 1) * BM)
    start = 0
    while start < end:
        c = start + cols
        k = tl.load(K + h * NK * D + dims[:, None] + c[None, :] * D, (dims[:, None] < D) & (c[None, :] < NK), 0)
        v = tl.load(V + h * NK * D + c[:, None] * D + dims[None, :], (c[:, None] < NK) & (dims[None, :] < D), 0)
        s = tl.dot(q, k, input_precision="tf32x3") * SCALE
        valid = c[None, :] < NK
        if CAUSAL:
            valid = valid & (rows[:, None] >= c[None, :])
        s = tl.where(valid, s, -float("inf"))
        new_m = tl.maximum(m, tl.max(s, 1))
        alpha = tl.exp(m - new_m)
        p = tl.exp(s - new_m[:, None])
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc, input_precision="tf32x3")
        ell = ell * alpha + tl.sum(p, 1)
        m = new_m
        start += BN
    result = acc / ell[:, None]
    tl.store(O + h * NQ * D + rows[:, None] * D + dims[None, :], result, (rows[:, None] < NQ) & (dims[None, :] < D))
    tl.store(L + h * NQ + rows, m + tl.log(ell), rows < NQ)


@triton.jit
def _delta(O, DO, DELTA, N: tl.constexpr, D: tl.constexpr, SH: tl.constexpr, SN: tl.constexpr, SD: tl.constexpr, BD: tl.constexpr, BM: tl.constexpr):
    h = tl.program_id(1).to(tl.int64)
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    d = tl.arange(0, BD)
    mask = (r[:, None] < N) & (d[None, :] < D)
    o = tl.load(O + h * N * D + r[:, None] * D + d[None, :], mask, 0).to(tl.float32)
    do = tl.load(DO + h * SH + r[:, None] * SN + d[None, :] * SD, mask, 0).to(tl.float32)
    tl.store(DELTA + h * N + r, tl.sum(o * do, 1), r < N)


@triton.jit
def _backward_q(
    Q,
    K,
    V,
    DO,
    L,
    DELTA,
    DQ,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    SH: tl.constexpr,
    SN: tl.constexpr,
    SD: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
):
    h = tl.program_id(1).to(tl.int64)
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.arange(0, BN)
    d = tl.arange(0, BD)
    mask = (r[:, None] < NQ) & (d[None, :] < D)
    q = tl.load(Q + h * NQ * D + r[:, None] * D + d[None, :], mask, 0)
    do = tl.load(DO + h * SH + r[:, None] * SN + d[None, :] * SD, mask, 0)
    l = tl.load(L + h * NQ + r, r < NQ, 0)
    delta = tl.load(DELTA + h * NQ + r, r < NQ, 0)
    dq = tl.zeros((BM, BD), tl.float32)
    end = NK
    if CAUSAL:
        end = tl.minimum(NK, (tl.program_id(0) + 1) * BM)
    start = 0
    while start < end:
        col = start + c
        kt = tl.load(K + h * NK * D + d[:, None] + col[None, :] * D, (d[:, None] < D) & (col[None, :] < NK), 0)
        vt = tl.load(V + h * NK * D + d[:, None] + col[None, :] * D, (d[:, None] < D) & (col[None, :] < NK), 0)
        s = tl.dot(q, kt, input_precision="tf32x3") * SCALE
        valid = (r[:, None] < NQ) & (col[None, :] < NK)
        if CAUSAL:
            valid = valid & (r[:, None] >= col[None, :])
        p = tl.where(valid, tl.exp(s - l[:, None]), 0.0)
        dp = tl.dot(do, vt, input_precision="tf32x3")
        ds = p * (dp - delta[:, None]) * SCALE
        dq = tl.dot(ds.to(kt.dtype), tl.trans(kt), dq, input_precision="tf32x3")
        start += BN
    tl.store(DQ + h * NQ * D + r[:, None] * D + d[None, :], dq, mask)


@triton.jit
def _backward_kv(
    Q,
    K,
    V,
    DO,
    L,
    DELTA,
    DK,
    DV,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    SH: tl.constexpr,
    SN: tl.constexpr,
    SD: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
):
    h = tl.program_id(1).to(tl.int64)
    c = tl.program_id(0) * BN + tl.arange(0, BN)
    r = tl.arange(0, BM)
    d = tl.arange(0, BD)
    mask = (c[:, None] < NK) & (d[None, :] < D)
    k = tl.load(K + h * NK * D + c[:, None] * D + d[None, :], mask, 0)
    v = tl.load(V + h * NK * D + c[:, None] * D + d[None, :], mask, 0)
    dk = tl.zeros((BN, BD), tl.float32)
    dv = tl.zeros((BN, BD), tl.float32)
    begin = 0
    if CAUSAL:
        begin = (tl.program_id(0) * BN // BM) * BM
    start = begin
    while start < NQ:
        row = start + r
        qt = tl.load(Q + h * NQ * D + d[:, None] + row[None, :] * D, (d[:, None] < D) & (row[None, :] < NQ), 0)
        dot = tl.load(DO + h * SH + d[:, None] * SD + row[None, :] * SN, (d[:, None] < D) & (row[None, :] < NQ), 0)
        l = tl.load(L + h * NQ + row, row < NQ, 0)
        delta = tl.load(DELTA + h * NQ + row, row < NQ, 0)
        st = tl.dot(k, qt, input_precision="tf32x3") * SCALE
        valid = (c[:, None] < NK) & (row[None, :] < NQ)
        if CAUSAL:
            valid = valid & (row[None, :] >= c[:, None])
        pt = tl.where(valid, tl.exp(st - l[None, :]), 0.0)
        dv = tl.dot(pt.to(dot.dtype), tl.trans(dot), dv, input_precision="tf32x3")
        dpt = tl.dot(v, dot, input_precision="tf32x3")
        dst = pt * (dpt - delta[None, :]) * SCALE
        dk = tl.dot(dst.to(qt.dtype), tl.trans(qt), dk, input_precision="tf32x3")
        start += BM
    tl.store(DK + h * NK * D + c[:, None] * D + d[None, :], dk, mask)
    tl.store(DV + h * NK * D + c[:, None] * D + d[None, :], dv, mask)


class FlashAttention2(torch.autograd.Function):
    INPUT_DTYPE = torch.float16

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        if q.ndim < 2 or k.ndim != q.ndim or k.shape != v.shape:
            raise ValueError("Expected Q (..., Nq, D), K/V (..., Nk, D)")
        if q.shape[:-2] != k.shape[:-2] or q.shape[-1] != k.shape[-1]:
            raise ValueError("Batch and head dimensions must match")
        if min(*q.shape, *k.shape) <= 0 or not 1 <= q.shape[-1] <= 256:
            raise ValueError("Expected nonempty tensors with 1 <= D <= 256")
        if any(t.device != q.device or t.dtype != q.dtype for t in (k, v)):
            raise ValueError("Q/K/V must share device and dtype")
        if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Supported dtypes: FP32, FP16, BF16")
        if q.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
            raise ValueError("Triton attention requires CUDA (or TRITON_INTERPRET=1)")
        ctx.shape = q.shape
        ctx.kshape = k.shape
        nq, d = q.shape[-2:]
        nk = k.shape[-2]
        h = math.prod(q.shape[:-2])
        q, k, v = [t.contiguous().reshape(h, t.shape[-2], d) for t in (q, k, v)]
        out = torch.empty_like(q)
        lse = torch.empty((h, nq), dtype=torch.float32, device=q.device)
        bd = max(16, triton.next_power_of_2(d))
        # Fixed configurations avoid costly runtime autotuning on the shared GPU.
        bm, bn = 32, 64
        with torch.cuda.device(q.device) if q.is_cuda else _nullcontext():
            _forward[(triton.cdiv(nq, bm), h)](q, k, v, out, lse, nq, nk, d, bd, bm, bn, is_causal, d**-0.5, num_warps=4, num_stages=2)
        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, out, lse.reshape(ctx.shape[:-1]))
        return out.reshape(ctx.shape)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        h, nq, d = q.shape
        nk = k.shape[-2]
        # In particular, do NOT call contiguous() on the broadcast gradient of sum().
        do = grad_out.reshape(h, nq, d)
        dq, dk, dv = [torch.empty_like(t) for t in (q, k, v)]
        delta = torch.empty((h, nq), device=q.device, dtype=torch.float32)
        bd = max(16, triton.next_power_of_2(d))
        bm, bn = 32, 64
        with torch.cuda.device(q.device) if q.is_cuda else _nullcontext():
            _delta[(triton.cdiv(nq, bm), h)](out, do, delta, nq, d, *do.stride(), bd, bm, num_warps=4)
            args = (q, k, v, do, lse, delta)
            meta = (nq, nk, d, *do.stride(), bd, bm, bn, ctx.is_causal, d**-0.5)
            _backward_q[(triton.cdiv(nq, bm), h)](*args, dq, *meta, num_warps=4, num_stages=2)
            _backward_kv[(triton.cdiv(nk, bn), h)](*args, dk, dv, *meta, num_warps=4, num_stages=2)
        grads = (dq.reshape(ctx.shape), dk.reshape(ctx.kshape), dv.reshape(ctx.kshape), None)
        return grads[: len(ctx.needs_input_grad)]
