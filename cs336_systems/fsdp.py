"""Layer-wise FSDP with FP32 master shards and recomputed gathered weights.

Weights are gathered on demand (no prefetch buffers), and freed after each
operation. Custom autograd saves inputs/shards, never gathered full weights.
"""

import math
import types

import torch
import torch.distributed as dist
from torch import nn

from .distributed import world


class _Weight:
    def __init__(self, parameter, dtype, rank, size):
        self.shape = parameter.shape
        self.numel = parameter.numel()
        self.chunk = math.ceil(self.numel / size)
        self.rank, self.size = rank, size
        self.dtype = dtype or parameter.dtype
        flat = parameter.detach().flatten()
        padded = torch.zeros(self.chunk * size, dtype=parameter.dtype, device=parameter.device)
        padded[: self.numel].copy_(flat)
        self.parameter = nn.Parameter(padded[rank * self.chunk : (rank + 1) * self.chunk].clone(), requires_grad=parameter.requires_grad)

    def gather(self, shard, dtype=None):
        local = shard.to(dtype or self.dtype).contiguous()
        if self.size == 1:
            full = local
        else:
            parts = [torch.empty_like(local) for _ in range(self.size)]
            dist.all_gather(parts, local)
            full = torch.cat(parts)
        return full[: self.numel].reshape(self.shape)

    def scatter_gradient(self, full):
        # FP32 communication avoids repeated low precision reduction rounding.
        flat = full.float().flatten()
        padded = torch.zeros(self.chunk * self.size, dtype=torch.float32, device=flat.device)
        padded[: self.numel].copy_(flat)
        out = torch.empty(self.chunk, dtype=padded.dtype, device=flat.device)
        if self.size == 1:
            out.copy_(padded)
        elif dist.get_backend() == "gloo":
            # Gloo has no portable reduce_scatter support on all torch versions.
            dist.all_reduce(padded)
            out.copy_(padded[self.rank * self.chunk : (self.rank + 1) * self.chunk])
        else:
            dist.reduce_scatter_tensor(out, padded)
        return out.div_(self.size).to(self.parameter.dtype)


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shard, spec):
        weight = spec.gather(shard)
        ctx.spec = spec
        ctx.save_for_backward(x, shard)
        return x.to(weight.dtype) @ weight.T

    @staticmethod
    def backward(ctx, do):
        x, shard = ctx.saved_tensors
        spec = ctx.spec
        weight = spec.gather(shard)
        do = do.to(weight.dtype)
        dx = (do @ weight).to(x.dtype) if ctx.needs_input_grad[0] else None
        dw = do.reshape(-1, do.shape[-1]).T @ x.to(weight.dtype).reshape(-1, x.shape[-1])
        return dx, spec.scatter_gradient(dw), None


class _Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, shard, spec):
        weight = spec.gather(shard)
        ctx.spec = spec
        ctx.save_for_backward(ids)
        return weight[ids]

    @staticmethod
    def backward(ctx, do):
        (ids,) = ctx.saved_tensors
        spec = ctx.spec
        dw = torch.zeros(spec.shape, dtype=spec.dtype, device=do.device)
        dw.index_add_(0, ids.flatten(), do.reshape(-1, spec.shape[1]).to(dw.dtype))
        return None, spec.scatter_gradient(dw), None


def _linear_forward(layer, x):
    return _Linear.apply(x, layer.weight, layer._fsdp_spec)


def _embedding_forward(layer, ids):
    return _Embedding.apply(ids, layer.weight, layer._fsdp_spec)


class FSDP(nn.Module):
    """Shard cs336_basics Linear/Embedding; replicate small layers such as norms.

    Construct the optimizer after wrapping, because wrapping replaces parameters.
    The same graph and layer order must run on every rank. Gloo uses an
    all-reduce/slice fallback; NCCL uses actual reduce-scatter.
    """

    def __init__(self, module, compute_dtype=None):
        super().__init__()
        from cs336_basics.model import Embedding, Linear

        self.module = module
        self.world_size = world()
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.pending = []
        self.hooks = []
        if self.world_size > 1:
            for tensor in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(tensor.data, 0)
        specs = {}
        for layer in module.modules():
            if isinstance(layer, (Linear, Embedding)):
                if layer.weight not in specs:
                    specs[layer.weight] = _Weight(layer.weight, compute_dtype, self.rank, self.world_size)
                spec = specs[layer.weight]
                layer.weight = spec.parameter
                layer._fsdp_spec = spec
                layer.forward = types.MethodType(_linear_forward if isinstance(layer, Linear) else _embedding_forward, layer)
        self.sharded = {spec.parameter for spec in specs.values()}
        for p in module.parameters():
            if p.requires_grad and p not in self.sharded:
                self.hooks.append(p.register_post_accumulate_grad_hook(self._replicated_ready))

    def _replicated_ready(self, p):
        if self.world_size > 1:
            buffer = p.grad.detach().clone()
            self.pending.append((dist.all_reduce(buffer, async_op=True), p, buffer))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for work, p, buffer in self.pending:
            work.wait()
            p.grad.copy_(buffer.div_(self.world_size))
        self.pending.clear()

    @torch.no_grad()
    def gather_full_params(self):
        specs = {layer.weight: layer._fsdp_spec for layer in self.module.modules() if hasattr(layer, "_fsdp_spec")}
        return {name: specs[p].gather(p, p.dtype) if p in specs else p.detach().clone() for name, p in self.module.named_parameters()}
