"""Assignment DDP variants and whole-parameter optimizer state sharding."""

import torch
import torch.distributed as dist
from torch import nn


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


class DDP(nn.Module):
    """Average each completed parameter gradient asynchronously.

    Ranks must execute the same graph/collective order. Call finish before step
    and zero_grad. Gradient accumulation/no_sync and unused parameters with
    rank-dependent control flow are outside this educational wrapper's contract.
    """

    def __init__(self, module, mode="overlap"):
        super().__init__()
        if mode not in ("overlap", "individual", "flat"):
            raise ValueError("mode must be overlap, individual, or flat")
        self.module, self.mode = module, mode
        self.world_size = world()
        self.pending = []
        self.hooks = []
        if self.world_size > 1:
            for t in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(t.data, src=0)
        if mode == "overlap":
            for p in module.parameters():
                if p.requires_grad:
                    self.hooks.append(p.register_post_accumulate_grad_hook(self._ready))

    def _ready(self, parameter):
        if self.world_size > 1:
            # Own the communication buffer until work.wait(); don't mutate a
            # gradient that the autograd engine may still reference.
            buffer = parameter.grad.detach().clone()
            work = dist.all_reduce(buffer, async_op=True)
            self.pending.append((work, parameter, buffer))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        if self.mode == "overlap":
            for work, p, buffer in self.pending:
                work.wait()
                p.grad.copy_(buffer.div_(self.world_size))
            self.pending.clear()
        elif self.world_size > 1:
            grads = [p.grad for p in self.module.parameters() if p.grad is not None]
            if self.mode == "individual":
                for grad in grads:
                    dist.all_reduce(grad)
                    grad.div_(self.world_size)
            else:
                # Keep different dtypes/devices in distinct flat buffers.
                groups = {}
                for grad in grads:
                    groups.setdefault((grad.device, grad.dtype), []).append(grad)
                for group in groups.values():
                    flat = torch.cat([g.flatten() for g in group])
                    dist.all_reduce(flat)
                    flat.div_(self.world_size)
                    offset = 0
                    for grad in group:
                        grad.copy_(flat[offset : offset + grad.numel()].view_as(grad))
                        offset += grad.numel()


class ShardedOptimizer(torch.optim.Optimizer):
    """ZeRO-1 style state ownership; gradients must already be averaged by DDP.

    Greedily assigns entire tensors to ranks by element count. Each owner updates
    its parameters, then broadcasts those parameters to the other replicas.
    Save one state_dict per rank, restoring with the same world size/order.
    """

    def __init__(self, params, optimizer_cls, **kwargs):
        self.world_size = world()
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.loads = [0] * self.world_size
        self.owners = {}
        self.local_optimizer = None
        super().__init__(params, kwargs)
        groups = [self._local_group(g) for g in self.param_groups]
        self.local_optimizer = optimizer_cls(groups, **kwargs)
        self.defaults.update(self.local_optimizer.defaults)
        for group, local in zip(self.param_groups, self.local_optimizer.param_groups):
            for key, value in local.items():
                if key != "params":
                    group.setdefault(key, value)
        self.state = self.local_optimizer.state

    def _local_group(self, group):
        return {**group, "params": [p for p in group["params"] if self.owners[p] == self.rank]}

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        for p in group["params"]:
            if p in self.owners:
                raise ValueError("A parameter cannot appear in multiple groups")
            owner = min(range(self.world_size), key=self.loads.__getitem__)
            self.owners[p] = owner
            self.loads[owner] += p.numel()
        if self.local_optimizer is not None:
            self.local_optimizer.add_param_group(self._local_group(group))

    def step(self, closure=None, **kwargs):
        for group, local in zip(self.param_groups, self.local_optimizer.param_groups):
            local.update({k: v for k, v in group.items() if k != "params"})
        loss = self.local_optimizer.step(closure=closure, **kwargs)
        if self.world_size > 1:
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group["params"]:
                        dist.broadcast(p.data, src=self.owners[p])
        return loss

    def state_dict(self):
        return {"rank": self.rank, "world_size": self.world_size, "optimizer": self.local_optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        if (state_dict["rank"], state_dict["world_size"]) != (self.rank, self.world_size):
            raise ValueError("Restore each rank with its own checkpoint and the same world size")
        self.local_optimizer.load_state_dict(state_dict["optimizer"])
        self.state = self.local_optimizer.state
        for group, local in zip(self.param_groups, self.local_optimizer.param_groups):
            group.update({k: v for k, v in local.items() if k != "params"})
