"""Small CPU regressions for checkpoint recomputation and optimizer ownership."""

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from cs336_systems.checkpointing import checkpoint_layers
from cs336_systems.distributed import DDP, ShardedOptimizer
from cs336_systems.fsdp import FSDP


@pytest.mark.parametrize("group,recursive", [(0, False), (1, False), (2, False), (3, False), (1, True)])
def test_checkpoint_gradients(group, recursive):
    torch.manual_seed(0)
    layers = nn.ModuleList([nn.Sequential(nn.Linear(7, 7), nn.Dropout(0.1), nn.ReLU()) for _ in range(5)])
    ref_layers = deepcopy(layers)
    x = torch.randn(2, 7, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    torch.manual_seed(123)
    actual = checkpoint_layers(layers, x, group, recursive)
    actual.sum().backward()
    torch.manual_seed(123)
    expected = nn.Sequential(*ref_layers)(xr)
    expected.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, xr.grad)
    for p, ref in zip(layers.parameters(), ref_layers.parameters()):
        torch.testing.assert_close(p.grad, ref.grad)


def _worker(rank, store):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2)
    try:
        # The first group leaves rank 1 empty, and the second is added at runtime.
        params = [nn.Parameter(torch.ones(3)), nn.Parameter(torch.ones(5))]
        refs = [nn.Parameter(p.detach().clone()) for p in params]
        optimizer = ShardedOptimizer([{"params": [params[0]], "lr": 0.1}], torch.optim.AdamW)
        optimizer.add_param_group({"params": [params[1]], "lr": 0.02})
        reference = torch.optim.AdamW([{"params": [refs[0]], "lr": 0.1}, {"params": [refs[1]], "lr": 0.02}])
        for _ in range(2):
            for p, ref in zip(params, refs):
                p.grad, ref.grad = torch.ones_like(p), torch.ones_like(ref)
            optimizer.step()
            reference.step()
        for p, ref in zip(params, refs):
            torch.testing.assert_close(p, ref)
        assert all(optimizer.owners[p] == rank for p in optimizer.state)
        state = deepcopy(optimizer.state_dict())
        restored = ShardedOptimizer([{"params": [params[0]], "lr": 9.0}], torch.optim.AdamW)
        restored.add_param_group({"params": [params[1]], "lr": 9.0})
        restored.load_state_dict(state)
        assert [g["lr"] for g in restored.param_groups] == [0.1, 0.02]
        for mode in ("individual", "flat", "overlap"):
            torch.manual_seed(0)
            model = nn.Linear(7, 3)
            ref = deepcopy(model)
            wrapped = DDP(model, mode)
            torch.manual_seed(10)
            x = torch.randn(4, 7)
            ref(x).square().mean().backward()
            wrapped(x[rank * 2 : (rank + 1) * 2]).square().mean().backward()
            wrapped.finish_gradient_synchronization()
            for p, expected in zip(model.parameters(), ref.parameters()):
                torch.testing.assert_close(p.grad, expected.grad)
        from cs336_basics.model import Embedding, Linear

        torch.manual_seed(1)
        model = nn.Sequential(Embedding(11, 7), Linear(7, 7), Linear(7, 7))
        model[2].weight = model[1].weight
        ref = deepcopy(model)
        wrapped = FSDP(model)
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        ref(x).square().mean().backward()
        wrapped(x[rank : rank + 1]).square().mean().backward()
        wrapped.finish_gradient_synchronization()
        torch.optim.SGD(wrapped.parameters(), lr=0.01).step()
        torch.optim.SGD(ref.parameters(), lr=0.01).step()
        full = wrapped.gather_full_params()
        for name, param in ref.named_parameters():
            torch.testing.assert_close(full[name], param)
    finally:
        dist.destroy_process_group()


def test_distributed_extensions(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
