"""Activation checkpoint schedules for a sequential Transformer stack."""

from torch.utils.checkpoint import checkpoint


def checkpoint_layers(layers, x, group_size=1, recursive=False):
    """Non-reentrant checkpointing, preserving RNG state for recomputation.

    group_size=0 disables checkpointing. Recursive bisection trades O(log N)
    saved boundaries for O(N log N) block work; flat groups recompute once.
    """
    if group_size < 0:
        raise ValueError("group_size must be nonnegative")

    def run(start, end, value):
        for i in range(start, end):
            value = layers[i](value)
        return value

    def recurse(start, end, value):
        if end - start <= 1:
            return run(start, end, value)
        mid = (start + end) // 2
        value = checkpoint(lambda t: recurse(start, mid, t), value, use_reentrant=False)
        return checkpoint(lambda t: recurse(mid, end, t), value, use_reentrant=False)

    if recursive:
        return recurse(0, len(layers), x)
    if not group_size:
        return run(0, len(layers), x)
    for start in range(0, len(layers), group_size):
        end = min(start + group_size, len(layers))
        # Bind indices now; recomputation runs after this loop has finished.
        x = checkpoint(lambda t, a=start, b=end: run(a, b, t), x, use_reentrant=False)
    return x
