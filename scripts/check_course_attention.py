"""One bounded correctness run against course FP32 references.

A supports 1..24 heads. B/C permit only the first head, exercising the real long
sequence without running the full workload. NPZ arrays are read as a streaming
prefix; no full B/C CPU arrays, sweeps, or benchmark repetitions are allocated.
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from cs336_systems.flash_attention import FlashAttention2

CONFIGS = {"A": (24, 16384, 128), "B": (64, 130000, 128), "C": (256, 131072, 100)}


def read_prefix(archive, key, expected_shape, heads):
    """Read only requested leading heads from an untrusted NPY ZIP member."""
    with archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in ((2, 0), (3, 0)):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported NPY version: {version}")
        if tuple(shape) != expected_shape or fortran or dtype != np.dtype("float32"):
            raise ValueError(f"Unexpected {key} header: {shape}, {dtype}, {fortran}")
        count = heads * expected_shape[1] * expected_shape[2]
        raw = stream.read(count * 4)
        if len(raw) != count * 4:
            raise EOFError(f"Truncated {key} data")
        return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy().reshape(heads, *expected_shape[1:]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configuration", choices=list(CONFIGS), default="A")
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--data-dir", type=Path, default=Path("/lmfs2026/data/2-fa2-data"))
    a = p.parse_args()
    h, n, d = shape = CONFIGS[a.configuration]
    limit = 24 if a.configuration == "A" else 1
    if not 1 <= a.heads <= limit:
        p.error(f"This bounded checker permits 1..{limit} heads for {a.configuration}")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    path = a.data_dir / f"fa2_nheads{h}_seq{n}_dhead{d}.npz"
    with zipfile.ZipFile(path) as archive:
        if a.configuration == "C":
            metadata = json.loads(archive.comment.decode("utf-8"))
            checks = {
                "shape": list(shape),
                "qkv_generator": "torch.CPU.Generator",
                "head_seed_rule": "seed + head_id",
                "input_dtype": "float32",
                "reference_dtype": "float32",
                "loss": "o.sum()",
            }
            if any(metadata.get(key) != value for key, value in checks.items()):
                raise ValueError("C generator metadata does not match the course evaluator")
            seed = metadata.get("seed")
            if type(seed) is not int or seed < 0:
                raise ValueError("Missing or invalid C generator seed")
            generator = torch.Generator(device="cpu").manual_seed(seed)
            # Head 0: FP32 CPU Q, then K, then V, and only then input conversion.
            cpu_inputs = [torch.randn((n, d), generator=generator, dtype=torch.float32).unsqueeze(0) for _ in range(3)]
        else:
            cpu_inputs = [read_prefix(archive, key, shape, a.heads) for key in ("q", "k", "v")]
        inputs = [t.to("cuda", FlashAttention2.INPUT_DTYPE).requires_grad_() for t in cpu_inputs]
        del cpu_inputs
        output = FlashAttention2.apply(*inputs)
        output.sum().backward()
        torch.cuda.synchronize()
        actual = {"o": output, "grad_q": inputs[0].grad, "grad_k": inputs[1].grad, "grad_v": inputs[2].grad}
        if a.configuration == "C":
            actual = {key: actual[key] for key in ("o", "grad_q")}
        errors = {}
        passed = True
        for key, result in actual.items():
            reference = read_prefix(archive, key, shape, a.heads).to("cuda")
            converted = result.detach().float()
            finite = bool(torch.isfinite(converted).all())
            differences = (converted - reference).abs()
            flat_index = int(differences.argmax().item())
            index = tuple(int(i) for i in np.unravel_index(flat_index, converted.shape))
            error = differences.flatten()[flat_index].item()
            thresholds = {str(eps): finite and torch.allclose(converted, reference, atol=eps, rtol=0) for eps in (1e-3, 1e-2, 1e-1)}
            errors[key] = {
                "max_absolute_error": error,
                "max_error_index": index,
                "actual_at_max": converted[index].item(),
                "reference_at_max": reference[index].item(),
                "rounded_reference_at_max": reference[index].to(FlashAttention2.INPUT_DTYPE).float().item(),
                "finite": finite,
                "passes": thresholds,
            }
            passed = passed and thresholds["0.001"]
            del reference, converted, differences
        print(
            json.dumps(
                {
                    "configuration": a.configuration,
                    "coverage": "full" if a.heads == h else "leading-head subset",
                    "heads_checked": a.heads,
                    "total_heads": h,
                    "dtype": str(FlashAttention2.INPUT_DTYPE),
                    "all_pass_1e-3": passed,
                    "errors": errors,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
