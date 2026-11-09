# Real-GPU validation and optimization, November 9, 2026

## GPU availability

At 14:52 CST, the assignment-2 endpoint (`172.23.166.144:32343`) exposed one
NVIDIA A800 80GB PCIe. Its CUDA index was **0**, with UUID
`GPU-bf236db6-0b88-f20e-a1d2-a9a06fce37b1`, backed by `/dev/nvidia2`.
It had 0 MiB allocated, 0% utilization, and no compute processes. **GPU 3 was
not visible through this login, so its availability could not be established.**
The other documented endpoint, port 37630, refused connections. Runs used the
idle GPU assigned to the assignment-2 container; no GPU numbering was inferred
from the reservation table. The assigned GPU was idle again after verification.

The runtime was the existing PyTorch 2.11.0+cu128 / Triton 3.6.0 environment.
No shared dependencies, reference datasets, or evaluator files were modified.

## Results

Real CUDA attention tests: **29 passed**, including FP32, FP16, BF16,
causal/noncausal masking, unequal query/key lengths, D=100/128, and arbitrary
upstream gradients. Eight new cases specifically cover the tuned low-precision
backward branch rather than checking only the evaluator's summed loss.

### End-to-end speed

Same GPU, same FP16 inputs/upstream gradient, noncausal attention, compilation
excluded. Each entry is the median of three short trials, each using
`triton.testing.do_bench(...,warmup=10,rep=30,return_mode='median')`.
Times include forward, Delta, both backward kernels, autograd dispatch, and
clearing input gradients. The A-sized timing uses generated random inputs; the
course accuracy check below separately uses the saved course data.

| H / N / D | Upstream gradient | Original F+B | Tuned F+B | Speedup |
| --- | --- | ---: | ---: | ---: |
| 4 / 4096 / 100 | Sum loss, zero strides | 15.256 ms | 1.812 ms | **8.42x** |
| 4 / 4096 / 128 | Random dense tensor | 15.244 ms | 2.197 ms | **6.94x** |
| 24 / 16384 / 128 | Sum loss, zero strides | 1312.883 ms | 152.110 ms | **8.63x** |

All three comparisons had zero observed difference between original/tuned O,
dQ, dK, and dV for their generated inputs. This is an observation for those
inputs, not a promise of bitwise equality on every shape. Timings are development
measurements, not official efficiency rankings. The 30ms requested repetition
window is shorter than one A-sized call; its trials therefore contain very few
measured calls, and are sufficient to establish a large speed difference rather
than a precise long-run throughput distribution. Baseline and tuned measurements
were sequential in the same process, with no competing jobs intentionally
launched. Clock drift and external interference remain possible.

The comparison's peak allocation includes retained baseline tensors and Triton's
benchmark cache-flush buffer; it is not the bare attention memory footprint or an
official evaluator memory result. No OOM occurred in these runs.

### Course FP32-reference accuracy

One forward/backward per correctness check, `output.sum().backward()`, inputs
converted to the class's declared FP16 dtype, absolute comparison with `rtol=0`.

| Configuration | Heads checked | O maximum error | dQ maximum error | dK maximum error | dV maximum error | Worst passing atol |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A: 24 / 16384 / 128 | **All 24** | 5.312e-5 | 1.031e-4 | **1.153e-3** | 6.303e-4 | **1e-2** |
| B: 64 / 130000 / 128 | **First 1 of 64** | 1.653e-5 | 3.829e-5 | **1.017e-3** | 6.586e-4 | **1e-2 for subset** |
| C: 256 / 131072 / 100 | **First 1 of 256** | 1.794e-5 | 3.649e-5 | No supplied reference | No supplied reference | **1e-3 for subset** |

All checked results were finite. C follows the course evaluator's seed-based
FP32 CPU input reconstruction and compares only O/dQ, as specified by its NPZ.
B/C were deliberately limited to one head each: these validate actual sequence
lengths and tail handling without running the full multi-head workloads.

**A does not pass the strict 1e-3 tier.** Its measured accuracy coefficient is
0.7 under the course thresholds. This is not an overall coefficient claim:
full B/C correctness and memory/performance evaluation remain outstanding.

The largest A dK error is at `[21,4830,11]`:

- FP32 reference: `1.3924018144607544`.
- Actual FP16 gradient: `1.3935546875`.
- FP16 rounding of the reference itself: `1.392578125`.
- A diagnostic recomputation of that element with FP32 score-gradient arithmetic
  on the FP16-rounded inputs: about `1.3931434`, which still rounds to
  `1.3935546875`.

Thus removing the score-gradient tile's FP16 rounding alone does not fix this
case; input quantization is already significant. The tile optimization retains
the original FP16 numerical behavior. No rounding bias or reference-dependent
special case was introduced to make this particular reference pass. FP32 inputs
are not simply enabled for the evaluator because config C's seven ordinary FP32
input/output/gradient tensors require 87.5 GiB before temporaries. A different
precision/storage design would require a separate validated implementation.

## What changed and why

Only the low-precision dK/dV launch configuration changed for `64 < D <= 128`:

```text
Before: BM=32, BN=64, num_warps=4, num_stages=2
After:  BM=32, BN=32, num_warps=4, num_stages=1
```

Forward and dQ retain their previous configurations. FP32 and other head widths
retain the old dK/dV dispatch. BF16 uses the same tuned low-precision branch and
passes the real-GPU numerical tests, but BF16 speed was not separately benchmarked.

The dK/dV kernel keeps two FP32 accumulators plus score/probability/derivative
tiles live. The original tile creates severe register pressure and local-memory
spills. More arithmetic per tile did not compensate for spill traffic. A smaller
key tile and fewer pipeline buffers reduced that pressure:

| Isolated dK/dV, H=4,N=4096,D=100,sum gradient | Original | Tuned |
| --- | ---: | ---: |
| Median runtime | 14.182 ms | 0.943 ms |
| Compiler-reported spills | 932 | 26 |
| Static shared memory | 32768 bytes | 24576 bytes |

The spill number is Triton's compiler metadata, not bytes of measured DRAM
traffic. The new configuration still spills somewhat; it is a measured useful
improvement, not a claim of optimality. A separate D=128 random-gradient
confirmation reduced isolated dK/dV time from about 14.087 to 1.117 ms.

Two explicit four-candidate tile experiments were run, then two-candidate
confirmation checks. There was no unrestricted autotuning, large-shape grid,
or full B/C benchmark. Most GPU work was short development calls; the bounded
A-sized baseline/tuned comparison was the longest experiment.

## Reproduce the bounded checks

In the personal server working copy, activate the existing environment and make
the repository/Basics importable:

```bash
source /lmfs2026/env/a1/bin/activate
cd /ai/lmfs2026/sunyuquan/assignment2-codex
export PYTHONPATH="/lmfs2026/data/assignment2-systems/cs336-basics:$PWD"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m pytest tests/test_attention.py tests/test_attention_edges.py -q
python -m scripts.check_course_attention --configuration A --heads 24
python -m scripts.check_course_attention --configuration B --heads 1
python -m scripts.check_course_attention --configuration C --heads 1
```

The checker streams only the requested leading heads from ZIP/NPY members,
validates headers, and rejects requests for more than one B/C head. It performs
no benchmark repetitions. It reports threshold results rather than treating
1e-3 as the only valid tier.

Small tile inspection and controlled comparison:

```bash
python -m scripts.tune_attention --length 4096 --head-dim 100 --sum-gradient --preset small
python -m scripts.compare_attention --baseline flash_attention_baseline.py \
  --length 4096 --heads 4 --head-dim 100 --sum-gradient
```

`flash_attention_baseline.py` is the standalone source from baseline commit
`57a68c3`; the server retains that snapshot in the personal working directory.
To reproduce locally before copying it to the server:

```bash
git show 57a68c3:cs336_systems/flash_attention.py > /tmp/flash_attention_baseline.py
```

The tuned code and regression tests are in commit `dbc13ca`. A reviewed tuned
standalone copy is maintained at
`/ai/lmfs2026/sunyuquan/assignment2-codex/sunyuquan.py`.
Formal shared-directory hand-in has not been performed.

## Raw evidence and remaining work

[Raw JSON/JSONL measurements](benchmarks/2026-11-09/) include every tile trial,
three-sample comparisons, and saved-reference accuracy checks. The initial
first-head A check is included for provenance; it is superseded by the full
24-head result when discussing A's accuracy tier.

Still unmeasured: full 64-head B and 256-head C accuracy, their device memory
peaks, official efficiency scores, NCCL multi-GPU execution, and leaderboard
training. Do not extrapolate the measured speedup into an official score or
claim the largest workload is OOM-free.
