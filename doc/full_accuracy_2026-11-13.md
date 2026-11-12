# Full B/C correctness validation — November 13, 2026

## Outcome

**Both complete configurations passed an accepted accuracy tier without OOM.**
The exact submitted file was tested, using the existing official evaluator in
accuracy-only mode. The prior full A result and these full B/C results establish
an overall observed accuracy coefficient of **0.7**.

| Case | Shape H/N/D | Coverage | Worst passing atol | Coefficient | Peak allocated | Peak reserved | Wall time |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| B | 64 / 130000 / 128 | All 64 heads; O/dQ/dK/dV | 1e-2 | **0.7** | **37.696 GiB** | 37.768 GiB | 56.54 s |
| C | 256 / 131072 / 100 | All 256 heads; O/dQ per official reference format | 1e-3 | **1.0** | **44.094 GiB** | 44.176 GiB | 192.69 s |

The two evaluator calls took about **4 minutes 9 seconds combined**, excluding
preparation and the gap between calls. Wall time includes reference/input I/O,
C's seeded input reconstruction, initial compilation if needed, forward/backward,
and numerical comparisons. These are **not efficiency benchmark timings**.

## Numerical results

All checked outputs and gradients were finite. Comparisons use the official
`torch.allclose(..., atol=eps, rtol=0)` thresholds and FP32 reference arrays.
Student inputs use the submitted class's declared FP16 dtype.

| Case / result | Maximum absolute error | Passing atol |
| --- | ---: | ---: |
| B / O | 2.226885e-5 | 1e-3 |
| B / dQ | 6.112456e-5 | 1e-3 |
| B / dK | **1.463294e-3** | **1e-2** |
| B / dV | 7.860661e-4 | 1e-3 |
| C / O | 4.020706e-5 | 1e-3 |
| C / dQ | 5.677715e-5 | 1e-3 |

C computed all three input gradients. Its supplied NPZ/evaluator compares only O
and dQ; full-C dK/dV reference accuracy is therefore not claimed. All heads were
reconstructed with the evaluator's seed 2026 and its per-head CPU RNG convention.

A was previously checked on all 24 heads and passes 1e-2, also with coefficient
0.7. Thus `min(A=0.7,B=0.7,C=1.0)=0.7`. Efficiency points and any penalties in
future official grading remain separate; this does not establish a final grade.
No OOM occurred in these validation runs.

## Environment, availability, and integrity

- GPU: NVIDIA A800 80GB PCIe, CUDA index 0, UUID
  `GPU-bf236db6-0b88-f20e-a1d2-a9a06fce37b1`.
- At 04:39 CST, it had 0 MiB allocated, 0% utilization, and no compute processes.
  It was checked empty again before C. Runs were sequential, with one process per
  case and no efficiency repetitions.
- Existing runtime: PyTorch 2.11.0+cu128 / Triton 3.6.0.
- Official evaluator: `/lmfs2026/data/evaluate_fa2_new.py`.
- Tested submission: `/ai/lmfs2026/data/assignment2/sunyuquan.py`.
- Submission SHA-256:
  `f7dac7738fdd8535728b7fde99c618836aaee081fe2289c94ac06bc93504be67`.
- Evaluator SHA-256:
  `5807dd0a205af62ddd00cf3a0e5fa143eb7430587b2765ee1bb9959a8bbd8b6d`.

Both hashes were checked before/after execution and remained unchanged. Shared
reference files and environment dependencies were not modified. Our evaluator
processes and device monitors exited. A later availability check found another
GPU process using about 74.5 GiB at 100% utilization; the GPU should not be assumed
free for further work.

## Reproduction and memory measurement

The new wrapper imports the official evaluator unchanged and calls its
`run_single()` with `mode='accuracy'`, the selected full configuration, no legacy
interface, and no `torch.compile` wrapper. It resets PyTorch peak counters before
the call and reports peak allocated/reserved bytes after return. It performs no
configuration loop or repeated efficiency timing. The wrapper's return status
and the printed accuracy coefficients are both checked; exit code alone does
not establish the numerical tier.

```bash
source /lmfs2026/env/a1/bin/activate
cd /ai/lmfs2026/sunyuquan/assignment2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
# Check the GPU is available before either command.
python scripts/run_course_accuracy.py --configuration B
python scripts/run_course_accuracy.py --configuration C
```

Each remote invocation had a 900-second timeout. A separate `nvidia-smi` monitor
sampled device allocation/utilization every five seconds and was stopped at
process exit. CSV samples are device-wide and coarse; the table above uses the
exact per-process PyTorch allocator peak counters, including comparison
temporaries, rather than the sampled device peak.

Raw evidence is in [benchmarks/2026-11-13](benchmarks/2026-11-13/): full evaluator
logs (`*-full-accuracy.txt`), parsed metrics (`*-result.json`), and device monitor
samples (`*-device-memory.csv`).

## Remaining work

The full A/B/C correctness coverage and full B/C single-run memory checks are now
complete for this submitted version. Official repeated efficiency measurements,
class ranking, and final grading remain unmeasured. No further kernel tuning or
submission replacement was performed in this test session.
