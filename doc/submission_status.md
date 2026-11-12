# ECNU Assignment 2 submission audit — November 13, 2026

## Outcome

**The required code is implemented and the standalone file has been placed in the
course's specified submission directory. Full A/B/C correctness coverage is now complete; official efficiency grading remains outstanding.**
This audit follows `doc/FA2.md`, including its embedded evaluator, rather than the
separate Stanford leaderboard or upstream zip/writeup workflow.

- Submitted file: `/ai/lmfs2026/data/assignment2/sunyuquan.py`.
- Personal working directory: `/ai/lmfs2026/sunyuquan/assignment2`.
- Local submission snapshot: [`../submission/sunyuquan.py`](../submission/sunyuquan.py).
- Implementation source: [`../cs336_systems/flash_attention.py`](../cs336_systems/flash_attention.py).
- SHA-256 of all copies: `f7dac7738fdd8535728b7fde99c618836aaee081fe2289c94ac06bc93504be67`.
- Size: 9,782 bytes. Remote submission owner: `sunyuquan` (UID 1008); permissions: `0644`.

The destination did not previously exist. The tested bytes were published
atomically without replacing another file, then read back and compared. This
confirms placement and contents; it is not an acknowledgement from an external
grading service.

## Requirement-by-requirement review

| Course requirement | Finding |
| --- | --- |
| A personally named Python file in `/ai/lmfs2026/data/assignment2` | Submitted as `sunyuquan.py` |
| Defines `FlashAttention2` | Present, subclass of `torch.autograd.Function` |
| Called as `FlashAttention2.apply(q,k,v)` | Supported; optional upstream causal flag defaults to False |
| Inputs `[H,N,D]`, noncausal attention | Supported, including the actual long sequence lengths in bounded head checks |
| Forward implemented in Triton | `_forward` kernel |
| Backward implemented in Triton | `_delta`, `_backward_q`, `_backward_kv` kernels |
| Stable tiled softmax | Online maximum/denominator recurrence with FP32 accumulators |
| No complete N-by-N attention allocation | Global state scales with HND/HN; score tiles stay inside kernels |
| No prebuilt FlashAttention/SDPA substitution | None in the submitted file |
| Selectable input dtype | Declares `INPUT_DTYPE=torch.float16`, which the evaluator explicitly permits |
| Public environment/data/evaluator unchanged | No shared dependency changes or reference-data/evaluator writes |
| Single 80GB GPU support for every complete configuration | Full B/C correctness runs completed without OOM on the available 80GB A800; peak allocation 37.696/44.094 GiB |

The submission is standalone: it imports only standard-library modules, PyTorch,
and Triton, without relative project imports. Static syntax/interface checks and
an importlib-based, three-argument FP16 forward/backward smoke test passed on
November 13. That smoke test used `CUDA_VISIBLE_DEVICES=''` and
`TRITON_INTERPRET=1`, so it performed no GPU work.

## What is and is not finished

The November 9 real-GPU tests passed **29 attention cases**. The submitted SHA
matches that tested implementation; no numerical or kernel changes were made
for packaging. Full configuration A was checked against the original FP32 course
references. On November 13, the unchanged official evaluator also checked every head of B and C against the exact submitted file.

| Configuration | Coverage | Observed result |
| --- | --- | --- |
| A `(24,16384,128)` | All 24 heads, O/dQ/dK/dV | Passes 1e-2; dK maximum error 0.001153 misses 1e-3 |
| B `(64,130000,128)` | All 64 heads, O/dQ/dK/dV | Passes 1e-2; coefficient 0.7; no OOM |
| C `(256,131072,100)` | All 256 heads, O/dQ per official reference format | Passes 1e-3; coefficient 1.0; no OOM |

The overall observed numerical accuracy coefficient is **0.7**, the minimum
across full A (0.7), B (0.7), and C (1.0). No OOM occurred in the full B/C
correctness runs. The remaining unmeasured part is official repeated efficiency
measurement/class ranking and final grading. The earlier 152.1ms A-shaped
attention benchmark remains a development result, not an official efficiency
score.

The first November 13 submission audit used only a CPU-interpreter smoke test.
After the user's explicit follow-up request and a fresh idle-GPU check, full B/C
correctness tests were run sequentially. They took 56.54 and 192.69 seconds,
including I/O and reference comparison. See [the full validation report](full_accuracy_2026-11-13.md)
for logs, hashes, exact memory peaks, and reproduction commands. The submitted
implementation was not changed.

## Directory cleanup

The personal project working directory was renamed to `assignment2`, and project
documentation references were updated. No project filenames containing the old
assistant-specific name remained in the renamed working tree. Shared course
paths, the GitHub repository name, and account-wide tool/configuration directories
were not renamed. The old working-directory path is no longer used.

See [the GPU report](gpu_optimization_2026-11-09.md) and its raw measurements for
the prior numerical/performance evidence.
