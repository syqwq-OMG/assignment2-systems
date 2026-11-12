# ECNU Assignment 2 submission audit — November 13, 2026

## Outcome

**The required code is implemented and the standalone file has been placed in the
course's specified submission directory. Full evaluation coverage is incomplete.**
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
| Single 80GB GPU support for every complete configuration | Analytically planned; **full B/C peak memory is not measured** |

The submission is standalone: it imports only standard-library modules, PyTorch,
and Triton, without relative project imports. Static syntax/interface checks and
an importlib-based, three-argument FP16 forward/backward smoke test passed on
November 13. That smoke test used `CUDA_VISIBLE_DEVICES=''` and
`TRITON_INTERPRET=1`, so it performed no GPU work.

## What is and is not finished

The November 9 real-GPU tests passed **29 attention cases**. The submitted SHA
matches that tested implementation; no numerical or kernel changes were made
for packaging. Full configuration A was checked against the original FP32 course
references, while B and C were each checked only on their first head.

| Configuration | Coverage | Observed result |
| --- | --- | --- |
| A `(24,16384,128)` | All 24 heads, O/dQ/dK/dV | Passes 1e-2; dK maximum error 0.001153 misses 1e-3 |
| B `(64,130000,128)` | First head only, O/dQ/dK/dV | Subset passes 1e-2 |
| C `(256,131072,100)` | First head only, O/dQ as in the embedded evaluator | Subset passes 1e-3 |

The course's full evaluator still needs to establish:

1. Full 64-head B and 256-head C correctness.
2. Full-shape GPU peak memory and absence of OOM.
3. Official efficiency measurements/ranking with the class's uniform settings.

The 1e-2 tier is accepted by the grading rules with an accuracy coefficient of
0.7. A alone limits the current overall coefficient to **at most 0.7**; unchecked
B/C results or OOM penalties could reduce the final score. Therefore, “submitted
implementation” is accurate; “all evaluation cases verified” or “full marks” is
not. The measured 152.1ms A-shaped attention benchmark is a development result,
not an official class efficiency score.

No large GPU evaluations were run on November 13, since the documented personal
reservation had ended and the earlier request called for bounded GPU usage.

## Directory cleanup

The personal project working directory was renamed to `assignment2`, and project
documentation references were updated. No project filenames containing the old
assistant-specific name remained in the renamed working tree. Shared course
paths, the GitHub repository name, and account-wide tool/configuration directories
were not renamed. The old working-directory path is no longer used.

See [the GPU report](gpu_optimization_2026-11-09.md) and its raw measurements for
the prior numerical/performance evidence.
