# 第二次实践作业：FlashAttention-2
参考 Stanford CS336 Spring 2026 Assignment 2（Systems）

+ 官方仓库：[https://github.com/stanford-cs336/assignment2-systems](https://github.com/stanford-cs336/assignment2-systems)
+ 任务详情：[https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf](https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf)
+ Assignment 分数：15 分
+ 作业类型：个人作业
+ 评测方式：统一评测脚本

## 一、任务
**实现 FlashAttention-2：**

+ Forward：必须使用 Triton 实现。
+ Backward：必须使用 Triton 实现，不允许使用 PyTorch 常规计算代替。
+ 本课程统一使用 `causal=False`（非因果注意力）。
+ 使用分块计算和数值稳定的 Softmax，不得显式构造完整的 (N\times N) 注意力矩阵。
+ 具体算法与实现方式参考 CS336 官方 Assignment 2 中 FlashAttention 相关内容。

**代码框架：**

```python
import torch

class FlashAttention2(torch.autograd.Function):
    INPUT_DTYPE = torch.float32   #可自行选择数值类型，不设置默认为torch.float32
    @staticmethod
    def forward(ctx, q, k, v):
        # Triton forward
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out):
        # Triton backward
        return grad_q, grad_k, grad_v
```

统一使用 `FlashAttention2.apply(q, k, v)`，其中 `q/k/v` 的形状为 `[n_heads, sequence_length, d_head]`。（不额外传入 causal 参数）

## 二、开发机环境
本次实验使用与第一次实践作业相同的公共 Python 环境，已配置 PyTorch 2.11.0+cu128、CUDA 12.8 和 Triton 3.6.0，GPU 为 NVIDIA A100 80GB。

登录开发机后执行：

```plain
source /lmfs2026/env/a1/bin/activate
```

检查环境：

```plain
python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)"
```

本次实验无需额外安装依赖，不需要手动配置 `CUDA_HOME` 或 CUDA 相关环境变量。

注意：公共环境为多人共享，不得自行安装、升级或修改公共环境中的依赖。

**2. 实验目录**

**官方作业代码及测试数据已统一存放在开发机系统盘：**

+ **官方代码：**`**/lmfs2026/data/assignment2-systems**`
+ **测评数据：**`**/lmfs2026/data/2-fa2-data**`

学生代码保存在个人目录：

```bash
mkdir -p /lmfs2026/$USER/assignment2
cd /lmfs2026/$USER/assignment2
```

**公共代码和测试数据仅供读取，不得修改或删除。学生可参考官方代码框架，在个人目录中完成实验。**

## 三、评测
本次实验通过统一评测脚本进行数值正确性与运行效率评估。,

| 组别 | `H, SL, D` | 正确性 Accuracy | 效率 Efficiency |
| --- | --- | --- | --- |
| A | `(24, 16384, 128)` | 进行 | 进行 |
| B | `(64, 130000, 128)` | 进行 | 进行 |
| C | `(256, 131072, 100)` | 进行 | 进行 |


**1. 数值正确性**

评测脚本比较学生实现与参考实现的前向输出 `o`，以及反向梯度 `grad_q`、`grad_k`、`grad_v`。

本课程统一使用 `causal=False`，反向传播的测试损失为：

```python
loss = o.sum()
loss.backward()
```

测试数据存放于 `/lmfs2026/data/2-fa2-data/*.npz`：

<img src="https://cdn.nlark.com/yuque/0/2026/png/54421635/1793459977700-35971e64-fa2b-40ea-95b8-28eaf44f4198.png" width="807" title="" crop="0,0,1,1" id="u57fc7a67" class="ne-image">

输入为 `float32`，每份数据包含 `q`、`k`、`v`、`o`、`grad_q`、`grad_k`、`grad_v`。

数值比较采用：

```python
torch.allclose(actual, expected, atol=eps, rtol=0)
```

精度系数：

| 精度要求 | 系数 |
| --- | ---: |
| `atol=1e-3` 全部通过 | 1.0 |
| `atol=1e-2` 全部通过 | 0.7 |
| `atol=1e-1` 全部通过 | 0.4 |
| 以上均未通过 | 0.2 |


以前向输出和三个反向梯度在全部测试配置中的最差精度等级为准。

**2. 运行效率**

使用统一评测脚本测试 Forward + Backward 的运行时间。

使用 Triton Benchmark 进行预热与重复计时，正式评测统一运行参数，记录稳定运行时间，不计入首次编译时间。

_**<font style="color:#DF2A3F;">注意：</font>**_

+ 学生实现必须能够在单张 A100 80GB GPU 上完成规定的前向、反向及性能测试。
+ 若正式评测过程中，显存需求超过 80GB 而发生 OOM（Out of Memory）：每发生一次OOM，本次实验成绩乘OOM系数0.5。即（一次：OOM系数=0.5；两次：OOM系数=0.25.....）
+ 根据全班正常结束测评同学的运行时间排名分配效率分

**最终得分 = 效率分 × 数值精度系数 x OOM系数 （满分 15 分）**

**统一评测脚本：**

```python

import argparse
import importlib.util
import json
import subprocess
import sys
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import triton
import triton.testing


# ============================================================
# FlashAttention-2 Evaluation
#
# A/B accuracy:
#   Original seven-array NPZ format.
#   Check O, dQ, dK, dV.
#
# C accuracy:
#   Seed-reconstructed Q/K/V.
#   NPZ contains only:
#       o.npy
#       grad_q.npy
#   Check O and dQ only.
#   Stream reference data one head at a time.
#
# Efficiency:
#   UNCHANGED for A/B/C.
#
# Student input dtype:
#   FlashAttention2.INPUT_DTYPE if declared.
#   Otherwise torch.float32.
#
# Accuracy and efficiency use the SAME input dtype.
# ============================================================


DATA_DIR = Path("/lmfs2026/data/2-fa2-data")

CONFIGS = [
    (24, 16384, 128),
    (64, 130000, 128),
    (256, 131072, 100),
]

# Third configuration uses seed-reconstructed inputs.
C_CONFIG_INDEX = 2
C_CONFIG = CONFIGS[C_CONFIG_INDEX]

# Must match generate_fa2_data.py.
C_DEFAULT_SEED = 2026

ATOLS = (1e-3, 1e-2, 1e-1)

ACCURACY_COEFFICIENTS = {
    1e-3: 1.0,
    1e-2: 0.7,
    1e-1: 0.4,
}

FAILED_ACCURACY_COEFFICIENT = 0.2


# ============================================================
# Student loader
# ============================================================

def load_student(student_path):
    """
    Load student's FlashAttention2 class.

    INPUT_DTYPE is optional.
    Default input dtype: torch.float32.
    """
    path = Path(student_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Student file not found: {path}"
        )

    sys.path.insert(0, str(path.parent))

    spec = importlib.util.spec_from_file_location(
        "fa2_student_submission",
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Failed to import student file: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    if not hasattr(module, "FlashAttention2"):
        raise ValueError(
            "Student file must define class FlashAttention2"
        )

    cls = module.FlashAttention2

    dtype = getattr(
        cls,
        "INPUT_DTYPE",
        torch.float32,
    )

    if not isinstance(dtype, torch.dtype):
        raise TypeError(
            f"INPUT_DTYPE must be a torch.dtype, "
            f"got {dtype!r}"
        )

    if not dtype.is_floating_point:
        raise TypeError(
            f"INPUT_DTYPE must be floating point, "
            f"got {dtype}"
        )

    print(
        f"Student INPUT_DTYPE: {dtype}",
        flush=True,
    )

    if not hasattr(cls, "INPUT_DTYPE"):
        print(
            "INPUT_DTYPE not declared; "
            "using default torch.float32",
            flush=True,
        )

    return cls, dtype


def make_flash(cls, legacy=False):
    """
    Current:
        FlashAttention2.apply(q, k, v)

    Legacy:
        FlashAttention2.apply(q, k, v, False)
    """

    def flash(q, k, v):
        if legacy:
            return cls.apply(q, k, v, False)

        return cls.apply(q, k, v)

    return flash


def npz_path(h, n, d):
    return DATA_DIR / (
        f"fa2_nheads{h}_seq{n}_dhead{d}.npz"
    )


# ============================================================
# Shared accuracy reporting
# ============================================================

def accuracy_coefficient(passed_atol):
    if passed_atol is None:
        return FAILED_ACCURACY_COEFFICIENT

    return ACCURACY_COEFFICIENTS[passed_atol]


def report_accuracy_item(
    key,
    max_error,
    passed_atol,
):
    coefficient = accuracy_coefficient(
        passed_atol
    )

    print(
        f"  {key:7s} "
        f"max_error={max_error:.6e} "
        f"atol={passed_atol} "
        f"coefficient={coefficient}",
        flush=True,
    )

    return coefficient


def report_accuracy_final(
    coefficients,
    passed_atols,
):
    final_coefficient = min(coefficients)

    if None in passed_atols:
        print(
            "  Worst atol: >0.1",
            flush=True,
        )
    else:
        print(
            f"  Worst atol: {max(passed_atols)}",
            flush=True,
        )

    print(
        f"ACCURACY_COEFFICIENT: {final_coefficient}",
        flush=True,
    )

    return final_coefficient


# ============================================================
# A/B accuracy
#
# Keep original seven-array NPZ evaluation unchanged.
# ============================================================

def accuracy_test_ab(
    flash,
    dtype,
    config,
    device,
):
    """
    Correctness for A/B:
      1. Read original seven-array NPZ.
      2. Convert inputs to student dtype.
      3. Run forward/backward.
      4. Compare all four results with FP32 references.
      5. Score by worst atol.

    Identical input dtype to efficiency.
    """
    h, n, d = config
    path = npz_path(h, n, d)

    print(
        f"\n[ACCURACY] H={h} N={n} D={d}",
        flush=True,
    )
    print(f"  Input dtype: {dtype}", flush=True)
    print(
        "  Reference format: original seven-array NPZ",
        flush=True,
    )

    if not path.is_file():
        print(
            f"  SKIP: missing NPZ: {path}",
            flush=True,
        )
        return None

    keys = (
        "q", "k", "v", "o",
        "grad_q", "grad_k", "grad_v",
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        missing = set(keys) - set(data.files)

        if missing:
            raise ValueError(
                f"NPZ missing keys: {sorted(missing)}"
            )

        q = torch.from_numpy(
            data["q"].copy()
        ).to(
            device=device,
            dtype=dtype,
        ).requires_grad_(True)

        k = torch.from_numpy(
            data["k"].copy()
        ).to(
            device=device,
            dtype=dtype,
        ).requires_grad_(True)

        v = torch.from_numpy(
            data["v"].copy()
        ).to(
            device=device,
            dtype=dtype,
        ).requires_grad_(True)

        output = flash(q, k, v)
        output.sum().backward()

        torch.cuda.synchronize(device)

        actual = {
            "o": output.detach(),
            "grad_q": q.grad,
            "grad_k": k.grad,
            "grad_v": v.grad,
        }

        coefficients = []
        worst_atols = []

        for key in (
            "o",
            "grad_q",
            "grad_k",
            "grad_v",
        ):
            result = actual[key]

            if result is None:
                raise RuntimeError(
                    f"Missing output or gradient: {key}"
                )

            result = result.detach().float()

            reference = torch.from_numpy(
                data[key].copy()
            ).to(
                device=device,
                dtype=torch.float32,
            )

            if result.shape != reference.shape:
                raise RuntimeError(
                    f"{key} shape mismatch: "
                    f"{tuple(result.shape)} vs "
                    f"{tuple(reference.shape)}"
                )

            if not bool(
                torch.isfinite(result).all().item()
            ):
                raise RuntimeError(
                    f"{key} contains NaN/Inf"
                )

            max_error = (
                result - reference
            ).abs().max().item()

            passed_atol = None

            for atol in ATOLS:
                if torch.allclose(
                    result,
                    reference,
                    atol=atol,
                    rtol=0,
                ):
                    passed_atol = atol
                    break

            coefficient = report_accuracy_item(
                key,
                max_error,
                passed_atol,
            )

            coefficients.append(coefficient)
            worst_atols.append(passed_atol)

            del reference

        return report_accuracy_final(
            coefficients,
            worst_atols,
        )


# ============================================================
# C: Fixed-seed Q/K/V regeneration
#
# This MUST match generate_fa2_data.py exactly.
# ============================================================

def generate_qkv_one_head(
    head_id,
    n,
    d,
    seed,
    device,
    dtype,
):
    """
    Match generator's random sequence exactly:

      CPU Generator
      manual_seed(seed + head_id)
      FP32 q, then FP32 k, then FP32 v

    Only after generating do we cast to the student's
    declared input dtype.

    Note:
      Do not replace with one large torch.randn call.
      Do not generate directly in BF16/FP16.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + head_id)

    q_cpu = torch.randn(
        (n, d),
        generator=generator,
        dtype=torch.float32,
    )

    k_cpu = torch.randn(
        (n, d),
        generator=generator,
        dtype=torch.float32,
    )

    v_cpu = torch.randn(
        (n, d),
        generator=generator,
        dtype=torch.float32,
    )

    q_head = q_cpu.to(
        device=device,
        dtype=dtype,
    )

    k_head = k_cpu.to(
        device=device,
        dtype=dtype,
    )

    v_head = v_cpu.to(
        device=device,
        dtype=dtype,
    )

    return q_head, k_head, v_head


def reconstruct_c_qkv(
    h,
    n,
    d,
    seed,
    device,
    dtype,
):
    """
    Reconstruct full [H,N,D] Q/K/V on CUDA,
    writing each Head into the preallocated tensors.

    Avoid holding full Q/K/V on CPU.

    Do not enable requires_grad until all Head copies
    are finished.
    """
    print(
        "  Reconstructing C Q/K/V "
        "from fixed seed...",
        flush=True,
    )

    q = torch.empty(
        (h, n, d),
        device=device,
        dtype=dtype,
    )

    k = torch.empty_like(q)
    v = torch.empty_like(q)

    with torch.no_grad():
        for head_id in range(h):
            q_head, k_head, v_head = (
                generate_qkv_one_head(
                    head_id=head_id,
                    n=n,
                    d=d,
                    seed=seed,
                    device=device,
                    dtype=dtype,
                )
            )

            q[head_id].copy_(q_head)
            k[head_id].copy_(k_head)
            v[head_id].copy_(v_head)

            del q_head, k_head, v_head

            if (
                (head_id + 1) % 32 == 0
                or head_id + 1 == h
            ):
                print(
                    f"  Reconstructed "
                    f"{head_id + 1}/{h} heads",
                    flush=True,
                )

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    return q, k, v


# ============================================================
# C: Large NPZ streaming utilities
# ============================================================

def read_npy_header(stream):
    """
    Read a .npy header from an open ZIP member.

    Does not load the array data into memory.
    """
    version = np.lib.format.read_magic(stream)

    if version == (1, 0):
        shape, fortran_order, arr_dtype = (
            np.lib.format.read_array_header_1_0(
                stream
            )
        )

    elif version in ((2, 0), (3, 0)):
        shape, fortran_order, arr_dtype = (
            np.lib.format.read_array_header_2_0(
                stream
            )
        )

    else:
        raise ValueError(
            f"Unsupported NPY version: {version}"
        )

    return shape, fortran_order, np.dtype(
        arr_dtype
    )


def validate_c_npy_stream(
    stream,
    name,
    expected_shape,
):
    """
    Check header only; do not read full array.
    """
    shape, fortran_order, arr_dtype = (
        read_npy_header(stream)
    )

    if tuple(shape) != tuple(expected_shape):
        raise ValueError(
            f"{name} shape mismatch: "
            f"{shape} vs {expected_shape}"
        )

    if fortran_order:
        raise ValueError(
            f"{name} must be C-contiguous "
            "in NPZ"
        )

    if arr_dtype != np.dtype("float32"):
        raise ValueError(
            f"{name} must be FP32; "
            f"got {arr_dtype}"
        )


def read_one_head_from_stream(
    stream,
    n,
    d,
    name,
    head_id,
):
    """
    Read one FP32 [N,D] Head from the NPZ stream.

    Memory consumption is O(N*D), not O(H*N*D).
    """
    num_elements = n * d
    num_bytes = num_elements * 4

    raw = stream.read(num_bytes)

    if len(raw) != num_bytes:
        raise EOFError(
            f"Incomplete {name}, head={head_id}: "
            f"read {len(raw)} bytes, "
            f"expected {num_bytes}"
        )

    # copy() gives writable, owned NumPy storage.
    result = np.frombuffer(
        raw,
        dtype=np.float32,
    ).copy().reshape(n, d)

    return result


def load_c_metadata(zf, expected_shape):
    """
    The C NPZ generator stores metadata in ZIP comment.

    Validate reproducibility-critical fields.
    """
    if not zf.comment:
        raise ValueError(
            "C NPZ is missing seed metadata "
            "in ZIP comment. Regenerate with "
            "the matching generator."
        )

    try:
        metadata = json.loads(
            zf.comment.decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(
            f"Invalid C NPZ metadata: {e}"
        ) from e

    if tuple(metadata.get("shape", ())) != (
        tuple(expected_shape)
    ):
        raise ValueError(
            "C NPZ shape metadata mismatch: "
            f"{metadata.get('shape')}"
        )

    seed = metadata.get("seed")

    if type(seed) is not int or seed < 0:
        raise ValueError(
            f"Invalid C NPZ seed: {seed!r}"
        )

    if metadata.get("qkv_generator") != (
        "torch.CPU.Generator"
    ):
        raise ValueError(
            "C NPZ QKV generator does not match "
            "the evaluator."
        )

    if metadata.get("head_seed_rule") != (
        "seed + head_id"
    ):
        raise ValueError(
            "C NPZ head seed rule does not match "
            "the evaluator."
        )

    if metadata.get("input_dtype") != "float32":
        raise ValueError(
            "C NPZ must record FP32 generated inputs"
        )

    if metadata.get("reference_dtype") != "float32":
        raise ValueError(
            "C NPZ references must be FP32"
        )

    if metadata.get("loss") != "o.sum()":
        raise ValueError(
            "C NPZ loss convention must be o.sum()"
        )

    return seed


def compare_c_streaming(
    actual,
    zf,
    h,
    n,
    d,
    device,
):
    """
    Compare C O and dQ head-by-head.

    Uses separate ZIP member streams:
        o.npy
        grad_q.npy

    Does NOT load the complete 24 GiB reference
    into host RAM or GPU VRAM.

    Each metric passes an atol only if ALL heads
    pass that threshold.
    """
    keys = ("o", "grad_q")

    max_errors = {
        key: 0.0
        for key in keys
    }

    threshold_passed = {
        key: {
            atol: True
            for atol in ATOLS
        }
        for key in keys
    }

    with ExitStack() as stack:
        streams = {}

        for key in keys:
            stream = stack.enter_context(
                zf.open(f"{key}.npy", "r")
            )

            validate_c_npy_stream(
                stream=stream,
                name=key,
                expected_shape=(h, n, d),
            )

            streams[key] = stream

        for head_id in range(h):
            for key in keys:
                result = actual[key]

                if result is None:
                    raise RuntimeError(
                        f"Missing C result: {key}"
                    )

                ref_np = read_one_head_from_stream(
                    stream=streams[key],
                    n=n,
                    d=d,
                    name=key,
                    head_id=head_id,
                )

                reference = torch.from_numpy(
                    ref_np
                ).to(
                    device=device,
                    dtype=torch.float32,
                )

                result_head = (
                    result[head_id]
                    .detach()
                    .float()
                )

                if (
                    tuple(result_head.shape)
                    != (n, d)
                ):
                    raise RuntimeError(
                        f"{key} head={head_id} "
                        f"shape mismatch: "
                        f"{tuple(result_head.shape)}"
                    )

                if not bool(
                    torch.isfinite(
                        result_head
                    ).all().item()
                ):
                    raise RuntimeError(
                        f"C {key}, head={head_id} "
                        "contains NaN/Inf"
                    )

                error = (
                    result_head - reference
                ).abs()

                head_max_error = (
                    error.max().item()
                )

                max_errors[key] = max(
                    max_errors[key],
                    head_max_error,
                )

                for atol in ATOLS:
                    if threshold_passed[key][atol]:
                        if not torch.allclose(
                            result_head,
                            reference,
                            atol=atol,
                            rtol=0,
                        ):
                            threshold_passed[key][atol] = False

                del (
                    ref_np,
                    reference,
                    result_head,
                    error,
                )

            if (
                (head_id + 1) % 32 == 0
                or head_id + 1 == h
            ):
                print(
                    f"  Compared "
                    f"{head_id + 1}/{h} heads",
                    flush=True,
                )

    coefficients = []
    worst_atols = []

    for key in keys:
        passed_atol = None

        for atol in ATOLS:
            if threshold_passed[key][atol]:
                passed_atol = atol
                break

        coefficient = report_accuracy_item(
            key=key,
            max_error=max_errors[key],
            passed_atol=passed_atol,
        )

        coefficients.append(coefficient)
        worst_atols.append(passed_atol)

    return report_accuracy_final(
        coefficients,
        worst_atols,
    )


# ============================================================
# C accuracy:
# fixed seed + O/dQ only
# ============================================================

def accuracy_test_c(
    flash,
    dtype,
    config,
    device,
):
    h, n, d = config
    path = npz_path(h, n, d)

    print(
        f"\n[ACCURACY] H={h} N={n} D={d}",
        flush=True,
    )
    print(f"  Input dtype: {dtype}", flush=True)
    print(
        "  Reference format: "
        "seed-reconstructed QKV + O/dQ only",
        flush=True,
    )

    if not path.is_file():
        print(
            f"  SKIP: missing NPZ: {path}",
            flush=True,
        )
        return None

    with zipfile.ZipFile(path, mode="r") as zf:
        required = {
            "o.npy",
            "grad_q.npy",
        }

        missing = required - set(
            zf.namelist()
        )

        if missing:
            raise ValueError(
                "C NPZ missing members: "
                f"{sorted(missing)}"
            )

        seed = load_c_metadata(
            zf=zf,
            expected_shape=config,
        )

        print(
            f"  QKV generator seed: {seed}",
            flush=True,
        )

        print(
            "  Reference arrays: O, dQ (FP32)",
            flush=True,
        )

        q, k, v = reconstruct_c_qkv(
            h=h,
            n=n,
            d=d,
            seed=seed,
            device=device,
            dtype=dtype,
        )

        print(
            "  Running full C Forward...",
            flush=True,
        )

        output = flash(q, k, v)

        print(
            "  Running full C Backward...",
            flush=True,
        )

        output.sum().backward()

        torch.cuda.synchronize(device)

        print(
            "  C Forward + Backward completed",
            flush=True,
        )

        actual = {
            "o": output.detach(),
            "grad_q": q.grad,
        }

        if actual["grad_q"] is None:
            raise RuntimeError(
                "Student implementation did not "
                "produce dQ"
            )

        if tuple(output.shape) != config:
            raise RuntimeError(
                f"C output shape mismatch: "
                f"{tuple(output.shape)} vs {config}"
            )

        if tuple(q.grad.shape) != config:
            raise RuntimeError(
                f"C dQ shape mismatch: "
                f"{tuple(q.grad.shape)} vs {config}"
            )

        print(
            "  Comparing O/dQ against NPZ "
            "one head at a time...",
            flush=True,
        )

        return compare_c_streaming(
            actual=actual,
            zf=zf,
            h=h,
            n=n,
            d=d,
            device=device,
        )


# ============================================================
# Accuracy dispatcher
#
# A/B remains original.
# Only C uses the new reference format.
# ============================================================

def accuracy_test(
    flash,
    dtype,
    config,
    device,
):
    if tuple(config) == tuple(C_CONFIG):
        return accuracy_test_c(
            flash,
            dtype,
            config,
            device,
        )

    return accuracy_test_ab(
        flash,
        dtype,
        config,
        device,
    )


# ============================================================
# Efficiency Test
#
# UNCHANGED from original evaluator.
# ============================================================

def efficiency_test(
    flash,
    dtype,
    config,
    device,
    warmup,
    rep,
):
    """
    Efficiency:
      - Use exactly the same declared input dtype.
      - Run forward + backward.
      - Measure with triton.testing.do_bench.
      - Report PyTorch peak allocated/reserved memory.
    """
    h, n, d = config

    print(
        f"\n[EFFICIENCY] H={h} N={n} D={d}",
        flush=True,
    )
    print(f"  Input dtype: {dtype}", flush=True)

    q = torch.randn(
        h, n, d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    k = torch.randn(
        h, n, d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    v = torch.randn(
        h, n, d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    def forward_backward():
        q.grad = None
        k.grad = None
        v.grad = None

        output = flash(q, k, v)
        output.sum().backward()

    # Trigger initial compilation before measurement.
    forward_backward()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)

    ms = triton.testing.do_bench(
        forward_backward,
        warmup=warmup,
        rep=rep,
    )

    torch.cuda.synchronize(device)

    peak_allocated = (
        torch.cuda.max_memory_allocated(device)
        / 1024**3
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved(device)
        / 1024**3
    )

    print(
        f"  Time: {ms:.4f} ms",
        flush=True,
    )

    print(
        f"  PyTorch peak allocated: "
        f"{peak_allocated:.3f} GiB",
        flush=True,
    )

    print(
        f"  PyTorch peak reserved:  "
        f"{peak_reserved:.3f} GiB",
        flush=True,
    )

    return ms


# ============================================================
# Single configuration execution
# ============================================================

def run_single(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    config = CONFIGS[args.index]

    cls, dtype = load_student(
        args.student
    )

    flash = make_flash(
        cls,
        legacy=args.legacy,
    )

    if args.compile:
        flash = torch.compile(flash)

    print(
        f"GPU: {torch.cuda.get_device_name(device)}",
        flush=True,
    )

    print(
        f"Configuration: {config}",
        flush=True,
    )

    try:
        if args.mode in (
            "accuracy",
            "all",
        ):
            accuracy_test(
                flash,
                dtype,
                config,
                device,
            )

        if args.mode in (
            "efficiency",
            "all",
        ):
            efficiency_test(
                flash,
                dtype,
                config,
                device,
                args.warmup,
                args.rep,
            )

    except torch.cuda.OutOfMemoryError as e:
        print(
            f"CUDA OOM: {e}",
            flush=True,
        )
        return 2

    return 0


# ============================================================
# Parent process:
# each configuration runs in a separate subprocess.
# ============================================================

def run_parent(args):
    print("=" * 65)
    print("FlashAttention-2 Evaluation")
    print(f"Student: {args.student}")
    print(f"GPU: {args.device}")
    print(f"Mode: {args.mode}")
    print(
        "Input dtype: student INPUT_DTYPE "
        "(default torch.float32)"
    )
    print("=" * 65, flush=True)

    if args.index is None:
        indices = range(
            len(CONFIGS)
        )
    else:
        indices = [args.index]

    exit_codes = []

    for i in indices:
        print("\n" + "=" * 65)
        print(
            f"CONFIG {i + 1}: {CONFIGS[i]}"
        )
        print("=" * 65, flush=True)

        cmd = [
            sys.executable,
            str(
                Path(__file__).resolve()
            ),
            "--student",
            str(args.student),
            "--mode",
            args.mode,
            "--device",
            args.device,
            "--warmup",
            str(args.warmup),
            "--rep",
            str(args.rep),
            "--index",
            str(i),
            "--child",
        ]

        if args.legacy:
            cmd.append("--legacy")

        if args.compile:
            cmd.append("--compile")

        result = subprocess.run(
            cmd,
            check=False,
        )

        exit_codes.append(
            result.returncode
        )

        print(
            f"CONFIG {i + 1} EXIT CODE: "
            f"{result.returncode}",
            flush=True,
        )

    print(
        "\n评测结束。",
        flush=True,
    )

    if any(
        code != 0
        for code in exit_codes
    ):
        print(
            "存在运行失败的配置，请检查日志。",
            flush=True,
        )
        return 1

    print(
        "全部配置已完成运行。"
        "效率排名需结合全班成绩计算。",
        flush=True,
    )

    return 0


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--student",
        required=True,
        help="Path to student FA2 Python file",
    )

    parser.add_argument(
        "--mode",
        choices=(
            "accuracy",
            "efficiency",
            "all",
        ),
        default="all",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--rep",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--index",
        type=int,
        choices=range(
            len(CONFIGS)
        ),
        default=None,
        help=(
            "0=A, 1=B, 2=C; "
            "omit to test all"
        ),
    )

    parser.add_argument(
        "--legacy",
        action="store_true",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
    )

    parser.add_argument(
        "--child",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if args.child:
        return run_single(args)

    return run_parent(args)


if __name__ == "__main__":
    sys.exit(main())


```

```bash
#准确性
python evaluate_fa2_new.py   --student *.py   --mode accuracy   --device cuda:0

#效率
python evaluate_fa2_new.py   --student *.py   --mode efficiency   --device cuda:0
```

## 四、提交
学生在目录 `/ai/lmfs2026/data/assignment2` 中提交个人命名的py文件（如 /ai/lmfs2026/data/assignment2/zhangsan.py），文件中包含 `FlashAttention2` 类及 Triton Forward、Backward 实现等

要求：

+ 使用统一接口 `FlashAttention2.apply(q, k, v)`。
+ 不得使用现成 FlashAttention、PyTorch SDPA 或其他完整注意力算子替代核心实现。
+ 不得修改公共测试数据及评测脚本。
