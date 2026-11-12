"""Run one full B/C correctness case with the unchanged official evaluator.

No efficiency mode, repetitions, or configuration sweep. Prints peak PyTorch
memory after the evaluator returns. Intended for an explicitly available GPU.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configuration", choices=["B", "C"], required=True)
    p.add_argument("--student", type=Path, default=Path("/ai/lmfs2026/data/assignment2/sunyuquan.py"))
    p.add_argument("--evaluator", type=Path, default=Path("/lmfs2026/data/evaluate_fa2_new.py"))
    a = p.parse_args()
    student_hash = hashlib.sha256(a.student.read_bytes()).hexdigest()
    evaluator_hash = hashlib.sha256(a.evaluator.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("course_official_evaluator", a.evaluator)
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluator
    spec.loader.exec_module(evaluator)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    args = SimpleNamespace(student=str(a.student), legacy=False, compile=False, device="cuda:0", index={"B": 1, "C": 2}[a.configuration], mode="accuracy", warmup=0, rep=0)
    start = perf_counter()
    code = 1
    try:
        code = evaluator.run_single(args)
    finally:
        result = {
            "configuration": a.configuration,
            "exit_code": code,
            "wall_seconds": perf_counter() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "student_sha256": student_hash,
            "evaluator_sha256": evaluator_hash,
        }
        print("RESULT_METADATA=" + json.dumps(result), flush=True)
    assert student_hash == hashlib.sha256(a.student.read_bytes()).hexdigest()
    assert evaluator_hash == hashlib.sha256(a.evaluator.read_bytes()).hexdigest()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
