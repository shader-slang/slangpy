# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Opt-in integration test for the optional standalone CUDA compiler experiment.

Build with the optional compiler roots and set SLANGPY_TEST_CUBIN_CUDA_PATH to a
matching NVRTC toolkit before running. SLANGPY_TEST_CUBIN_COMPILERS selects a
comma-separated list of built backends. Each check starts a fresh process.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
COMPILERS = os.environ.get("SLANGPY_TEST_CUBIN_COMPILERS", "nvptxcompiler").split(",")


@pytest.mark.skipif(
    not os.environ.get("SLANGPY_TEST_CUBIN_CUDA_PATH"),
    reason="Requires optional CUBIN compiler build and matching NVRTC toolkit",
)
@pytest.mark.parametrize("compiler", COMPILERS)
def test_cubin_persistent_cache(tmp_path: Path, compiler: str) -> None:
    def run(
        mode: str, cubin: bool = True, level: int = 3, rounds: int = 4, backend: str = compiler
    ) -> dict[str, Any]:
        command = [
            sys.executable,
            str(ROOT / "tools/benchmark_pipeline_compilation.py"),
            "--backend",
            "cuda",
            "--worker",
            mode,
            "--cuda-path",
            os.environ["SLANGPY_TEST_CUBIN_CUDA_PATH"],
            "--nvrtc-no-cache",
            "--variants",
            "1",
            "--rounds",
            str(rounds),
            "--cache-path",
            str(tmp_path / "cache"),
        ]
        if cubin:
            command += ["--cubin", "--cubin-opt-level", str(level), "--cubin-compiler", backend]
        env = dict(os.environ, CUDA_CACHE_DISABLE="1")
        completed = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        sample = next(
            json.loads(line.removeprefix("BENCHMARK_JSON="))
            for line in completed.stdout.splitlines()
            if line.startswith("BENCHMARK_JSON=")
        )
        trace = [
            json.loads(line.removeprefix("CUBIN_TRACE="))
            for line in completed.stderr.splitlines()
            if line.startswith("CUBIN_TRACE=")
        ]
        assert len(trace) == (3 if cubin else 0)
        assert all(item["success"] for item in trace)
        assert all(
            item["backend"] == backend and item["intermediate_cache_disabled"] for item in trace
        )
        if sample["nvjitlink"]:
            assert all(item["fingerprint"] == sample["nvjitlink"]["fingerprint"] for item in trace)
        sample["cubin_hits"] = sum(item["cache_hit"] for item in trace)
        return sample

    cold = run("pending_dispatch")
    assert cold["cached_entry_points"] == 0
    assert cold["cubin_hits"] == 0
    for mode in ("serial", "command_list", "pending", "pending_dispatch"):
        warm = run(mode)
        assert warm["cached_entry_points"] == 3
        assert warm["cubin_hits"] == 3
        np.testing.assert_array_equal(warm["outputs"], cold["outputs"])

    # PTX is unchanged, but a compiler option change must miss the CUBIN cache.
    changed = run("pending_dispatch", level=0)
    assert changed["cached_entry_points"] == 3
    assert changed["cubin_hits"] == 0
    np.testing.assert_allclose(changed["outputs"], cold["outputs"], rtol=1e-5, atol=1e-6)
    assert run("pending_dispatch", level=0)["cubin_hits"] == 3
    assert run("pending_dispatch")["cubin_hits"] == 3

    changed_ptx = run("pending_dispatch", rounds=5)
    assert changed_ptx["cached_entry_points"] == 0
    assert changed_ptx["cubin_hits"] == 0
    changed_warm = run("pending_dispatch", rounds=5)
    assert changed_warm["cubin_hits"] == 3
    np.testing.assert_array_equal(changed_warm["outputs"], changed_ptx["outputs"])

    for other in COMPILERS:
        if other == compiler:
            continue
        switched = run("pending_dispatch", backend=other)
        assert switched["cached_entry_points"] == 3
        assert switched["cubin_hits"] == 0
        np.testing.assert_allclose(switched["outputs"], cold["outputs"], rtol=1e-5, atol=1e-6)
        assert run("pending_dispatch", backend=other)["cubin_hits"] == 3
        assert run("pending_dispatch")["cubin_hits"] == 3

    # The original PTX cache remains usable with CUBIN conversion disabled.
    ptx = run("serial", cubin=False)
    assert ptx["cached_entry_points"] == 3
    np.testing.assert_allclose(ptx["outputs"], cold["outputs"], rtol=1e-5, atol=1e-6)
