# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pytest

from slangpy import DeviceType
from slangpy.core.generator import (
    MAX_DISPATCH_THREAD_GROUPS_X,
    resolve_max_dispatch_groups_x,
)
from slangpy.experimental.gridarg import grid
from slangpy.testing import helpers

MODULE = r"""
float sq(int i) { return float(i * i); }
"""


def test_zero_dispatch_group_limit_is_unbounded() -> None:
    # A zero (unset) limit must resolve to the ceiling, not 0 -- a zero stride
    # would collapse the physical->logical group flattening.
    assert resolve_max_dispatch_groups_x(0) == MAX_DISPATCH_THREAD_GROUPS_X
    assert resolve_max_dispatch_groups_x(5) == 5
    assert (
        resolve_max_dispatch_groups_x(MAX_DISPATCH_THREAD_GROUPS_X * 4)
        == MAX_DISPATCH_THREAD_GROUPS_X
    )


@pytest.mark.parametrize("device_type", [DeviceType.cpu])
def test_cpu_dispatch_grid(device_type: DeviceType) -> None:
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    # Span many thread groups (> one group of 32) to exercise multi-group X indexing.
    n = 1000
    result = module.sq(grid(shape=(n,)), _result="numpy")

    expected = np.arange(n, dtype=np.float32) ** 2
    np.testing.assert_array_equal(result, expected)


# Runs a CPU dispatch the way an installed wheel does: nothing adds the package directory to the
# DLL search path, so Slang must be told where slang-llvm (the CPU target's compiler) is. The
# development build adds its build directory with os.add_dll_directory(), which hides the problem,
# so this script closes that directory again once slangpy is imported.
WHEEL_LAYOUT_CPU_SCRIPT = r"""
import os

added = []
add_dll_directory = os.add_dll_directory
os.add_dll_directory = lambda path: added.append(add_dll_directory(path)) or added[-1]

import slangpy as spy

for cookie in added:
    cookie.close()

device = spy.Device(type=spy.DeviceType.cpu, enable_hot_reload=False)
module = spy.Module(device.load_module_from_source("cpu_llvm_path", "float sq(int i) { return float(i * i); }"))
from slangpy.experimental.gridarg import grid

print(module.sq(grid(shape=(4,)), _result="numpy").tolist())
"""


# Windows ARM64 Slang packages ship no slang-llvm.dll, so only x64 has one to find.
@pytest.mark.skipif(
    sys.platform != "win32" or platform.machine().lower() != "amd64",
    reason="slang-llvm.dll ships with Windows x64 only",
)
@pytest.mark.parametrize("device_type", [DeviceType.cpu])
def test_cpu_dispatch_without_dll_search_path(device_type: DeviceType, tmp_path: Path) -> None:
    # Drop PATH entries that would let Slang find slang-llvm on its own.
    path = os.pathsep.join(
        p
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p and not (Path(p) / "slang-llvm.dll").exists()
    )
    result = subprocess.run(
        [sys.executable, "-c", WHEEL_LAYOUT_CPU_SCRIPT],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PATH": path},
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[0.0, 1.0, 4.0, 9.0]"
