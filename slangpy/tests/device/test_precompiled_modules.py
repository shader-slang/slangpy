# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import pytest
import subprocess
import shutil
from pathlib import Path

import slangpy as spy
from slangpy.testing import helpers
from slangpy.core.calldata import SLANG_PATH


def is_running_in_ci():
    """Check if we're running in a CI environment."""
    ci_indicators = ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL", "CIRCLECI"]
    return any(os.environ.get(indicator) for indicator in ci_indicators)


def find_slangc():
    """Find the slangc compiler binary.

    Returns:
        Path to slangc if found, None otherwise.
    """
    # Look for slangc in the build directory
    root_dir = Path(__file__).parent.parent.parent.parent
    build_dirs = [
        root_dir / "build/pip/_deps/slang-src/bin",
        root_dir / "build/linux-gcc/_deps/slang-src/bin",
        root_dir / "build/windows-msvc/_deps/slang-src/bin",
        root_dir / "build/macos-arm64-clang/_deps/slang-src/bin",
    ]

    for build_dir in build_dirs:
        slangc_path = build_dir / "slangc"
        if slangc_path.exists():
            return slangc_path
        slangc_path = build_dir / "slangc.exe"
        if slangc_path.exists():
            return slangc_path

    return None


def compile_precompiled_module(
    slangc_path: Path, source_file: Path, output_file: Path, include_paths: list[Path] = None
) -> None:
    """Compile a .slang file to a .slang-module precompiled module."""
    cmd = [
        str(slangc_path),
        "-emit-ir",
        "-o",
        str(output_file),
    ]

    if include_paths:
        for include_path in include_paths:
            cmd.extend(["-I", str(include_path)])

    cmd.append(str(source_file))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to compile {source_file}: {result.stderr}\nstdout: {result.stdout}"
        )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_precompiled_module_without_source(device_type: spy.DeviceType, tmpdir: str):
    """Test that precompiled modules are loaded when source .slang file is not present.

    This definitively verifies that .slang-module files are used by compiling a module
    to a different name (test_no_source) and loading it without the corresponding .slang file.
    """
    # Check if slangc is available
    slangc = find_slangc()
    if slangc is None:
        if is_running_in_ci():
            raise RuntimeError("slangc compiler not found in CI environment")
        pytest.skip("slangc compiler not found (skipping outside CI)")

    test_dir = Path(__file__).parent
    cache_dir = Path(tmpdir)

    source_file = test_dir / "test_precompiled_module.slang"
    precompiled_file = cache_dir / "test_no_source.slang-module"

    compile_precompiled_module(slangc, source_file, precompiled_file)
    assert precompiled_file.exists()

    include_paths_list = [cache_dir, SLANG_PATH]

    device = spy.Device(
        type=device_type,
        enable_print=True,
        compiler_options={"include_paths": include_paths_list},
        label=f"test-precompiled-no-source-{device_type.name}",
    )

    device.slang_session.load_module("slangpy")

    module = device.load_module("test_no_source")
    m = spy.Module(module)

    # Verify it works correctly
    result = m.add_floats(3.0, 4.0)
    assert abs(result - 7.0) < 1e-5

    device.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
