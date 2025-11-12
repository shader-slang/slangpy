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

    Raises:
        RuntimeError: If running in CI and slangc is not found.
    """
    # Look for slangc in the build directory
    build_dirs = [
        Path(__file__).parent.parent.parent.parent
        / "build"
        / "pip"
        / "_deps"
        / "slang-src"
        / "bin",
        Path(__file__).parent.parent.parent.parent
        / "build"
        / "linux-gcc"
        / "_deps"
        / "slang-src"
        / "bin",
        Path(__file__).parent.parent.parent.parent
        / "build"
        / "windows-msvc"
        / "_deps"
        / "slang-src"
        / "bin",
        Path(__file__).parent.parent.parent.parent
        / "build"
        / "macos-arm64-clang"
        / "_deps"
        / "slang-src"
        / "bin",
    ]

    for build_dir in build_dirs:
        slangc_path = build_dir / "slangc"
        if slangc_path.exists():
            return slangc_path
        slangc_path = build_dir / "slangc.exe"
        if slangc_path.exists():
            return slangc_path

    # Try to find slangc in PATH
    slangc_path = shutil.which("slangc")
    if slangc_path:
        return Path(slangc_path)

    # If not found, fail in CI but return None locally
    if is_running_in_ci():
        raise RuntimeError("slangc compiler not found in CI environment")

    return None


def compile_precompiled_module(
    source_file: Path, output_file: Path, include_paths: list[Path] = None
) -> None:
    """Compile a .slang file to a .slang-module precompiled module."""
    slangc = find_slangc()

    cmd = [
        str(slangc),
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

    This definitively verifies that .slang-module files are used by deleting the source
    after compilation and confirming the module still loads and executes correctly.
    """
    # Check if slangc is available
    slangc = find_slangc()
    if slangc is None:
        pytest.skip("slangc compiler not found (skipping outside CI)")

    test_dir = Path(__file__).parent
    cache_dir = Path(tmpdir)

    source_file = test_dir / "test_precompiled_module.slang"
    source_copy = cache_dir / "test_no_source.slang"
    source_copy.write_text(source_file.read_text())
    precompiled_file = cache_dir / "test_no_source.slang-module"

    # Find include paths
    sgl_shaders_dir = None
    potential_shader_dirs = [
        test_dir.parent.parent.parent
        / "build"
        / "lib.linux-x86_64-cpython-312"
        / "slangpy"
        / "shaders",
        test_dir.parent.parent.parent
        / "build"
        / "lib.linux-x86_64-cpython-311"
        / "slangpy"
        / "shaders",
        test_dir.parent.parent.parent / "src",
    ]
    for shader_dir in potential_shader_dirs:
        if shader_dir.exists() and (shader_dir / "sgl" / "device" / "print.slang").exists():
            sgl_shaders_dir = shader_dir
            break

    include_paths = []
    if sgl_shaders_dir:
        include_paths.append(sgl_shaders_dir)

    compile_precompiled_module(source_copy, precompiled_file, include_paths)
    assert precompiled_file.exists()

    # Delete the source file - this proves we're using .slang-module
    source_copy.unlink()
    assert not source_copy.exists()

    include_paths_list = [cache_dir, SLANG_PATH]
    if sgl_shaders_dir:
        include_paths_list.append(sgl_shaders_dir)

    device = spy.Device(
        type=device_type,
        enable_print=True,
        compiler_options={"include_paths": include_paths_list},
        label=f"test-precompiled-no-source-{device_type.name}",
    )

    device.slang_session.load_module("slangpy")

    # This MUST load from .slang-module since .slang doesn't exist
    module = device.load_module("test_no_source")
    m = spy.Module(module)

    # Verify it works correctly
    result = m.add_floats(3.0, 4.0)
    assert abs(result - 7.0) < 1e-5

    device.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
