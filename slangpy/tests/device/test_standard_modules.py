# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

import slangpy as spy
from slangpy.core.calldata import SLANG_PATH
from slangpy.testing import helpers
from slangpy.testing.helpers import test_id  # type: ignore (pytest fixture)

REPO_ROOT = Path(__file__).resolve().parents[3]
SLANGC_EXE = "slangc.exe" if os.name == "nt" else "slangc"

STANDARD_MODULE_SOURCE = """
import slangpy;
import slang.neural;

float use_slangpy_and_neural(uint3 tid)
{
    Context0D context = { tid };

    InlineVector<float, 2> input = InlineVector<float, 2>(-1.0f);
    input[1] = 2.0f;

    ReLU<float> activation = ReLU<float>();
    let output = activation.eval(input);

    return output[0] + output[1] + float(context.thread_id.x);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID, RWStructuredBuffer<float> result)
{
    result[0] = use_slangpy_and_neural(tid);
}
"""

RUNTIME_DEVICE_TYPES = [
    device_type
    for device_type in helpers.DEFAULT_DEVICE_TYPES
    if device_type in (spy.DeviceType.vulkan, spy.DeviceType.metal)
]


def find_slangc() -> Path | None:
    paths: list[Path] = []
    if "SLANG_PATH" in os.environ:
        paths.append(Path(os.environ["SLANG_PATH"]))

    paths += [
        REPO_ROOT / "build/pip/_deps/slang-src/bin",
        REPO_ROOT / "build/linux-gcc/_deps/slang-src/bin",
        REPO_ROOT / "build/windows-msvc/_deps/slang-src/bin",
        REPO_ROOT / "build/macos-arm64-clang/_deps/slang-src/bin",
    ]

    for path in paths:
        slangc_path = path / SLANGC_EXE
        if slangc_path.exists():
            return slangc_path

    path_slangc = shutil.which(SLANGC_EXE)
    if path_slangc is not None:
        return Path(path_slangc)

    return None


def find_slang_runtime_dirs() -> list[Path]:
    package_dir = Path(spy.__file__).resolve().parent
    candidate_dirs = [package_dir]

    build_dir_file = package_dir / ".build_dir"
    if build_dir_file.exists():
        candidate_dirs.insert(0, Path(build_dir_file.read_text().strip()))

    return [path for path in candidate_dirs if path.exists() and any(path.glob("*slang-compiler*"))]


def slangc_environment(runtime_dirs: list[Path]) -> dict[str, str]:
    env = os.environ.copy()

    if os.name == "nt":
        library_path_name = "PATH"
    elif sys.platform == "darwin":
        library_path_name = "DYLD_LIBRARY_PATH"
    else:
        library_path_name = "LD_LIBRARY_PATH"

    existing_library_path = env.get(library_path_name)
    library_paths = [str(path) for path in runtime_dirs]
    if existing_library_path:
        library_paths.append(existing_library_path)
    env[library_path_name] = os.pathsep.join(library_paths)

    return env


def compile_standard_module_source(
    slangc: Path,
    target: str,
    source_file: Path,
    output_file: Path,
    runtime_dirs: list[Path],
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(slangc),
        "-experimental-feature",
        "-Wno-30856",
        "-I",
        str(SLANG_PATH),
        "-target",
        target,
        "-profile",
        "sm_6_0",
        "-entry",
        "compute_main",
        "-stage",
        "compute",
        "-o",
        str(output_file),
        str(source_file),
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=slangc_environment(runtime_dirs),
    )


@pytest.mark.parametrize("device_type", RUNTIME_DEVICE_TYPES)
def test_import_neural_standard_module(device_type: spy.DeviceType, test_id: str) -> None:
    if helpers.should_skip_test_for_device(device_type):
        pytest.skip(f"Skipping {device_type.name} device test")

    device = spy.Device(type=device_type, label=f"standard-module-{device_type.name}")

    try:
        session = device.create_slang_session(
            compiler_options={
                "enable_experimental_features": True,
                "include_paths": [SLANG_PATH],
            },
            add_default_include_paths=False,
        )

        module = session.load_module_from_source(
            module_name=f"standard_neural_{test_id}",
            source=STANDARD_MODULE_SOURCE,
        )
        entry_point = module.entry_point("compute_main")

        assert entry_point.stage == spy.ShaderStage.compute

        program = session.link_program([module], [entry_point])
        kernel = device.create_compute_kernel(program)
        result = device.create_buffer(
            data=np.array([0.0], dtype=np.float32),
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        )

        kernel.dispatch(thread_count=[1, 1, 1], result=result)

        np.testing.assert_allclose(result.to_numpy().view(dtype=np.float32), [2.0])

    finally:
        device.close()


@pytest.mark.parametrize("target,extension", [("hlsl", "hlsl"), ("metal", "metal")])
def test_import_neural_standard_module_source_targets(
    target: str, extension: str, tmp_path: Path
) -> None:
    slangc = find_slangc()
    if slangc is None:
        pytest.skip("slangc compiler not found")

    runtime_dirs = find_slang_runtime_dirs()
    if not runtime_dirs:
        pytest.skip("SlangPy runtime compiler library not found")

    source_file = tmp_path / "standard_modules.slang"
    output_file = tmp_path / f"standard_modules.{extension}"
    source_file.write_text(STANDARD_MODULE_SOURCE)

    result = compile_standard_module_source(slangc, target, source_file, output_file, runtime_dirs)

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert output_file.stat().st_size > 0
    assert "compute_main" in output_file.read_text()
