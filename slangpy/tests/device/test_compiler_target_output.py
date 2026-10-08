# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check generated backend artifacts through production SlangPy sessions."""

from pathlib import Path
from typing import Any
import re
import struct

import numpy as np
import pytest
import slangpy as spy
from slangpy.testing import helpers
from slangpy.tests.device.test_compiler_profiles import (
    SOURCE,
    run_shader,
    require_cuda_profile,
)


CASES = [
    (spy.DeviceType.d3d12, "sm_6_0", "6.0"),
    (spy.DeviceType.d3d12, "sm_6_6", "6.6"),
    (spy.DeviceType.vulkan, "spirv_1_0", "1.0"),
    (spy.DeviceType.vulkan, "spirv_1_3", "1.3"),
    (spy.DeviceType.vulkan, "spirv_1_6", "1.6"),
]


@pytest.mark.parametrize(
    "device_type, profile, version",
    [case for case in CASES if case[0] in helpers.DEFAULT_DEVICE_TYPES],
)
def test_emitted_version(
    device_type: spy.DeviceType, profile: str, version: str, tmp_path: Path
) -> None:
    with spy.Device(type=device_type) as device:
        if "_" + profile not in device.capabilities:
            pytest.skip(f"Device does not advertise {profile}")
        session = device.create_slang_session(
            {
                "profile": profile,
                "dump_intermediates": True,
                "dump_intermediates_prefix": str(tmp_path / "target"),
            }
        )
        source = SOURCE
        if profile == "spirv_1_3":
            # Exercise subgroup capability emission alongside the selected SPIR-V version.
            source = SOURCE.replace("= 7", "= WaveActiveSum(tid.x + 7)")
        assert run_shader(device, session, source) == 7
        if device_type == spy.DeviceType.d3d12:
            artifacts = list(tmp_path.glob("*.dxil-asm"))
            assert artifacts
            major, minor = version.split(".")
            for artifact in artifacts:
                assert f'!{{!"cs", i32 {major}, i32 {minor}}}' in artifact.read_text()
        else:
            artifacts = list(tmp_path.glob("*.spv"))
            assert artifacts
            for artifact in artifacts:
                magic, encoded_version = struct.unpack_from("<II", artifact.read_bytes())
                assert magic == 0x07230203
                assert f"{(encoded_version >> 16) & 255}.{(encoded_version >> 8) & 255}" == version

        if profile == "spirv_1_3":
            artifacts = list(tmp_path.glob("*.spv-asm"))
            assert artifacts
            assert all(
                "OpCapability GroupNonUniformArithmetic" in path.read_text() for path in artifacts
            )


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.d3d12]
)
@pytest.mark.parametrize("profile", [None, "sm_6_6", "sm_6_9"])
def test_ser_nvapi(device_type: spy.DeviceType, profile: str | None, tmp_path: Path) -> None:
    with spy.Device(type=device_type) as device:
        if not device.has_feature(spy.Feature.ray_tracing) or not device.has_feature(
            spy.Feature.shader_execution_reordering
        ):
            pytest.skip("Requires ray tracing and SER API support")
        if "hlsl_nvapi" not in device.capabilities:
            pytest.skip("Requires NVAPI-enabled device")
        if profile is not None and "_" + profile not in device.capabilities:
            pytest.skip("Requires the selected shader model")
        options: dict[str, Any] = {
            "dump_intermediates": True,
            "dump_intermediates_prefix": str(tmp_path / "ser"),
        }
        options["profile"] = profile
        session = device.create_slang_session(options)
        source = """
RWStructuredBuffer<uint> output;
[shader("raygeneration")]
void main()
{
    HitObject hit = HitObject::MakeNop();
    ReorderThread(hit);
    output[0] = hit.IsNop() ? 7 : 0;
}
"""
        module = session.load_module_from_source("ser_implementation", source)
        program = session.link_program([module], [module.entry_point("main")])
        pipeline = device.create_ray_tracing_pipeline(
            program=program,
            hit_groups=[],
            max_recursion=1,
            compilation_policy=spy.PipelineCompilationPolicy.immediate,
        )
        table = device.create_shader_table(program=program, ray_gen_entry_points=["main"])
        buffer = device.create_buffer(size=4, usage=spy.BufferUsage.unordered_access)
        encoder = device.create_command_encoder()
        with encoder.begin_ray_tracing_pass() as ray_pass:
            shader_object = ray_pass.bind_pipeline(pipeline, table)
            spy.ShaderCursor(shader_object).output = buffer
            ray_pass.dispatch_rays(0, [1, 1, 1])
        device.submit_command_buffer(encoder.finish())
        assert buffer.to_numpy().view(np.uint32)[0] == 7
        artifacts = list(tmp_path.glob("*.hlsl"))
        assert artifacts
        assert any("NvHitObject" in path.read_text() for path in artifacts)
        assert all("dx::HitObject" not in path.read_text() for path in artifacts)
        assert list(tmp_path.glob("*.dxil"))


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.vulkan]
)
def test_spirv_int64_atomic_output(device_type: spy.DeviceType, tmp_path: Path) -> None:
    with spy.Device(type=device_type) as device:
        if not device.has_feature(spy.Feature.atomic_int64):
            pytest.skip("Requires 64-bit atomic support")
        session = device.create_slang_session(
            {
                "profile": "spirv_1_3",
                "enable_warnings": ["41012"],
                "warnings_as_errors": ["41012"],
                "dump_intermediates": True,
                "dump_intermediates_prefix": str(tmp_path / "atomic"),
            }
        )
        source = SOURCE.replace("<uint>", "<uint64_t>").replace(
            "output[tid.x] = 7;", "InterlockedAdd(output[0], uint64_t(1));"
        )
        module = session.load_module_from_source("atomic64", source)
        program = session.link_program([module], [module.entry_point("profile_main")])
        kernel = device.create_compute_kernel(program)
        buffer = device.create_buffer(
            data=np.zeros(1, dtype=np.uint64), usage=spy.BufferUsage.unordered_access
        )
        kernel.dispatch(thread_count=[1, 1, 1], vars={"output": buffer})
        assert buffer.to_numpy().view(np.uint64)[0] == 1
        artifacts = list(tmp_path.glob("*.spv-asm"))
        assert artifacts
        assert all("OpCapability Int64Atomics" in path.read_text() for path in artifacts)


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.cuda]
)
@pytest.mark.parametrize("profile", ["compute_75", "compute_86", "compute_120"])
def test_cuda_exact_architecture(device_type: spy.DeviceType, profile: str, tmp_path: Path) -> None:
    options: dict[str, Any] = {
        "dump_intermediates": True,
        "dump_intermediates_prefix": str(tmp_path / "exact"),
    }
    options["profile"] = profile
    with spy.Device(type=device_type) as device:
        require_cuda_profile(device, profile)
        session = device.create_slang_session(options)
        source = SOURCE.replace("<uint>", "<half>").replace("= 7", "= half(tid.x + 1)")
        module = session.load_module_from_source("exact_architecture", source)
        program = session.link_program([module], [module.entry_point("profile_main")])
        # Code generation stays deferred until dispatch.
        assert not list(tmp_path.glob("*.ptx"))
        pipeline = device.create_compute_pipeline(
            program=program, compilation_policy=spy.PipelineCompilationPolicy.deferred
        )
        assert not list(tmp_path.glob("*.ptx"))
        buffer = device.create_buffer(size=4, usage=spy.BufferUsage.unordered_access)
        encoder = device.create_command_encoder()
        with encoder.begin_compute_pass() as compute:
            root = compute.bind_pipeline(pipeline)
            spy.ShaderCursor(root).output = buffer
            compute.dispatch([1, 1, 1])
        device.submit_command_buffer(encoder.finish())
        assert buffer.to_numpy().view(np.float16)[0] == 1
        artifacts = list(tmp_path.glob("*.ptx"))
        assert artifacts
        for artifact in artifacts:
            target = re.search(r"(?m)^\s*\.target\s+sm_(\d+)", artifact.read_text())
            assert target
            assert int(target[1]) == int(profile.removeprefix("compute_"))


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.cuda]
)
def test_cuda_rejects_shader_architecture_upgrade(device_type: spy.DeviceType) -> None:
    with spy.Device(type=device_type) as device:
        require_cuda_profile(device, "compute_50")
        session = device.create_slang_session({"profile": "compute_50"})
        source = SOURCE.replace("<uint>", "<half>").replace("= 7", "= half(tid.x + 1)")
        module = session.load_module_from_source("unsupported_half_architecture", source)
        program = session.link_program([module], [module.entry_point("profile_main")])
        with pytest.raises(RuntimeError):
            device.create_compute_pipeline(
                program=program, compilation_policy=spy.PipelineCompilationPolicy.immediate
            )
