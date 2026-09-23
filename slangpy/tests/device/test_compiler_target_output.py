# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check generated backend artifacts through production SlangPy sessions."""

from pathlib import Path
import re
import struct

import numpy as np
import pytest
import slangpy as spy
from slangpy.testing import helpers
from slangpy.tests.device.test_compiler_capabilities import SOURCE, run_shader


CASES = [
    (spy.DeviceType.d3d12, "sm_6_0", "6.0"),
    (spy.DeviceType.d3d12, "sm_6_6", "6.6"),
    (spy.DeviceType.vulkan, "_spirv_1_0", "1.0"),
    (spy.DeviceType.vulkan, "_spirv_1_3", "1.3"),
    (spy.DeviceType.vulkan, "_spirv_1_6", "1.6"),
    (spy.DeviceType.cuda, "cuda_sm_7_0", "70"),
]


@pytest.mark.parametrize(
    "device_type, capability, version",
    [case for case in CASES if case[0] in helpers.DEFAULT_DEVICE_TYPES],
)
def test_emitted_version(
    device_type: spy.DeviceType, capability: str, version: str, tmp_path: Path
) -> None:
    # Use a fresh device without a persistent cache so the dump belongs to this compilation.
    with spy.Device(type=device_type) as device:
        raw = capability if capability.startswith("_") else "_" + capability
        if raw not in device.capabilities:
            pytest.skip(f"Device does not advertise {raw}")
        session = device.create_slang_session(
            {
                "capabilities": [capability],
                "dump_intermediates": True,
                "dump_intermediates_prefix": str(tmp_path / "target"),
            }
        )
        assert run_shader(device, session) == 7
        if device_type == spy.DeviceType.d3d12:
            artifacts = list(tmp_path.glob("*.dxil-asm"))
            assert artifacts
            major, minor = version.split(".")
            for artifact in artifacts:
                assert f'!{{!"cs", i32 {major}, i32 {minor}}}' in artifact.read_text()
        elif device_type == spy.DeviceType.vulkan:
            artifacts = list(tmp_path.glob("*.spv"))
            assert artifacts
            for artifact in artifacts:
                magic, encoded_version = struct.unpack_from("<II", artifact.read_bytes())
                assert magic == 0x07230203
                assert f"{(encoded_version >> 16) & 255}.{(encoded_version >> 8) & 255}" == version
        else:
            artifacts = list(tmp_path.glob("*.ptx"))
            assert artifacts
            for artifact in artifacts:
                text = artifact.read_text()
                assert re.search(rf"(?m)^\s*\.target\s+sm_{version}\s*$", text)
                assert re.search(r"(?m)^\s*\.version\s+\d+\.\d+\s*$", text)


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.vulkan]
)
def test_spirv_subgroup_output(device_type: spy.DeviceType, tmp_path: Path) -> None:
    with spy.Device(type=device_type) as device:
        if "_spirv_1_3" not in device.capabilities:
            pytest.skip("Requires SPIR-V 1.3")
        session = device.create_slang_session(
            {
                "capabilities": ["_spirv_1_3", "spvGroupNonUniformArithmetic"],
                "enable_warnings": ["41012"],
                "warnings_as_errors": ["41012"],
                "dump_intermediates": True,
                "dump_intermediates_prefix": str(tmp_path / "wave"),
            }
        )
        assert run_shader(device, session, SOURCE.replace("= 7", "= WaveActiveSum(tid.x + 1)")) >= 1
        artifacts = list(tmp_path.glob("*.spv-asm"))
        assert artifacts
        assert all(
            "OpCapability GroupNonUniformArithmetic" in path.read_text() for path in artifacts
        )


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.d3d12]
)
@pytest.mark.parametrize("implementation", ["native", "nvapi"])
def test_ser_implementation(
    device_type: spy.DeviceType, implementation: str, tmp_path: Path
) -> None:
    with spy.Device(type=device_type) as device:
        if not device.has_feature(spy.Feature.ray_tracing) or not device.has_feature(
            spy.Feature.shader_execution_reordering
        ):
            pytest.skip("Requires ray tracing and SER API support")
        if implementation == "native":
            if "_sm_6_9" not in device.capabilities:
                pytest.skip("Native SER requires shader model 6.9")
            capabilities = ["ser_hlsl_native"]
            marker = "dx::HitObject"
            other_marker = "NvHitObject"
        else:
            if "hlsl_nvapi" not in device.capabilities:
                pytest.skip("Requires NVAPI-enabled device")
            capabilities = ["sm_6_6", "hlsl_nvapi"]
            marker = "NvHitObject"
            other_marker = "dx::HitObject"
        session = device.create_slang_session(
            {
                "capabilities": capabilities,
                "dump_intermediates": True,
                "dump_intermediates_prefix": str(tmp_path / "ser"),
            }
        )
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
        assert any(marker in path.read_text() for path in artifacts)
        assert all(other_marker not in path.read_text() for path in artifacts)
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
                "capabilities": ["_spirv_1_3", "spvInt64Atomics"],
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
        program = session.link_program([module], [module.entry_point("capability_main")])
        kernel = device.create_compute_kernel(program)
        buffer = device.create_buffer(
            data=np.zeros(1, dtype=np.uint64), usage=spy.BufferUsage.unordered_access
        )
        kernel.dispatch(thread_count=[1, 1, 1], vars={"output": buffer})
        assert buffer.to_numpy().view(np.uint64)[0] == 1
        artifacts = list(tmp_path.glob("*.spv-asm"))
        assert artifacts
        assert all("OpCapability Int64Atomics" in path.read_text() for path in artifacts)
