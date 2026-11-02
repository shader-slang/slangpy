# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiler profiles: validation, execution, caching, and hot reload."""

from typing import Iterator
from pathlib import Path

import numpy as np
import pytest
import slangpy as spy
from slangpy.testing import helpers


SOURCE = """
RWStructuredBuffer<uint> output;
[shader("compute")]
[numthreads(1, 1, 1)]
void profile_main(uint3 tid : SV_DispatchThreadID) { output[tid.x] = 7; }
"""


def run_shader(device: spy.Device, session: spy.SlangSession, source: str = SOURCE) -> int:
    module = session.load_module_from_source("profile_test", source)
    program = session.link_program([module], [module.entry_point("profile_main")])
    kernel = device.create_compute_kernel(program)
    buffer = device.create_buffer(size=4, usage=spy.BufferUsage.unordered_access)
    kernel.dispatch(thread_count=[1, 1, 1], vars={"output": buffer})
    return int(buffer.to_numpy().view(np.uint32)[0])


def require_cuda_profile(device: spy.Device, profile: str) -> None:
    try:
        device.create_slang_session({"profile": profile})
    except RuntimeError as error:
        if any(
            message in str(error)
            for message in (
                "not supported by Slang's NVRTC",
                "exceeds detected device",
                "exceeds SGL_MAX_CUDA_COMPUTE_CAPABILITY",
            )
        ):
            pytest.skip(str(error))
        raise


class CompilerMessages(spy.LoggerOutput):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def write(self, level: spy.LogLevel, name: str, msg: str) -> None:
        self.messages.append(msg)


@pytest.fixture
def compiler_messages() -> Iterator[list[str]]:
    output = CompilerMessages()
    logger = spy.Logger.get()
    logger.add_output(output)
    try:
        yield output.messages
    finally:
        logger.remove_output(output)


def test_profile_option() -> None:
    options = spy.SlangCompilerOptions()
    assert options.profile is None
    options.profile = "sm_6_6"
    assert options.profile == "sm_6_6"
    options.profile = None
    assert options.profile is None
    with pytest.raises(RuntimeError, match="Unknown key shader_model"):
        spy.SlangCompilerOptions({"shader_model": None})


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_session_options(device_type: spy.DeviceType) -> None:
    with spy.Device(
        type=device_type, compiler_options={"defines": {"CUSTOM_TARGET": "7"}}
    ) as device:
        assert (
            run_shader(device, device.slang_session, SOURCE.replace("= 7", "= CUSTOM_TARGET")) == 7
        )
        options = spy.SlangCompilerOptions()
        session = device.create_slang_session(options)
        options.profile = "typo"
        desc = session.desc
        desc.compiler_options.profile = "typo"
        assert session.desc.compiler_options.profile is None
        assert "CUSTOM_TARGET" not in session.desc.compiler_options.defines
        assert run_shader(device, session) == 7


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("profile", ["not_a_slang_profile", ""])
def test_invalid_profile(device_type: spy.DeviceType, profile: str) -> None:
    device = helpers.get_device(device_type)
    with pytest.raises(RuntimeError, match="Unknown Slang profile"):
        device.create_slang_session({"profile": profile})
    device.reload_all_programs()


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.vulkan]
)
@pytest.mark.parametrize("profile", ["sm_6_6", "glsl_460"])
def test_vulkan_cross_family_profile(device_type: spy.DeviceType, profile: str) -> None:
    device = helpers.get_device(device_type)
    assert run_shader(device, device.create_slang_session({"profile": profile})) == 7


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.d3d12]
)
def test_stage_profile(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    assert run_shader(device, device.create_slang_session({"profile": "cs_6_0"})) == 7
    session = device.create_slang_session({"profile": "ps_6_0"})
    with pytest.raises(spy.SlangCompileError):
        run_shader(device, session)


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.vulkan]
)
def test_capability_upgrade_warnings_visible(
    device_type: spy.DeviceType, compiler_messages: list[str]
) -> None:
    source = (
        "[require(SPV_EXT_physical_storage_buffer)] uint feature() { return 7; }\n"
        + SOURCE.replace("= 7", "= feature()")
    )
    with spy.Device(type=device_type) as device:
        session = device.create_slang_session({"profile": "spirv_1_0"})
        assert run_shader(device, session, source) == 7
        assert any("41012" in message for message in compiler_messages)
        strict = device.create_slang_session(
            {"profile": "spirv_1_0", "warnings_as_errors": ["41012"]}
        )
        with pytest.raises((RuntimeError, spy.SlangCompileError), match="41012"):
            run_shader(device, strict, source)


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.cuda]
)
@pytest.mark.parametrize(
    "profile",
    [
        "compute_9990",
        "compute_90a",
        "compute_0",
        "compute_075",
        "compute_",
        "compute_999999999999999999999",
    ],
)
def test_invalid_cuda_profile(device_type: spy.DeviceType, profile: str) -> None:
    device = helpers.get_device(device_type)
    with pytest.raises(RuntimeError, match="CUDA profile"):
        device.create_slang_session({"profile": profile})


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.cuda]
)
def test_cuda_unsupported_nvrtc_profile(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    # 7.3 is neither an NVRTC architecture nor a valid approximation to 7.5.
    with pytest.raises(
        RuntimeError, match="not supported by Slang's NVRTC|exceeds detected device"
    ):
        device.create_slang_session({"profile": "compute_73"})
    device.reload_all_programs()


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.cuda]
)
def test_cuda_profile_excludes_newer_runtime_features(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    require_cuda_profile(device, "compute_75")
    require_cuda_profile(device, "compute_90")
    if "optix_coopvec" not in device.capabilities:
        pytest.skip("Requires OptiX cooperative-vector support")
    source = "[require(optix_coopvec)] uint feature() { return 7; }\n" + SOURCE.replace(
        "= 7", "= feature()"
    )
    older = device.create_slang_session({"profile": "compute_75", "warnings_as_errors": ["41012"]})
    with pytest.raises(spy.SlangCompileError, match="optix_coopvec"):
        older.load_module_from_source("missing_optix_requirement", source)
    supported = device.create_slang_session(
        {"profile": "compute_90", "warnings_as_errors": ["41012"]}
    )
    assert run_shader(device, supported, source) == 7


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.d3d12]
)
@pytest.mark.parametrize("profile", ["sm_6_6", "sm_6_7", "sm_6_9"])
def test_ray_payload_compatibility(device_type: spy.DeviceType, profile: str) -> None:
    device = helpers.get_device(device_type)
    if "_" + profile not in device.capabilities or not device.has_feature(spy.Feature.ray_tracing):
        pytest.skip("Requires the selected shader model and ray tracing")
    session = device.create_slang_session({"profile": profile})
    # Compiling miss/hit entry points separately used to fail at SM 6.7+ because
    # Slang omits raypayload annotations. Exercise actual DXC and RHI compilation.
    module = session.load_module_from_source(
        "ray_payload", Path(__file__).with_name("test_pipeline_rt.slang").read_text()
    )
    program = session.link_program(
        [module],
        [module.entry_point(name) for name in ("rt_ray_gen", "rt_miss", "rt_closest_hit")],
    )
    device.create_ray_tracing_pipeline(
        program=program,
        hit_groups=[{"hit_group_name": "hit", "closest_hit_entry_point": "rt_closest_hit"}],
        max_recursion=1,
        max_ray_payload_size=16,
        compilation_policy=spy.PipelineCompilationPolicy.immediate,
    )


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.cuda]
)
def test_cuda_runtime_specialization(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    require_cuda_profile(device, "compute_75")
    source = """
interface IValue { uint get(); }
struct Value : IValue { uint get() { return 7; } }
ParameterBlock<IValue> value;
RWStructuredBuffer<uint> output;
[shader("compute")]
[numthreads(1, 1, 1)]
void profile_main(uint3 tid : SV_DispatchThreadID) { output[tid.x] = value.get(); }
"""
    session = device.create_slang_session(
        {"profile": "compute_75", "include_paths": [spy.SHADER_PATH]}
    )
    module = session.load_module_from_source("runtime_specialization", source)
    # Interface specialization can remain unresolved until RHI dispatch.
    program = session.link_program([module], [module.entry_point("profile_main")])
    kernel = device.create_compute_kernel(program)
    packed = spy.pack(spy.Module(module), {"_type": "Value"})
    # Bind the concrete type from the original module used by the linked program.
    # The functional API's composed module has a different type identity.
    layout = module.layout.get_type_layout(module.layout.find_type_by_name("Value"))
    value = device.create_shader_object(layout)
    argument = spy.slangpy.NativePackedArg(packed.python, value, {})
    buffer = device.create_buffer(size=4, usage=spy.BufferUsage.unordered_access)
    kernel.dispatch(thread_count=[1, 1, 1], vars={"output": buffer, "value": argument})
    assert buffer.to_numpy().view(np.uint32)[0] == 7


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.cuda]
)
def test_cuda_profile_cache(device_type: spy.DeviceType, tmp_path: Path) -> None:
    # Both targets use the same recognized Slang 8.0 tier; NVRTC arguments must distinguish them.
    profiles = ("compute_80", "compute_86")
    for profile in profiles:
        require_cuda_profile(helpers.get_device(device_type), profile)
    for cached in (False, True):
        with spy.Device(type=device_type, shader_cache_path=tmp_path) as device:
            for profile in profiles:
                misses = device.shader_cache_stats.miss_count
                assert run_shader(device, device.create_slang_session({"profile": profile})) == 7
                if cached:
                    assert device.shader_cache_stats.hit_count > 0
                    assert device.shader_cache_stats.miss_count == misses
                else:
                    assert device.shader_cache_stats.miss_count > misses


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.cuda]
)
def test_cuda_profile_reload(device_type: spy.DeviceType, tmp_path: Path) -> None:
    path = tmp_path / "reload_target.slang"
    path.write_text(SOURCE)
    with spy.Device(type=device_type) as device:
        require_cuda_profile(device, "compute_75")
        session = device.create_slang_session({"profile": "compute_75"})
        module = session.load_module(str(path))
        program = session.link_program([module], [module.entry_point("profile_main")])
        kernel = device.create_compute_kernel(program)
        buffer = device.create_buffer(size=4, usage=spy.BufferUsage.unordered_access)

        kernel.dispatch(thread_count=[1, 1, 1], vars={"output": buffer})
        assert buffer.to_numpy().view(np.uint32)[0] == 7
        # Reload preserves the selected architecture when shader requirements change.
        path.write_text(SOURCE.replace("<uint>", "<half>").replace("= 7", "= half(tid.x + 1)"))
        device.reload_all_programs()
        kernel.dispatch(thread_count=[1, 1, 1], vars={"output": buffer})
        assert buffer.to_numpy().view(np.float16)[0] == 1
        assert session.desc.compiler_options.profile == "compute_75"


INCOMPATIBLE_PROFILES = [
    (spy.DeviceType.d3d12, "compute_75"),
    (spy.DeviceType.vulkan, "compute_75"),
    (spy.DeviceType.metal, "compute_75"),
    (spy.DeviceType.cuda, "sm_6_6"),
    (spy.DeviceType.cpu, "sm_6_6"),
    (spy.DeviceType.wgpu, "sm_6_6"),
]


@pytest.mark.parametrize(
    "device_type, profile",
    [case for case in INCOMPATIBLE_PROFILES if case[0] in helpers.DEFAULT_DEVICE_TYPES],
)
def test_incompatible_profile(device_type: spy.DeviceType, profile: str) -> None:
    device = helpers.get_device(device_type)
    with pytest.raises(RuntimeError, match="not supported"):
        device.create_slang_session({"profile": profile})
