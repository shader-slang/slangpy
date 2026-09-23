# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public target selection: real session inputs, diagnostics, caches, and execution."""

from typing import Any
from pathlib import Path

import numpy as np
import pytest

import slangpy as spy
from slangpy.testing import helpers


BASELINES = {
    spy.DeviceType.d3d12: "hlsl",
    spy.DeviceType.vulkan: "spirv",
    spy.DeviceType.cuda: "cuda",
    spy.DeviceType.metal: "metal",
    spy.DeviceType.cpu: "cpp",
    spy.DeviceType.wgpu: "wgsl",
}
SOURCE = """
RWStructuredBuffer<uint> output;
[shader("compute")]
[numthreads(1, 1, 1)]
void capability_main(uint3 tid : SV_DispatchThreadID) { output[tid.x] = 7; }
"""


def run_shader(device: spy.Device, session: spy.SlangSession, source: str = SOURCE) -> int:
    module = session.load_module_from_source("capability_probe", source)
    program = session.link_program([module], [module.entry_point("capability_main")])
    kernel = device.create_compute_kernel(program)
    buffer = device.create_buffer(size=4, usage=spy.BufferUsage.unordered_access)
    kernel.dispatch(thread_count=[1, 1, 1], vars={"output": buffer})
    return int(buffer.to_numpy().view(np.uint32)[0])


def test_option_fields() -> None:
    options = spy.SlangCompilerOptions()
    assert options.profile is None
    assert options.capabilities is None
    assert options.capability_overrides == {}
    options.profile = "sm_6_6"
    options.capabilities = []
    options.capability_overrides = {"hlsl_nvapi": False}
    assert options.profile == "sm_6_6"
    assert options.capabilities == []
    assert options.capability_overrides == {"hlsl_nvapi": False}
    options = spy.SlangCompilerOptions({"profile": None, "capabilities": None})
    assert options.profile is None and options.capabilities is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_defaults_and_empty(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    before = list(device.capabilities)
    baseline = BASELINES[device_type]
    assert device.slang_session.target_info.legacy
    # A nonempty neutral override opts into device defaults during the transition.
    inherited = device.create_slang_session(
        {
            "capabilities": None,
            "capability_overrides": {baseline: True},
        }
    )
    assert not inherited.target_info.legacy
    assert inherited.target_info.input_capabilities == before
    assert inherited.target_info.capability_origins[baseline] == "override"
    assert set(inherited.target_info.ignored_capabilities) <= set(before)
    assert run_shader(device, inherited) == 7
    empty = device.create_slang_session({"capabilities": []})
    assert empty.target_info.input_capabilities == []
    assert empty.target_info.capabilities == [baseline]
    assert empty.target_info.capability_origins == {baseline: "baseline"}
    assert run_shader(device, empty) == 7
    assert list(device.capabilities) == before


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_device_options_and_fresh_session(device_type: spy.DeviceType) -> None:
    with spy.Device(type=device_type, compiler_options={"capabilities": []}) as device:
        assert not device.slang_session.target_info.legacy
        assert device.slang_session.desc.compiler_options.capabilities == []
        assert run_shader(device, device.slang_session) == 7
        fresh = device.create_slang_session()
        assert fresh.target_info.legacy
        assert fresh.desc.compiler_options.capabilities is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_reports_and_options_are_snapshots(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    options = spy.SlangCompilerOptions({"capabilities": []})
    session = device.create_slang_session(options)
    info = session.target_info
    digest = info.session_digest
    with pytest.raises(AttributeError):
        info.profile = "sm_6_9"  # type: ignore[misc]
    info.capabilities.append("typo")
    info.capability_origins["typo"] = "explicit"
    options.capabilities = ["typo"]
    desc = session.desc
    desc.compiler_options.capabilities = ["typo"]
    assert session.desc.compiler_options.capabilities == []
    assert "typo" not in session.target_info.capabilities
    assert run_shader(device, session) == 7
    device.reload_all_programs()
    assert session.target_info.session_digest == digest
    assert session.desc.compiler_options.capabilities == []


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "options, message",
    [
        ({"capabilities": ["not_a_slang_capability"]}, "Unknown explicit capability"),
        ({"capability_overrides": {"not_a_slang_capability": True}}, "Unknown capability override"),
        (
            {"capability_overrides": {"not_a_slang_capability": False}},
            "Unknown capability override",
        ),
        ({"profile": "not_a_slang_profile"}, "Unknown Slang profile"),
        (
            {"shader_model": spy.ShaderModel.sm_6_0, "capabilities": []},
            "shader_model cannot be combined",
        ),
    ],
)
def test_invalid_options(
    device_type: spy.DeviceType, options: dict[str, Any], message: str
) -> None:
    device = helpers.get_device(device_type)
    with pytest.raises(RuntimeError, match=message):
        device.create_slang_session(options)
    # Failed construction must not poison the hot-reload registry.
    device.reload_all_programs()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mandatory_baseline_removal(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    baseline = BASELINES[device_type]
    session = device.create_slang_session(
        {"capabilities": [], "capability_overrides": {baseline: False}}
    )
    assert session.target_info.capabilities == [baseline]
    assert baseline in session.target_info.removed_capabilities
    assert any("Mandatory backend baseline" in note for note in session.target_info.notes)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_functional_api_uses_selected_session(device_type: spy.DeviceType) -> None:
    if device_type == spy.DeviceType.cpu:
        pytest.skip(
            "CPU functional dispatch reports zero dispatch groups (also reproduced with legacy options)"
        )
    device = helpers.get_device(device_type)
    options: dict[str, Any]
    if device_type == spy.DeviceType.d3d12 and "_sm_6_6" in device.capabilities:
        options = {"capabilities": ["sm_6_6"]}
        # Actual capability-sensitive operation; verifies generated functional kernels share the session.
        body = "return WaveMatch(x).x & 1;"
        expected = 1
    else:
        # Exclude compiler-unknown detections by using the neutral override path instead.
        options = {"capability_overrides": {BASELINES[device_type]: True}}
        body = "return x + 1;"
        expected = 8
    # Fresh sessions intentionally do not inherit the device's default-session include paths.
    options["include_paths"] = [spy.SHADER_PATH]
    session = device.create_slang_session(options)
    module = spy.Module(
        session.load_module_from_source(
            "functional_target", "uint selected(uint x) { " + body + " }"
        )
    )
    assert module.device_module.session == session
    assert module.selected(7) == expected
    device.reload_all_programs()
    assert module.selected(7) == expected


@pytest.mark.parametrize(
    "device_type",
    [t for t in helpers.DEFAULT_DEVICE_TYPES if t in (spy.DeviceType.d3d12, spy.DeviceType.cuda)],
)
def test_aliases_and_digest(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    public = "sm_6_0" if device_type == spy.DeviceType.d3d12 else "cuda_sm_5_0"
    raw = "_" + public
    baseline = BASELINES[device_type]
    a = device.create_slang_session({"capabilities": [public, baseline, raw]})
    b = device.create_slang_session({"capabilities": [baseline, raw]})
    assert a.target_info.capabilities == b.target_info.capabilities
    assert a.target_info.session_digest == b.target_info.session_digest
    removed = device.create_slang_session(
        {"capabilities": [public], "capability_overrides": {raw: False}}
    )
    assert raw not in removed.target_info.capabilities
    with pytest.raises(RuntimeError, match="Conflicting overrides"):
        device.create_slang_session({"capability_overrides": {public: True, raw: False}})


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.d3d12]
)
def test_dx_profile_reconciliation(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    session = device.create_slang_session({"profile": "sm_6_0"})
    assert session.target_info.profile == "sm_6_0"
    assert not session.target_info.profile_automatic
    assert session.target_info.requested_profile == "sm_6_0"
    assert "_sm_6_6" not in session.target_info.capabilities
    if "_sm_6_6" in device.capabilities:
        assert "_sm_6_6" in session.target_info.removed_capabilities
        with pytest.raises(RuntimeError, match="override.*exceeding profile"):
            device.create_slang_session(
                {"profile": "sm_6_0", "capability_overrides": {"sm_6_6": True}}
            )
    with pytest.raises(RuntimeError, match="explicit.*exceeding profile"):
        device.create_slang_session({"profile": "sm_6_0", "capabilities": ["sm_6_6"]})
    with pytest.raises(RuntimeError, match="ser_hlsl_native.*6.9.*sm_6_6"):
        device.create_slang_session({"profile": "sm_6_6", "capabilities": ["ser_hlsl_native"]})
    with pytest.raises(RuntimeError, match="not supported"):
        device.create_slang_session({"profile": "spirv_1_6", "capabilities": []})
    if "_sm_6_6" in device.capabilities:
        capable = device.create_slang_session({"capabilities": ["sm_6_6"]})
        assert capable.target_info.profile == "sm_6_6"
        # This failed DXC compilation in the capability-only probe before the profile adapter.
        assert run_shader(device, capable, SOURCE.replace("= 7", "= WaveMatch(tid.x).x")) & 1
        assert capable.target_info.session_digest != session.target_info.session_digest


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.vulkan]
)
def test_spirv_profiles_and_bundles(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    if "_spirv_1_6" not in device.capabilities:
        pytest.skip("Requires SPIR-V 1.6 device support")
    raw = device.create_slang_session({"capabilities": ["_spirv_1_6"]})
    bundle = device.create_slang_session({"capabilities": ["spirv_1_6"]})
    assert raw.target_info.profile == "spirv_1_0"
    assert "_spirv_1_6" in raw.target_info.capabilities
    assert "spirv_1_6" in bundle.target_info.capabilities
    assert raw.target_info.session_digest != bundle.target_info.session_digest
    # E41012 is suppressed by existing policy; explicitly promote it for this requirement test.
    source = (
        "[require(SPV_EXT_physical_storage_buffer)] uint feature() { return 7; }\n"
        + SOURCE.replace("= 7", "= feature()")
    )
    explicit = device.create_slang_session(
        {
            "profile": "spirv_1_6",
            "capabilities": [],
            "capability_overrides": {"SPV_EXT_physical_storage_buffer": False},
            "enable_warnings": ["41012"],
            "warnings_as_errors": ["41012"],
        }
    )
    assert any("still supplies" in note for note in explicit.target_info.notes)
    assert run_shader(device, explicit, source) == 7
    restricted = device.create_slang_session(
        {
            "capabilities": ["_spirv_1_6"],
            "enable_warnings": ["41012"],
            "warnings_as_errors": ["41012"],
        }
    )
    with pytest.raises(
        (RuntimeError, spy.SlangCompileError), match="SPV_EXT_physical_storage_buffer"
    ):
        restricted.load_module_from_source("missing_profile_feature", source)
    with pytest.raises(RuntimeError, match="SPV_EXT_physical_storage_buffer.*1.3"):
        device.create_slang_session(
            {"profile": "spirv_1_0", "capabilities": ["SPV_EXT_physical_storage_buffer"]}
        )
    if "SPV_KHR_cooperative_matrix" in device.capabilities:
        with pytest.raises(RuntimeError, match="device.*exceeding profile"):
            device.create_slang_session({"profile": "spirv_1_3"})
    with pytest.raises(RuntimeError, match="requires an explicit capabilities list"):
        device.create_slang_session({"profile": "sm_6_6"})
    cross = device.create_slang_session({"profile": "sm_6_6", "capabilities": []})
    assert cross.target_info.profile == "sm_6_6"
    assert run_shader(device, cross) == 7


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.cuda]
)
def test_cuda_selection(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    if "_cuda_sm_7_0" not in device.capabilities:
        pytest.skip("Requires CUDA compute capability 7.0 or newer")
    session = device.create_slang_session({"capabilities": ["cuda_sm_7_0"]})
    assert session.target_info.profile is None
    assert "_cuda_sm_7_0" in session.target_info.capabilities
    assert run_shader(device, session) == 7
    with pytest.raises(RuntimeError, match="not supported"):
        device.create_slang_session({"profile": "sm_6_6"})
    inherited = device.create_slang_session({"capability_overrides": {"cuda": True}})
    for unknown in inherited.target_info.ignored_capabilities:
        removed = device.create_slang_session({"capability_overrides": {unknown: False}})
        assert unknown not in removed.target_info.ignored_capabilities
        with pytest.raises(RuntimeError, match="Unknown explicit capability"):
            device.create_slang_session({"capabilities": [unknown]})


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.cuda]
)
def test_cuda_exact_target(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    if "_cuda_sm_7_0" not in device.capabilities:
        pytest.skip("Requires CUDA compute capability 7.0 or newer")
    half_source = SOURCE.replace("<uint>", "<half>").replace("= 7", "= half(tid.x + 1)")
    exact = device.create_slang_session({"capabilities": ["cuda_sm_5_0"]})
    module = exact.load_module_from_source("requires_half", half_source)
    # Slang raises half-using code to at least sm_60, even with warning 41012 suppressed.
    with pytest.raises(RuntimeError, match="Exact CUDA target 5.0.*emitted sm_"):
        exact.link_program([module], [module.entry_point("capability_main")])
    # This checks emitted architecture, not Slang's semantic requirement closure: an annotation
    # alone need not change PTX, and the existing permissive capability policy remains in effect.
    source = SOURCE.replace('[shader("compute")]', '[require(_cuda_sm_7_0)] [shader("compute")]')
    assert run_shader(device, exact, source) == 7
    # Selecting the actual required tier succeeds, including after session reconstruction.
    matching = device.create_slang_session({"capabilities": ["cuda_sm_7_0"]})
    assert run_shader(device, matching, source) == 7
    device.reload_all_programs()
    assert run_shader(device, matching, source) == 7
    # NVRTC's minimum is separate from hardware support and Slang's capability registry.
    ancient = device.create_slang_session({"capabilities": ["cuda_sm_1_0"]})
    with pytest.raises(RuntimeError, match="Exact CUDA target 1.0.*emitted sm_"):
        run_shader(device, ancient)
    overridden = device.create_slang_session(
        {"capabilities": [], "capability_overrides": {"cuda_sm_5_0": True}}
    )
    module = overridden.load_module_from_source("override_half", half_source)
    with pytest.raises(RuntimeError, match="Exact CUDA target 5.0.*emitted sm_"):
        overridden.link_program([module], [module.entry_point("capability_main")])


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.cuda]
)
def test_cuda_runtime_specialization(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    source = """
interface IValue { uint get(); }
struct Value : IValue { uint get() { return 7; } }
ParameterBlock<IValue> value;
RWStructuredBuffer<uint> output;
[shader("compute")]
[numthreads(1, 1, 1)]
void capability_main(uint3 tid : SV_DispatchThreadID) { output[tid.x] = value.get(); }
"""
    exact = device.create_slang_session({"capabilities": ["cuda_sm_5_0"]})
    module = exact.load_module_from_source("runtime_specialization", source)
    with pytest.raises(RuntimeError, match="requires a fully specialized program"):
        exact.link_program([module], [module.entry_point("capability_main")])
    inherited = device.create_slang_session({"capability_overrides": {"cuda": True}})
    module = inherited.load_module_from_source("runtime_specialization", source)
    assert inherited.link_program([module], [module.entry_point("capability_main")])


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.cuda]
)
def test_cuda_exact_target_with_cache(device_type: spy.DeviceType, tmp_path: Path) -> None:
    for iteration in range(2):
        with spy.Device(type=device_type, shader_cache_path=tmp_path) as device:
            if "_cuda_sm_7_0" not in device.capabilities:
                pytest.skip("Requires CUDA compute capability 7.0 or newer")
            session = device.create_slang_session({"capabilities": ["cuda_sm_7_0"]})
            assert run_shader(device, session) == 7
            if iteration:
                assert device.shader_cache_stats.hit_count > 0
            exact = device.create_slang_session({"capabilities": ["cuda_sm_5_0"]})
            source = SOURCE.replace("<uint>", "<half>").replace("= 7", "= half(tid.x + 1)")
            module = exact.load_module_from_source("cache_mismatch", source)
            with pytest.raises(RuntimeError, match="Exact CUDA target 5.0.*emitted sm_"):
                exact.link_program([module], [module.entry_point("capability_main")])


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.cuda]
)
def test_cuda_exact_target_rejects_preexisting_cache(
    device_type: spy.DeviceType, tmp_path: Path
) -> None:
    source = SOURCE.replace("<uint>", "<half>").replace("= 7", "= half(tid.x + 1)")
    with spy.Device(type=device_type, shader_cache_path=tmp_path) as device:
        # Keep 5.0 inherited so this remains an input assumption; it can generate sm_60.
        automatic = device.create_slang_session(
            {
                "capability_overrides": {
                    name: False
                    for name in device.capabilities
                    if name not in ("cuda", "_cuda_sm_5_0")
                }
            }
        )
        module = automatic.load_module_from_source("cached_half", source)
        program = automatic.link_program([module], [module.entry_point("capability_main")])
        kernel = device.create_compute_kernel(program)
        buffer = device.create_buffer(size=4, usage=spy.BufferUsage.unordered_access)
        kernel.dispatch(thread_count=[1, 1, 1], vars={"output": buffer})
        assert buffer.to_numpy().view(np.float16)[0] == 1
        assert device.shader_cache_stats.entry_count > 0
        exact = device.create_slang_session({"capabilities": automatic.target_info.capabilities})
        assert exact.target_info.session_digest == automatic.target_info.session_digest
        module = exact.load_module_from_source("cached_half", source)
        with pytest.raises(RuntimeError, match="Exact CUDA target 5.0.*emitted sm_"):
            exact.link_program([module], [module.entry_point("capability_main")])


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.cuda]
)
def test_cuda_exact_target_failed_reload(device_type: spy.DeviceType, tmp_path: Path) -> None:
    path = tmp_path / "reload_target.slang"
    path.write_text(SOURCE)
    with spy.Device(type=device_type) as device:
        session = device.create_slang_session({"capabilities": ["cuda_sm_5_0"]})
        module = session.load_module(str(path))
        program = session.link_program([module], [module.entry_point("capability_main")])
        kernel = device.create_compute_kernel(program)
        buffer = device.create_buffer(size=4, usage=spy.BufferUsage.unordered_access)

        def dispatch() -> int:
            kernel.dispatch(thread_count=[1, 1, 1], vars={"output": buffer})
            return int(buffer.to_numpy().view(np.uint32)[0])

        assert dispatch() == 7
        path.write_text(SOURCE.replace("<uint>", "<half>").replace("= 7", "= half(tid.x + 1)"))
        device.reload_all_programs()
        # A failed hot reload keeps the old program and pipeline alive.
        assert dispatch() == 7
        path.write_text(SOURCE.replace("= 7", "= 9"))
        device.reload_all_programs()
        assert dispatch() == 9


@pytest.mark.parametrize(
    "device_type",
    [t for t in helpers.DEFAULT_DEVICE_TYPES if t in (spy.DeviceType.d3d12, spy.DeviceType.cuda)],
)
def test_downstream_target_conflicts(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    flag = "-Tcs_6_6" if device_type == spy.DeviceType.d3d12 else "--gpu-architecture=compute_70"
    with pytest.raises(RuntimeError, match="conflicts with compiler target selection"):
        device.create_slang_session({"capabilities": [], "downstream_args": [flag]})
    session = device.create_slang_session({"capabilities": []})
    module = session.load_module_from_source("target_conflict", SOURCE)
    with pytest.raises(RuntimeError, match="conflicts with compiler target selection"):
        session.link_program(
            [module], [module.entry_point("capability_main")], {"downstream_args": [flag]}
        )


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == spy.DeviceType.d3d12]
)
def test_nvapi_selection(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    source = SOURCE.replace("= 7", "= SGL_ENABLE_NVAPI")
    disabled = device.create_slang_session({"capabilities": []})
    assert run_shader(device, disabled, source) == 0
    assert not any("nvapi" in arg for arg in disabled.target_info.generated_downstream_args)
    if "hlsl_nvapi" in device.capabilities:
        enabled = device.create_slang_session({"capabilities": ["hlsl_nvapi"]})
        assert run_shader(device, enabled, source) == 1
        assert any("nvapi" in arg for arg in enabled.target_info.generated_downstream_args)


@pytest.mark.parametrize(
    "device_type",
    [
        t
        for t in helpers.DEFAULT_DEVICE_TYPES
        if t in (spy.DeviceType.cuda, spy.DeviceType.cpu, spy.DeviceType.wgpu)
    ],
)
def test_backend_without_profiles(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    with pytest.raises(RuntimeError, match="not supported"):
        device.create_slang_session({"profile": "sm_6_6"})
