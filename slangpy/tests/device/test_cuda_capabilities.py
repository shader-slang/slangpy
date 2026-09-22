# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest
import slangpy as spy
from slangpy.testing import helpers


CUDA_DEVICE_TYPES = [d for d in helpers.DEFAULT_DEVICE_TYPES if d == spy.DeviceType.cuda]
NON_CUDA_DEVICE_TYPES = [d for d in helpers.DEFAULT_DEVICE_TYPES if d != spy.DeviceType.cuda]


def _targets(device: spy.Device) -> list[int]:
    compiler = device.cuda_compiler_info
    assert compiler is not None
    targets = [
        a
        for a in compiler.supported_architectures
        if device.has_capability(f"_cuda_sm_{a // 10}_{a % 10}")
    ]
    assert targets
    return targets


def _read_architecture(
    device: spy.Device, session: spy.SlangSession, link_options: dict | None = None
) -> int:
    module = session.load_module_from_source(
        "architecture_readback",
        """
        uint architecture() { __intrinsic_asm "(__CUDA_ARCH__)"; }
        [shader("compute")][numthreads(1,1,1)]
        void entry(RWStructuredBuffer<uint> output) { output[0] = architecture(); }
        """,
    )
    program = session.link_program(
        [module], [module.entry_point("entry")], link_options=link_options
    )
    output = device.create_buffer(
        data=np.zeros(1, dtype=np.uint32), usage=spy.BufferUsage.unordered_access
    )
    device.create_compute_kernel(program).dispatch(thread_count=[1, 1, 1], output=output)
    return int(output.to_numpy().view(np.uint32)[0])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_cuda_compiler_metadata(device_type: spy.DeviceType) -> None:
    device = spy.create_device({"type": device_type})
    try:
        compiler = device.cuda_compiler_info
        if device_type != spy.DeviceType.cuda:
            assert compiler is None
            return
        assert compiler is not None
        assert compiler.path and compiler.version_major > 0
        assert compiler.supported_architectures
        architectures = compiler.supported_architectures
        compiler.supported_architectures.clear()
        assert device.cuda_compiler_info.supported_architectures == architectures
    finally:
        device.close()
    assert compiler.path and compiler.supported_architectures == architectures
    assert device.cuda_compiler_info.supported_architectures == architectures


@pytest.mark.parametrize("device_type", CUDA_DEVICE_TYPES)
def test_cuda_default_and_independent_sessions(device_type: spy.DeviceType) -> None:
    device = spy.create_device({"type": device_type})
    try:
        targets = _targets(device)
        assert device.slang_session.desc.compiler_options.cuda_architecture is None
        assert _read_architecture(device, device.slang_session) == max(targets) * 10
        options = spy.SlangCompilerOptions(
            {"cuda_architecture": None, "downstream_args": ["--use_fast_math"]}
        )
        options.cuda_architecture = min(targets)
        options.cuda_architecture = None
        for _ in range(2):
            session = device.create_slang_session(compiler_options=options)
            assert (
                _read_architecture(
                    device, session, {"downstream_args": [f"-arch=compute_{min(targets)}"]}
                )
                == min(targets) * 10
            )
            assert options.cuda_architecture is None
            assert options.downstream_args == ["--use_fast_math"]
    finally:
        device.close()
    device = spy.Device(type=device_type, compiler_options={"cuda_architecture": min(targets)})
    try:
        assert _read_architecture(device, device.slang_session) == min(targets) * 10
        # Unrelated link settings and downstream options must not reset an exact session target.
        assert (
            _read_architecture(
                device, device.slang_session, {"optimization": spy.SlangOptimizationLevel.high}
            )
            == min(targets) * 10
        )
        assert (
            _read_architecture(
                device, device.slang_session, {"downstream_args": ["--use_fast_math"]}
            )
            == min(targets) * 10
        )
        assert _read_architecture(device, device.create_slang_session()) == max(targets) * 10
    finally:
        device.close()


@pytest.mark.parametrize("device_type", CUDA_DEVICE_TYPES)
def test_cuda_invalid_and_raw_requests(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    target = min(_targets(device))
    for request, diagnostic in [
        (0, "greater than zero"),
        (999, "does not support compute_999"),
    ]:
        with pytest.raises(RuntimeError, match=diagnostic):
            spy.Device(type=device_type, compiler_options={"cuda_architecture": request})
        with pytest.raises(RuntimeError, match=diagnostic):
            device.create_slang_session(compiler_options={"cuda_architecture": request})
    # Compilation for another device is limited by NVRTC, not this GPU's capabilities.
    compiler_target = max(device.cuda_compiler_info.supported_architectures)
    device.create_slang_session(compiler_options={"cuda_architecture": compiler_target})
    options = spy.SlangCompilerOptions({"downstream_args": ["-arch", f"sm_{target}"]})
    raw_session = device.create_slang_session(compiler_options=options)
    assert _read_architecture(device, raw_session) == target * 10
    # Explicit link arguments replace raw session arguments, as on the graphics path.
    assert (
        _read_architecture(device, raw_session, {"downstream_args": []})
        == max(_targets(device)) * 10
    )
    assert options.downstream_args == ["-arch", f"sm_{target}"]
    # Raw arguments follow even an explicit typed target, leaving precedence to NVRTC.
    exact = device.create_slang_session(
        compiler_options={"cuda_architecture": max(_targets(device))}
    )
    assert (
        _read_architecture(device, exact, {"downstream_args": [f"-arch=compute_{target}"]})
        == target * 10
    )
    raw_exact = device.create_slang_session(
        compiler_options={
            "cuda_architecture": max(_targets(device)),
            "downstream_args": ["--gpu-architecture", f"compute_{target}"],
        }
    )
    assert _read_architecture(device, raw_exact) == target * 10


@pytest.mark.parametrize("device_type", NON_CUDA_DEVICE_TYPES)
def test_cuda_exact_request_on_non_cuda(device_type: spy.DeviceType) -> None:
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        spy.Device(type=device_type, compiler_options={"cuda_architecture": 90})


@pytest.mark.parametrize("device_type", CUDA_DEVICE_TYPES)
def test_cuda_link_target_cache_identity(tmp_path: Path, device_type: spy.DeviceType):
    targets = _targets(helpers.get_device(device_type))
    keys = {}
    # Recreate the device to exercise persistent cache identity across Slang sessions.
    for cache_pass in range(2):
        device = spy.Device(
            type=device_type,
            enable_compilation_reports=True,
            shader_cache_path=tmp_path,
        )
        try:
            module = device.load_module_from_source(
                "target_cache_identity",
                """
                uint compiled_architecture() { __intrinsic_asm "(__CUDA_ARCH__)"; }
                uint cache_value()
                {
                    __requirePrelude("#ifndef CACHE_VALUE\\n#define CACHE_VALUE 0\\n#endif\\n");
                    __intrinsic_asm "(CACHE_VALUE)";
                }
                [shader("compute")][numthreads(1,1,1)]
                void entry(RWStructuredBuffer<uint> output)
                {
                    output[0] = compiled_architecture();
                    output[1] = cache_value();
                }
                """,
            )
            # Include unchanged/default links, two architectures, unrelated NVRTC options, and warm hits.
            # This must also pass on Slang versions affected by shader-slang/slang#13197.
            requests = list(
                dict.fromkeys([(None, 0), (min(targets), 1), (max(targets), 1), (min(targets), 2)])
            )
            for target, value in requests:
                link_options = None
                if target is not None:
                    link_options = spy.SlangLinkOptions(
                        {
                            "downstream_args": [
                                f"-arch=compute_{target}",
                                f"-DCACHE_VALUE={value}",
                            ]
                        }
                    )
                program = device.link_program(
                    [module],
                    [module.entry_point("entry")],
                    link_options=link_options,
                )
                kernel = device.create_compute_kernel(program)
                output = device.create_buffer(
                    data=np.zeros(2, dtype=np.uint32), usage=spy.BufferUsage.unordered_access
                )
                kernel.dispatch(thread_count=[1, 1, 1], output=output)
                assert list(output.to_numpy().view(np.uint32)) == [
                    (target or max(targets)) * 10,
                    value,
                ]
                report = program.get_compilation_report()["entry_point_reports"][0]
                key = report["cache_key"]
                assert key is not None
                assert report["is_cached"] == (cache_pass > 0)
                request = (target, value)
                if request in keys:
                    assert key == keys[request]
                else:
                    assert key not in keys.values()
                    keys[request] = key
        finally:
            device.close()


@pytest.mark.parametrize(
    "pipeline_type,target_mode",
    [("compute", "lower"), ("optix", "exact"), ("optix", "link_lower")],
)
@pytest.mark.parametrize("device_type", CUDA_DEVICE_TYPES)
def test_cuda_target_execution(
    tmp_path: Path, pipeline_type: str, target_mode: str, device_type: spy.DeviceType
):
    targets = _targets(helpers.get_device(device_type))
    device = spy.Device(type=device_type, compiler_options={})
    try:
        if pipeline_type == "optix" and not device.has_feature(spy.Feature.ray_tracing):
            pytest.skip("Ray tracing not supported")
        target = max(targets) if target_mode == "exact" else min(targets)
        session = device.create_slang_session(
            compiler_options={
                "cuda_architecture": target if target_mode in ("lower", "exact") else None,
                "dump_intermediates": True,
                "dump_intermediates_prefix": str(tmp_path / "target-"),
            }
        )
        source = (
            '[shader("compute")][numthreads(1,1,1)] void target_entry(RWStructuredBuffer<uint> output) { output[0] = 42; }'
            if pipeline_type == "compute"
            else 'RWStructuredBuffer<uint> output; [shader("raygeneration")] void target_entry() { output[0] = 42; }'
        )
        module = session.load_module_from_source("target_execution", source)
        link_options = (
            spy.SlangLinkOptions({"downstream_args": [f"-arch=compute_{target}"]})
            if target_mode == "link_lower"
            else None
        )
        program = session.link_program(
            [module], [module.entry_point("target_entry")], link_options=link_options
        )
        output = device.create_buffer(
            data=np.zeros(1, dtype=np.uint32), usage=spy.BufferUsage.unordered_access
        )
        if pipeline_type == "compute":
            device.create_compute_kernel(program).dispatch(thread_count=[1, 1, 1], output=output)
        else:
            pipeline = device.create_ray_tracing_pipeline(
                program=program, hit_groups=[], max_recursion=1
            )
            table = device.create_shader_table(
                program=program, ray_gen_entry_points=["target_entry"]
            )
            encoder = device.create_command_encoder()
            with encoder.begin_ray_tracing_pass() as ray_pass:
                root = ray_pass.bind_pipeline(pipeline, table)
                spy.ShaderCursor(root).output = output
                ray_pass.dispatch_rays(0, [1, 1, 1])
            device.submit_command_buffer(encoder.finish())
        assert output.to_numpy().view(np.uint32)[0] == 42
        ptx_files = list(tmp_path.glob("*.ptx"))
        assert ptx_files
        assert any(re.search(rf"\.target\s+sm_{target}\b", path.read_text()) for path in ptx_files)
        # Unpatched Slang may warn about its earlier default architecture; NVRTC uses our target.
        # Restore the warning check after https://github.com/shader-slang/slang/issues/13198 is fixed.
    finally:
        device.close()
