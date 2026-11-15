# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc

import pytest

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("parallel", [False, True])
def test_pending_pipeline_lifetimes(device_type: spy.DeviceType, parallel: bool) -> None:
    device = spy.Device(
        type=device_type,
        enable_ray_tracing=False,
        enable_compilation_reports=True,
        pipeline_compilation_mode=(
            spy.PipelineCompilationMode.parallel if parallel else spy.PipelineCompilationMode.serial
        ),
    )
    device.create_pending_pipelines()
    module = device.load_module_from_source(
        "pending_lifetime", '[shader("compute")][numthreads(1,1,1)] void compute_main() {}'
    )
    program = device.link_program([module], [module.entry_point("compute_main")])
    discarded = device.create_compute_pipeline(
        program, compilation_policy=spy.PipelineCompilationPolicy.deferred
    )
    del discarded
    gc.collect()
    device.create_pending_pipelines()
    assert not program.get_compilation_report()["pipeline_reports"]
    assert not program.get_compilation_report()["entry_point_reports"]

    # Two live wrappers share one shader program but require two backend pipelines.
    pipelines = [
        device.create_compute_pipeline(
            program, compilation_policy=spy.PipelineCompilationPolicy.deferred
        )
        for _ in range(2)
    ]
    device.create_pending_pipelines()
    report = program.get_compilation_report()
    assert len(report["entry_point_reports"]) == 1
    assert len(report["pipeline_reports"]) == 2
    device.create_pending_pipelines()
    assert program.get_compilation_report() == report
    del pipelines
    gc.collect()
    device.create_pending_pipelines()

    # Resolving through the existing encoder path must also unregister pending work.
    pipeline = device.create_compute_pipeline(
        program, compilation_policy=spy.PipelineCompilationPolicy.deferred
    )
    encoder = device.create_command_encoder()
    compute = encoder.begin_compute_pass()
    compute.bind_pipeline(pipeline)
    compute.dispatch([1, 1, 1])
    compute.end()
    command_buffer = encoder.finish()
    report = program.get_compilation_report()
    assert len(report["pipeline_reports"]) == 3
    device.create_pending_pipelines()
    assert program.get_compilation_report() == report
    device.submit_command_buffer(command_buffer)
    device.wait_for_idle()
