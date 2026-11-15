# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("parallel", [False, True])
def test_prepare(device_type: spy.DeviceType, parallel: bool) -> None:
    device = spy.Device(
        type=device_type,
        enable_ray_tracing=False,
        enable_compilation_reports=True,
        pipeline_compilation_mode=(
            spy.PipelineCompilationMode.parallel if parallel else spy.PipelineCompilationMode.serial
        ),
    )
    module = spy.Module.load_from_source(
        device,
        "prepare_test",
        "float square(float x) { return x*x; } float cube(float x) { return x*x*x; }",
    )
    values = np.arange(16, dtype=np.float32)
    x = spy.Tensor.from_numpy(device, values)
    outputs = [spy.Tensor.from_numpy(device, np.full_like(values, -1)) for _ in range(2)]
    calls = [module.square, module.cube]
    prepared = [function.prepare(x, _result=out) for function, out in zip(calls, outputs)]

    for function, out, call_data in zip(calls, outputs, prepared):
        assert function.prepare(x, _result=out) is call_data
        np.testing.assert_array_equal(out.to_numpy(), -1)
    assert not any(report["entry_point_reports"] for report in device.get_compilation_reports())

    device.create_pending_pipelines()
    reports = device.get_compilation_reports()
    assert sum(len(report["entry_point_reports"]) for report in reports) == 2
    assert sum(len(report["pipeline_reports"]) for report in reports) == 2
    # A compilation barrier does not execute the prepared calls.
    for out in outputs:
        np.testing.assert_array_equal(out.to_numpy(), -1)
    device.create_pending_pipelines()

    for power, (function, out, data) in enumerate(zip(calls, outputs, prepared), 2):
        function(x, _result=out)
        np.testing.assert_array_equal(out.to_numpy(), values**power)
        assert function.prepare(x, _result=out) is data
        assert device.get_compilation_reports() == reports


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_prepare_result_type_and_modifiers(device_type: spy.DeviceType) -> None:
    device = spy.Device(type=device_type, enable_ray_tracing=False)
    module = spy.Module.load_from_source(
        device, "prepare_result_type", "int add(int a, int b) { return a+b; }"
    )
    function = module.add.map((), ())
    data = function.prepare(2, 3, _result=spy.Tensor)
    assert function.return_type(spy.Tensor).prepare(2, 3) is data
    device.create_pending_pipelines()
    result = function(2, 3, _result=spy.Tensor)
    np.testing.assert_array_equal(result.to_numpy(), 5)
    assert function.prepare(4, 7, _result=spy.Tensor) is data
