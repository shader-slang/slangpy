# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check the coverage progression promised by the image-processing example."""

from pathlib import Path

import pytest
import slangpy as spy
from slangpy.testing import helpers
from examples.shader_coverage.image_processing import SOURCE, ImageProcessor, make_inputs
from examples.shader_coverage.report import save_capture, summarize


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_image_example_coverage(device_type: spy.DeviceType, tmp_path: Path) -> None:
    if device_type not in (spy.DeviceType.vulkan, spy.DeviceType.cuda):
        pytest.skip("Shader coverage supports Vulkan and CUDA")
    scenarios = make_inputs()
    with spy.Device(
        type=device_type,
        enable_hot_reload=False,
        compiler_options={"coverage": {"counter_width": 32}},
    ) as device:
        processor = ImageProcessor(device)
        _, image, denoise = scenarios[0]
        processor.process(image, denoise)
        before = device.shader_coverage.snapshot()
        for _, image, denoise in scenarios[1:]:
            processor.process(image, denoise)
        after = device.shader_coverage.snapshot(reset=True)
        cleared = device.shader_coverage.snapshot()
        assert all(not any(program.counters) for program in cleared.programs)

    # Export also exercises the example's use of snapshots after closing the device.
    stats = []
    for label, snapshot, calls in (("before", before, 1), ("after", after, 3)):
        capture = save_capture(tmp_path, label, snapshot, SOURCE.read_text(encoding="utf-8"))
        current = summarize(capture, SOURCE)
        assert int(current["function_calls"]["denoiseToneMap"]) == calls * 240 * 320
        stats.append(current)
    first, last = stats
    assert first["generation_id"] == last["generation_id"]
    assert first["line"]["hit"] < first["line"]["total"]
    assert first["branch"]["hit"] < first["branch"]["total"]
    assert last["line"]["hit"] == last["line"]["total"]
    assert last["branch"]["hit"] == last["branch"]["total"]
