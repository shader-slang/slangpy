# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
PPISP pipeline benchmarks: PyTorch vs SlangPy vs Slangtorch.

Benchmarks a 4-stage differentiable ISP (Image Signal Processor) pipeline
(exposure, vignetting, color correction, CRF) across three backends.
Uses benchmark_python_function fixture for wall-clock timing with
torch.cuda.synchronize() to capture full GPU execution time.
"""

import pytest

import slangpy as spy
from slangpy.testing import helpers
from slangpy.testing.benchmark import BenchmarkPythonFunction

HAS_TORCH = False
try:
    import torch

    HAS_TORCH = True
except ImportError:
    pass

HAS_SLANGTORCH = False
try:
    import slangtorch  # noqa: F401

    HAS_SLANGTORCH = True
except ImportError:
    pass

# Benchmark parameters
NUM_CAMERAS = 6
NUM_FRAMES = 200
RESOLUTION_W = 1920
RESOLUTION_H = 1080

# Fixture parameters: 10 outer × 100 inner = 1000 total timed calls
ITERATIONS = 10
SUB_ITERATIONS = 100
WARMUP_ITERATIONS = 10


def _skip_if_no_torch() -> None:
    if not HAS_TORCH:
        pytest.skip("PyTorch is not installed")


def _skip_if_no_slangtorch() -> None:
    _skip_if_no_torch()
    if not HAS_SLANGTORCH:
        pytest.skip("slangtorch is not installed")


def create_test_data(
    batch_size: int, num_cameras: int, num_frames: int,
    resolution_w: int, resolution_h: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random test data matching the PPISP API."""
    rgb = torch.rand(batch_size, 3, device=device)
    pixel_coords = torch.stack([
        torch.rand(batch_size, device=device) * resolution_w,
        torch.rand(batch_size, device=device) * resolution_h,
    ], dim=-1)
    camera_idcs = torch.randint(0, num_cameras, (batch_size,), device=device, dtype=torch.int16)
    frame_idcs = torch.randint(0, num_frames, (batch_size,), device=device, dtype=torch.int32)
    return rgb, pixel_coords, camera_idcs, frame_idcs


# =============================================================================
# Forward benchmarks (torch.no_grad)
# =============================================================================


@pytest.mark.parametrize("batch_size", [100_000, 1_000_000])
def test_ppisp_forward_pytorch(
    batch_size: int,
    benchmark_python_function: BenchmarkPythonFunction,
) -> None:
    _skip_if_no_torch()
    device = helpers.get_torch_device(spy.DeviceType.cuda)
    torch_device = torch.device("cuda")

    from slangpy.benchmarks.ppisp.ppisp_pytorch import PPISPPyTorch

    model = PPISPPyTorch(NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)
    rgb, pixel_coords, _, _ = create_test_data(
        batch_size, NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)

    def run() -> None:
        with torch.no_grad():
            model(rgb, pixel_coords, camera_idx=0, frame_idx=0)
        torch.cuda.synchronize()

    benchmark_python_function(
        device, run,
        iterations=ITERATIONS, sub_iterations=SUB_ITERATIONS,
        warmup_iterations=WARMUP_ITERATIONS, sleeps=True,
    )


@pytest.mark.parametrize("batch_size", [100_000, 1_000_000])
def test_ppisp_forward_slangpy(
    batch_size: int,
    benchmark_python_function: BenchmarkPythonFunction,
) -> None:
    _skip_if_no_torch()
    device = helpers.get_torch_device(spy.DeviceType.cuda)
    torch_device = torch.device("cuda")

    from slangpy.benchmarks.ppisp.ppisp_slangpy import PPISPSlangPy

    model = PPISPSlangPy(
        NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H,
        torch_device, spy_device=device,
    )
    rgb, pixel_coords, camera_idcs, frame_idcs = create_test_data(
        batch_size, NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)

    def run() -> None:
        with torch.no_grad():
            model(rgb, pixel_coords, camera_idcs, frame_idcs)
        torch.cuda.synchronize()

    benchmark_python_function(
        device, run,
        iterations=ITERATIONS, sub_iterations=SUB_ITERATIONS,
        warmup_iterations=WARMUP_ITERATIONS, sleeps=True,
    )


@pytest.mark.parametrize("batch_size", [100_000, 1_000_000])
def test_ppisp_forward_slangtorch(
    batch_size: int,
    benchmark_python_function: BenchmarkPythonFunction,
) -> None:
    _skip_if_no_slangtorch()
    device = helpers.get_torch_device(spy.DeviceType.cuda)
    torch_device = torch.device("cuda")

    from slangpy.benchmarks.ppisp.ppisp_slangtorch import PPISPSlangtorch

    model = PPISPSlangtorch(NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)
    rgb, pixel_coords, camera_idcs, frame_idcs = create_test_data(
        batch_size, NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)

    def run() -> None:
        with torch.no_grad():
            model(rgb, pixel_coords, camera_idcs, frame_idcs)
        torch.cuda.synchronize()

    benchmark_python_function(
        device, run,
        iterations=ITERATIONS, sub_iterations=SUB_ITERATIONS,
        warmup_iterations=WARMUP_ITERATIONS, sleeps=True,
    )


# =============================================================================
# Backward benchmarks (forward + backward)
# =============================================================================


@pytest.mark.parametrize("batch_size", [100_000, 1_000_000])
def test_ppisp_backward_pytorch(
    batch_size: int,
    benchmark_python_function: BenchmarkPythonFunction,
) -> None:
    _skip_if_no_torch()
    device = helpers.get_torch_device(spy.DeviceType.cuda)
    torch_device = torch.device("cuda")

    from slangpy.benchmarks.ppisp.ppisp_pytorch import PPISPPyTorch

    model = PPISPPyTorch(NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)
    rgb, pixel_coords, _, _ = create_test_data(
        batch_size, NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)

    def run() -> None:
        rgb_copy = rgb.clone().requires_grad_(True)
        output = model(rgb_copy, pixel_coords, camera_idx=0, frame_idx=0)
        output.sum().backward()
        torch.cuda.synchronize()

    benchmark_python_function(
        device, run,
        iterations=ITERATIONS, sub_iterations=SUB_ITERATIONS,
        warmup_iterations=WARMUP_ITERATIONS, sleeps=True,
    )


@pytest.mark.parametrize("batch_size", [100_000, 1_000_000])
def test_ppisp_backward_slangpy(
    batch_size: int,
    benchmark_python_function: BenchmarkPythonFunction,
) -> None:
    _skip_if_no_torch()
    device = helpers.get_torch_device(spy.DeviceType.cuda)
    torch_device = torch.device("cuda")

    from slangpy.benchmarks.ppisp.ppisp_slangpy import PPISPSlangPy

    model = PPISPSlangPy(
        NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H,
        torch_device, spy_device=device,
    )
    rgb, pixel_coords, camera_idcs, frame_idcs = create_test_data(
        batch_size, NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)

    def run() -> None:
        rgb_copy = rgb.clone().requires_grad_(True)
        output = model(rgb_copy, pixel_coords, camera_idcs, frame_idcs)
        output.sum().backward()
        torch.cuda.synchronize()

    benchmark_python_function(
        device, run,
        iterations=ITERATIONS, sub_iterations=SUB_ITERATIONS,
        warmup_iterations=WARMUP_ITERATIONS, sleeps=True,
    )


@pytest.mark.parametrize("batch_size", [100_000, 1_000_000])
def test_ppisp_backward_slangtorch(
    batch_size: int,
    benchmark_python_function: BenchmarkPythonFunction,
) -> None:
    _skip_if_no_slangtorch()
    device = helpers.get_torch_device(spy.DeviceType.cuda)
    torch_device = torch.device("cuda")

    from slangpy.benchmarks.ppisp.ppisp_slangtorch import PPISPSlangtorch

    model = PPISPSlangtorch(NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)
    rgb, pixel_coords, camera_idcs, frame_idcs = create_test_data(
        batch_size, NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)

    def run() -> None:
        rgb_copy = rgb.clone().requires_grad_(True)
        output = model(rgb_copy, pixel_coords, camera_idcs, frame_idcs)
        output.sum().backward()
        torch.cuda.synchronize()

    benchmark_python_function(
        device, run,
        iterations=ITERATIONS, sub_iterations=SUB_ITERATIONS,
        warmup_iterations=WARMUP_ITERATIONS, sleeps=True,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
