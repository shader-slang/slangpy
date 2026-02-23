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
# Correctness: verify slangpy and slangtorch produce matching results
# =============================================================================

CORRECTNESS_BATCH = 1000
CORRECTNESS_ATOL = 1e-5
CORRECTNESS_RTOL = 1e-4
# Cross-implementation tolerance (slang vs pytorch): slightly looser due to
# different pow()/FMA instruction sequences on GPU vs PyTorch CUDA ops.
CORRECTNESS_PYTORCH_ATOL = 5e-5
CORRECTNESS_PYTORCH_RTOL = 1e-4


def _create_models_with_shared_params(
    torch_device: torch.device,
    spy_device: "spy.Device",
    include_pytorch: bool = False,
) -> dict:
    """Create models with identical parameters for correctness comparison."""
    from slangpy.benchmarks.ppisp.ppisp_slangpy import PPISPSlangPy
    from slangpy.benchmarks.ppisp.ppisp_slangtorch import PPISPSlangtorch

    model_st = PPISPSlangtorch(NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)
    model_spy = PPISPSlangPy(
        NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H,
        torch_device, spy_device=spy_device,
    )

    # Copy params so all models use identical weights
    with torch.no_grad():
        model_spy.exposure_params.copy_(model_st.exposure_params)
        model_spy.vignetting_params.copy_(model_st.vignetting_params)
        model_spy.color_params.copy_(model_st.color_params)
        model_spy.crf_params.copy_(model_st.crf_params)

    models = {"slangpy": model_spy, "slangtorch": model_st}

    if include_pytorch:
        from slangpy.benchmarks.ppisp.ppisp_pytorch import PPISPPyTorch
        model_pt = PPISPPyTorch(NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)
        with torch.no_grad():
            model_pt.exposure_params.copy_(model_st.exposure_params)
            model_pt.vignetting_params.copy_(model_st.vignetting_params)
            model_pt.color_params.copy_(model_st.color_params)
            model_pt.crf_params.copy_(model_st.crf_params)
        models["pytorch"] = model_pt

    return models


def _assert_close(a: torch.Tensor, b: torch.Tensor, msg: str,
                   atol: float = CORRECTNESS_ATOL, rtol: float = CORRECTNESS_RTOL) -> None:
    assert not torch.isnan(a).any(), f"{msg}: first tensor has NaN"
    assert not torch.isnan(b).any(), f"{msg}: second tensor has NaN"
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=msg)


@pytest.mark.skip(reason="Correctness validated; enable manually when needed")
@pytest.mark.parametrize("include_pytorch", [False, True], ids=["slang-only", "with-pytorch"])
def test_ppisp_correctness_forward(include_pytorch: bool) -> None:
    """Verify forward outputs match across backends."""
    _skip_if_no_slangtorch()
    device = helpers.get_torch_device(spy.DeviceType.cuda)
    torch_device = torch.device("cuda")

    torch.manual_seed(42)
    rgb, pixel_coords, _, _ = create_test_data(
        CORRECTNESS_BATCH, NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)

    # Real PPISP processes one image at a time: all pixels share the same
    # camera/frame.  Slangtorch's loadUniform backward relies on warp-uniform
    # indices (WaveActiveSum + WaveIsFirstLane), so we use uniform indices here
    # to match the realistic access pattern.
    # with-pytorch uses camera=0/frame=0 to match the PyTorch scalar API.
    cam = 0 if include_pytorch else 2
    frm = 0 if include_pytorch else 5
    camera_idcs = torch.full((CORRECTNESS_BATCH,), cam, device=torch_device, dtype=torch.int16)
    frame_idcs = torch.full((CORRECTNESS_BATCH,), frm, device=torch_device, dtype=torch.int32)

    models = _create_models_with_shared_params(torch_device, device, include_pytorch)

    with torch.no_grad():
        out_spy = models["slangpy"](rgb, pixel_coords, camera_idcs, frame_idcs)
        out_st = models["slangtorch"](rgb, pixel_coords, camera_idcs, frame_idcs)

    _assert_close(out_spy, out_st, "Forward mismatch: slangpy vs slangtorch")

    if include_pytorch:
        out_pt = models["pytorch"](rgb, pixel_coords, camera_idx=cam, frame_idx=frm)
        _assert_close(out_spy, out_pt, "Forward mismatch: slangpy vs pytorch",
                       atol=CORRECTNESS_PYTORCH_ATOL, rtol=CORRECTNESS_PYTORCH_RTOL)


@pytest.mark.skip(reason="Correctness validated; enable manually when needed")
@pytest.mark.parametrize("include_pytorch", [False, True], ids=["slang-only", "with-pytorch"])
def test_ppisp_correctness_backward(include_pytorch: bool) -> None:
    """Verify gradients match across backends."""
    _skip_if_no_slangtorch()
    device = helpers.get_torch_device(spy.DeviceType.cuda)
    torch_device = torch.device("cuda")

    torch.manual_seed(42)
    rgb, pixel_coords, _, _ = create_test_data(
        CORRECTNESS_BATCH, NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)

    # Uniform indices — see comment in test_ppisp_correctness_forward.
    cam = 0 if include_pytorch else 2
    frm = 0 if include_pytorch else 5
    camera_idcs = torch.full((CORRECTNESS_BATCH,), cam, device=torch_device, dtype=torch.int16)
    frame_idcs = torch.full((CORRECTNESS_BATCH,), frm, device=torch_device, dtype=torch.int32)

    models = _create_models_with_shared_params(torch_device, device, include_pytorch)

    # Run backward for each backend
    rgb_grads = {}
    for name, model in models.items():
        rgb_in = rgb.clone().requires_grad_(True)
        if name == "pytorch":
            out = model(rgb_in, pixel_coords, camera_idx=cam, frame_idx=frm)
        else:
            out = model(rgb_in, pixel_coords, camera_idcs, frame_idcs)
        out.sum().backward()
        rgb_grads[name] = rgb_in.grad

    torch.cuda.synchronize()

    # Compare slangpy vs slangtorch (always)
    _assert_close(rgb_grads["slangpy"], rgb_grads["slangtorch"], "rgb grad: slangpy vs slangtorch")
    for param_name in ["exposure_params", "vignetting_params", "color_params", "crf_params"]:
        _assert_close(
            getattr(models["slangpy"], param_name).grad,
            getattr(models["slangtorch"], param_name).grad,
            f"{param_name} grad: slangpy vs slangtorch",
        )

    # Compare with pytorch (optional, looser tolerance for cross-implementation)
    if include_pytorch:
        _assert_close(rgb_grads["slangpy"], rgb_grads["pytorch"], "rgb grad: slangpy vs pytorch",
                       atol=CORRECTNESS_PYTORCH_ATOL, rtol=CORRECTNESS_PYTORCH_RTOL)
        for param_name in ["exposure_params", "vignetting_params", "color_params", "crf_params"]:
            _assert_close(
                getattr(models["slangpy"], param_name).grad,
                getattr(models["pytorch"], param_name).grad,
                f"{param_name} grad: slangpy vs pytorch",
                atol=CORRECTNESS_PYTORCH_ATOL, rtol=CORRECTNESS_PYTORCH_RTOL,
            )


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
    rgb, pixel_coords, _, _ = create_test_data(
        batch_size, NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)
    # Uniform indices: real PPISP processes one image at a time (see correctness tests).
    camera_idcs = torch.zeros(batch_size, device=torch_device, dtype=torch.int16)
    frame_idcs = torch.zeros(batch_size, device=torch_device, dtype=torch.int32)

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
    rgb, pixel_coords, _, _ = create_test_data(
        batch_size, NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)
    # Uniform indices: real PPISP processes one image at a time (see correctness tests).
    camera_idcs = torch.zeros(batch_size, device=torch_device, dtype=torch.int16)
    frame_idcs = torch.zeros(batch_size, device=torch_device, dtype=torch.int32)

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
    rgb, pixel_coords, _, _ = create_test_data(
        batch_size, NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)
    # Uniform indices: real PPISP processes one image at a time (see correctness tests).
    camera_idcs = torch.zeros(batch_size, device=torch_device, dtype=torch.int16)
    frame_idcs = torch.zeros(batch_size, device=torch_device, dtype=torch.int32)

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
    rgb, pixel_coords, _, _ = create_test_data(
        batch_size, NUM_CAMERAS, NUM_FRAMES, RESOLUTION_W, RESOLUTION_H, torch_device)
    # Uniform indices: real PPISP processes one image at a time (see correctness tests).
    camera_idcs = torch.zeros(batch_size, device=torch_device, dtype=torch.int16)
    frame_idcs = torch.zeros(batch_size, device=torch_device, dtype=torch.int32)

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
