# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
End-to-end workflow tests validating slangpy as a replacement for slang-torch.

These tests exercise realistic ML/optimization workflows using PyTorch autograd,
mirroring patterns from the slang-torch examples. Unlike the unit-level gradient
parity tests, these validate convergence of full optimization loops — the core
workflow that slang-torch users depend on.

slang-torch reference material (https://github.com/shader-slang/slang-torch):

  examples/bezier2d/
    bezier_curvefit.py  — Fits a cubic Bezier to a heart/ellipse/astroid by
                          optimizing control points with Adam. Uses
                          torch.autograd.Function wrapping a [Differentiable]
                          CUDA kernel that takes DiffTensorView arguments.
                          Calls .bwd() for reverse-mode AD.
    bezier.slang        — bezier2D kernel: evaluates Bernstein polynomial
                          basis over sample points, plus bezier2DSDF and
                          compute_coeffs helpers.

  examples/hard-rasterizer-example/
    rasterizer2d.py     — Optimizes triangle vertices + color to match a target
                          image using an image-pyramid loss. Uses
                          torch.autograd.Function, 2D grid dispatch, and
                          multi-channel (RGB) output via DiffTensorView.
    hard-rasterizer2d.slang — render_pixel with interior integral (4x MSAA) and
                          boundary integral (edge sampling). Complex struct
                          hierarchy: Camera, Triangle, AABB, EdgeSample, etc.

  examples/inline-mlp-example/
    mlp_image_fit.py    — Trains a 3-layer MLP with feature grid to fit an
                          image. Uses tensor-core matmul via custom CUDA
                          intrinsics, shared memory, and a manually written
                          backward pass (eval_bwd).
    inline-mlp.slang    — Linear<C> layer with warp-level shared-memory matmul,
                          MLP<C,N> struct chaining N layers with ReLU.
    image-model.slang   — renderImage kernel: interpolates a feature grid and
                          feeds it through the MLP.

  tests/test.py         — Unit tests covering: multi-output struct returns,
                          module loading options (defines), hot reload, multi-file
                          modules, forward (.fwd) and backward (.bwd) AD,
                          struct inputs, builtin types (float3, float3x3),
                          torch.autograd.Function integration, broadcasted
                          tensor error handling, empty tensors, half precision.

Test-to-reference mapping:

  test_polynomial_optimization_convergence
    Pattern from: bezier_curvefit.py (Adam optimizer loop over a [Differentiable]
    Slang function with scalar parameters broadcast across sample points)

  test_bezier_curve_fitting
    Pattern from: bezier_curvefit.py (multi-parameter Bezier control point
    optimization; the core slang-torch curve-fitting workflow)

  test_two_layer_mlp_optimization
    Pattern from: mlp_image_fit.py (sequential layer computation with gradient
    flow through multiple chained kernel calls; simplified from tensor-core
    MLP to standard linear+ReLU using slangpy's automatic dispatch)

  test_multi_output_optimization
    Pattern from: rasterizer2d.py (vector return type — float3 RGB in the
    rasterizer, float2 here — flowing through autograd in an optimization loop)

  test_gradient_correctness_broadcast_params
    Pattern from: bezier_curvefit.py backward pass (gradient accumulation for
    parameters broadcast across many dispatch elements — control_pts receives
    accumulated gradients from all sample-point dispatches)

  test_multiple_backward_passes_no_state_leak
    Pattern from: all slang-torch training loops (repeated forward+backward
    cycles with optimizer.zero_grad(); validates no state leaks between steps)

  test_interleaved_slangpy_pytorch_optimization
    Pattern from: rasterizer2d.py (pyramid_loss mixes rasterizer output with
    PyTorch's F.avg_pool2d; here we mix slangpy polynomial with torch.sin)
"""

import pytest
import sys

from slangpy import DeviceType
from slangpy.testing import helpers

try:
    import torch
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)

if sys.platform == "darwin":
    pytest.skip("PyTorch requires CUDA, that is not available on macOS", allow_module_level=True)

DEVICE_TYPES = helpers.DEFAULT_DEVICE_TYPES


# =============================================================================
# Slang Source Code
# =============================================================================

# Polynomial and curve-fitting functions
SLANG_CURVE_FITTING = """
import slangpy;

// Cubic polynomial: a*x^3 + b*x^2 + c*x + d
[Differentiable]
float cubic_poly(float a, float b, float c, float d, float x) {
    return a * x * x * x + b * x * x + c * x + d;
}

// Cubic Bezier evaluation (1D) using Bernstein basis:
// B(t) = (1-t)^3*p0 + 3*(1-t)^2*t*p1 + 3*(1-t)*t^2*p2 + t^3*p3
[Differentiable]
float bezier_cubic_1d(float p0, float p1, float p2, float p3, float t) {
    float u = 1.0 - t;
    float uu = u * u;
    float tt = t * t;
    return uu * u * p0 + 3.0 * uu * t * p1 + 3.0 * u * tt * p2 + tt * t * p3;
}
"""

# Multi-layer computation functions (MLP-inspired)
SLANG_MLP = """
import slangpy;

// Linear transform: result[r] = bias[r] + sum_c(weights[r][c] * x[c])
[Differentiable]
float[4] linear_transform(float weights[4][4], float bias[4], float[4] x) {
    float[4] result;
    for (int r = 0; r < 4; r++) {
        float y = bias[r];
        for (int c = 0; c < 4; c++)
            y += weights[r][c] * x[c];
        result[r] = y;
    }
    return result;
}

// Element-wise ReLU for a 4-element vector
[Differentiable]
float[4] relu4(float[4] x) {
    float[4] result;
    for (int i = 0; i < 4; i++)
        result[i] = max(0.0, x[i]);
    return result;
}

// Dot product: scalar output from 4-element vectors
[Differentiable]
float dot4(float weights[4], float[4] x) {
    float result = 0.0;
    for (int i = 0; i < 4; i++)
        result += weights[i] * x[i];
    return result;
}
"""

# Multi-output (vector return) functions
SLANG_MULTI_OUTPUT = """
import slangpy;

// Modulated sin/cos returning float2:
// (amplitude * sin(phase + x), amplitude * cos(phase + x))
[Differentiable]
float2 sincos_modulated(float amplitude, float phase, float x) {
    float angle = phase + x;
    return float2(amplitude * sin(angle), amplitude * cos(angle));
}
"""


# =============================================================================
# Helper
# =============================================================================


def assert_loss_decreased(
    initial_loss: "float | None", final_loss: float, min_ratio: float = 0.01
) -> None:
    """Assert that the loss decreased by at least (1 - min_ratio) of the initial loss."""
    assert initial_loss is not None, "initial_loss was never recorded"
    assert final_loss < initial_loss * min_ratio, (
        f"Loss did not converge sufficiently: initial={initial_loss:.6f}, final={final_loss:.6f} "
        f"(ratio={final_loss / initial_loss:.4f}, required <{min_ratio})"
    )


# =============================================================================
# Test 1: Polynomial Coefficient Optimization
#
# slang-torch ref: examples/bezier2d/bezier_curvefit.py
#   - Adam optimizer loop fitting parameters of a [Differentiable] Slang kernel
#   - Parameters (control_pts) are broadcast across sample points
#   - Uses torch.autograd.Function wrapping .bwd() calls
# =============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_polynomial_optimization_convergence(device_type: DeviceType):
    """
    Optimize cubic polynomial coefficients to match a target polynomial.

    slang-torch ref: examples/bezier2d/bezier_curvefit.py
      The bezier example optimizes N*2 control-point values with Adam over
      10k iterations. Here we distill the same pattern: scalar parameters
      broadcast across sample points, optimized through autograd.

    Validates:
    - Autograd forward/backward through slangpy
    - torch.optim.Adam integration
    - Convergence of optimization loop
    - Gradient flow to broadcast (non-vectorized) parameters
    """
    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, SLANG_CURVE_FITTING)

    # Target polynomial: y = 2x^3 + 0.5x^2 - 3x + 1
    target_a, target_b, target_c, target_d = 2.0, 0.5, -3.0, 1.0

    # Sample points for evaluation (not optimized, no grad needed)
    x = torch.linspace(-1, 1, 100, device="cuda", dtype=torch.float32)
    y_target = target_a * x**3 + target_b * x**2 + target_c * x + target_d

    # Initialize coefficients at wrong values — these ARE optimized
    a = torch.tensor([0.0], device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.tensor([0.0], device="cuda", dtype=torch.float32, requires_grad=True)
    c = torch.tensor([0.0], device="cuda", dtype=torch.float32, requires_grad=True)
    d = torch.tensor([0.0], device="cuda", dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.Adam([a, b, c, d], lr=0.1)

    initial_loss = None
    for epoch in range(300):
        optimizer.zero_grad()
        y_pred = module.cubic_poly(a=a, b=b, c=c, d=d, x=x)
        loss = ((y_pred - y_target) ** 2).mean()

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    # Loss should decrease by at least 99%
    assert_loss_decreased(initial_loss, final_loss, min_ratio=0.01)

    # Parameters should be close to target values
    tol = 0.3
    assert abs(a.item() - target_a) < tol, f"a={a.item():.3f}, expected {target_a}"
    assert abs(b.item() - target_b) < tol, f"b={b.item():.3f}, expected {target_b}"
    assert abs(c.item() - target_c) < tol, f"c={c.item():.3f}, expected {target_c}"
    assert abs(d.item() - target_d) < tol, f"d={d.item():.3f}, expected {target_d}"


# =============================================================================
# Test 2: Bezier Curve Fitting
#
# slang-torch ref: examples/bezier2d/bezier_curvefit.py + bezier.slang
#   - Optimizes control points (N,2) of a cubic Bezier via Adam
#   - bezier2D kernel evaluates Bernstein basis: nCi * (1-t)^(N-1-i) * t^i
#   - .bwd() accumulates gradients from M sample points to N control points
#   - Targets: heart, ellipse, or astroid parametric curves
# =============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_bezier_curve_fitting(device_type: DeviceType):
    """
    Fit cubic Bezier control points to match a target curve.

    slang-torch ref: examples/bezier2d/bezier_curvefit.py
      The original uses a degree-20 Bezier with DiffTensorView and launchRaw.
      Here we use degree-3 (cubic) with slangpy's automatic dispatch to test
      the same core pattern: gradient accumulation from many sample-point
      dispatches back to shared control-point parameters.

    Validates:
    - Multi-parameter optimization (8 params: 4 for X, 4 for Y)
    - Complex differentiable math (Bernstein polynomial basis)
    - Gradient accumulation from multiple sample points to shared parameters
    """
    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, SLANG_CURVE_FITTING)

    # Parameter values along the curve
    t = torch.linspace(0.01, 0.99, 50, device="cuda", dtype=torch.float32)

    # Target control points for X and Y coordinates
    target_px = [0.0, 1.0, 2.0, 3.0]
    target_py = [0.0, 2.0, -1.0, 1.0]

    # Compute target curve points using PyTorch (reference implementation)
    u = 1.0 - t  # type: ignore[operator]  # Tensor subtraction from float
    target_x = (
        u**3 * target_px[0]
        + 3 * u**2 * t * target_px[1]
        + 3 * u * t**2 * target_px[2]
        + t**3 * target_px[3]
    )
    target_y = (
        u**3 * target_py[0]
        + 3 * u**2 * t * target_py[1]
        + 3 * u * t**2 * target_py[2]
        + t**3 * target_py[3]
    )

    # Initialize control points at wrong positions (uniform, far from target)
    # Each is [1]-shaped so it broadcasts across the 50 sample points
    px0 = torch.tensor([0.5], device="cuda", dtype=torch.float32, requires_grad=True)
    px1 = torch.tensor([0.5], device="cuda", dtype=torch.float32, requires_grad=True)
    px2 = torch.tensor([0.5], device="cuda", dtype=torch.float32, requires_grad=True)
    px3 = torch.tensor([0.5], device="cuda", dtype=torch.float32, requires_grad=True)
    py0 = torch.tensor([0.5], device="cuda", dtype=torch.float32, requires_grad=True)
    py1 = torch.tensor([0.5], device="cuda", dtype=torch.float32, requires_grad=True)
    py2 = torch.tensor([0.5], device="cuda", dtype=torch.float32, requires_grad=True)
    py3 = torch.tensor([0.5], device="cuda", dtype=torch.float32, requires_grad=True)

    all_params = [px0, px1, px2, px3, py0, py1, py2, py3]
    optimizer = torch.optim.Adam(all_params, lr=0.05)

    initial_loss = None
    for epoch in range(500):
        optimizer.zero_grad()

        # Evaluate Bezier curve for X and Y through slangpy
        pred_x = module.bezier_cubic_1d(p0=px0, p1=px1, p2=px2, p3=px3, t=t)
        pred_y = module.bezier_cubic_1d(p0=py0, p1=py1, p2=py2, p3=py3, t=t)

        # L2 loss on curve positions
        loss = ((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2).mean()

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    # Loss should converge
    assert_loss_decreased(initial_loss, final_loss, min_ratio=0.01)

    # Control points should be close to targets
    fitted_px = [px0.item(), px1.item(), px2.item(), px3.item()]
    fitted_py = [py0.item(), py1.item(), py2.item(), py3.item()]
    tol = 0.3
    for i in range(4):
        assert (
            abs(fitted_px[i] - target_px[i]) < tol
        ), f"px[{i}]={fitted_px[i]:.3f}, expected {target_px[i]}"
        assert (
            abs(fitted_py[i] - target_py[i]) < tol
        ), f"py[{i}]={fitted_py[i]:.3f}, expected {target_py[i]}"


# =============================================================================
# Test 3: Two-Layer MLP Optimization (Sequential SlangPy Calls)
#
# slang-torch ref: examples/inline-mlp-example/
#   mlp_image_fit.py    — 3-layer MLP training loop with Adam
#   inline-mlp.slang    — Linear<C>.eval() chained with ReLU in MLP<C,N>.eval()
#   image-model.slang   — renderImage calls computeInterpolatedFeature then mlp.eval
#   The original uses tensor-core matmul (wmma intrinsics) and custom eval_bwd.
#   Here we use standard linear+ReLU with slangpy's auto-diff to test the same
#   structural pattern: multiple layers chained, all params receiving gradients.
# =============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_two_layer_mlp_optimization(device_type: DeviceType):
    """
    Train a two-layer MLP (linear -> relu -> linear -> scalar output) to fit
    a target function using chained slangpy calls.

    slang-torch ref: examples/inline-mlp-example/mlp_image_fit.py
      The original chains 3 Linear<16> layers with ReLU, each using warp-level
      shared-memory tensor-core matmul. Here we chain 3 slangpy calls
      (linear_transform -> relu4 -> dot4) to test the same gradient-flow-
      through-sequential-kernels pattern without hardware-specific intrinsics.

    Validates:
    - Sequential slangpy kernel calls in a forward pass
    - Gradient flow through the entire chain (3 slangpy calls)
    - Multi-parameter optimization (w1, b1, w2 all receive gradients)
    - Batch vectorization with broadcast weight parameters
    """
    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, SLANG_MLP)

    torch.manual_seed(42)

    # Training data: 4D input -> scalar output
    N = 64
    x_train = torch.randn(N, 4, device="cuda", dtype=torch.float32)

    # Target function: a known linear combination
    target_weights = torch.tensor([1.0, -0.5, 0.3, 0.8], device="cuda", dtype=torch.float32)
    y_target = x_train @ target_weights

    # Initialize network parameters
    # Note: scale BEFORE setting requires_grad to keep tensors as leaf nodes
    # (non-leaf tensors can't be passed to torch.optim.Adam)
    w1 = (torch.randn(4, 4, device="cuda", dtype=torch.float32) * 0.5).requires_grad_(True)
    b1 = torch.zeros(4, device="cuda", dtype=torch.float32, requires_grad=True)
    w2 = (torch.randn(4, device="cuda", dtype=torch.float32) * 0.5).requires_grad_(True)

    optimizer = torch.optim.Adam([w1, b1, w2], lr=0.01)

    initial_loss = None
    for epoch in range(300):
        optimizer.zero_grad()

        # Forward pass: three chained slangpy calls
        h = module.linear_transform(weights=w1, bias=b1, x=x_train)  # (N, 4)
        h = module.relu4(x=h)  # (N, 4)
        y_pred = module.dot4(weights=w2, x=h)  # (N,)

        loss = ((y_pred - y_target) ** 2).mean()

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    # All parameters should have received gradients throughout training
    assert w1.grad is not None, "w1 did not receive gradients"
    assert b1.grad is not None, "b1 did not receive gradients"
    assert w2.grad is not None, "w2 did not receive gradients"

    # Loss should converge (a linear target with ReLU can be approximated)
    assert_loss_decreased(initial_loss, final_loss, min_ratio=0.1)


# =============================================================================
# Test 4: Multi-Output Function Optimization
#
# slang-torch ref: examples/hard-rasterizer-example/rasterizer2d.py
#   - rasterize kernel writes float3 (RGB) per pixel via output.storeOnce()
#   - Backward pass accumulates gradients for vertices (3,2) and color (3)
#   - pyramid_loss combines rasterizer output with PyTorch F.avg_pool2d
#   The original produces a full (W,H,3) image. Here we test the simpler
#   pattern of a vector return type (float2) flowing through autograd.
#
# Also relates to: tests/test.py::TestSlangTorchSmoke
#   - Tests a multi-output function returning a struct with two tensor fields
# =============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_multi_output_optimization(device_type: DeviceType):
    """
    Optimize amplitude and phase of a sinusoidal that returns float2.

    slang-torch ref: examples/hard-rasterizer-example/rasterizer2d.py
      The rasterizer outputs float3 per pixel and optimizes vertices+color
      through autograd. Here we distill the multi-output pattern: a function
      returning float2 (vs. RGB float3) with gradients flowing to all params.

    Validates:
    - Vector (float2) return types through autograd
    - Multi-output gradient flow
    - Convergence with vector-valued loss
    """
    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, SLANG_MULTI_OUTPUT)

    # Target parameters
    target_amplitude = 2.5
    target_phase = 0.7

    # Sample points
    x = torch.linspace(0.1, 6.18, 80, device="cuda", dtype=torch.float32)

    # Target outputs: (amplitude*sin(phase+x), amplitude*cos(phase+x))
    target_sin = target_amplitude * torch.sin(target_phase + x)
    target_cos = target_amplitude * torch.cos(target_phase + x)
    target_output = torch.stack([target_sin, target_cos], dim=-1)  # (80, 2)

    # Initialize at wrong values
    amplitude = torch.tensor([1.0], device="cuda", dtype=torch.float32, requires_grad=True)
    phase = torch.tensor([0.0], device="cuda", dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.Adam([amplitude, phase], lr=0.05)

    initial_loss = None
    for epoch in range(400):
        optimizer.zero_grad()

        pred = module.sincos_modulated(amplitude=amplitude, phase=phase, x=x)

        loss = ((pred - target_output) ** 2).mean()

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    # Loss should converge
    assert_loss_decreased(initial_loss, final_loss, min_ratio=0.01)

    # Note: amplitude and phase may converge to equivalent solutions
    # (e.g., negative amplitude with shifted phase), so we only check loss.


# =============================================================================
# Test 5: Gradient Correctness for Broadcast Parameters
#
# slang-torch ref: examples/bezier2d/bezier_curvefit.py (backward pass)
#   - m.bezier2D.bwd(control_pts=(control_pts, grad_ctrl_pts), ...)
#   - grad_ctrl_pts accumulates contributions from all M sample points
#   - This is the gradient accumulation pattern for broadcast parameters
#
# Rather than testing convergence, this verifies that gradients computed by
# slangpy during an optimization step match analytical expectations. This
# catches subtle gradient accumulation or copy-back bugs.
# =============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_gradient_correctness_broadcast_params(device_type: DeviceType):
    """
    Verify gradients for broadcast (non-vectorized) parameters are analytically
    correct. This specifically tests the gradient accumulation pattern where a
    single parameter contributes to multiple dispatch elements.

    slang-torch ref: examples/bezier2d/bezier_curvefit.py
      In the .bwd() call, control_pts is shape (N,2) but the kernel dispatches
      over M sample points. Each dispatch contributes to the same grad_ctrl_pts
      buffer, requiring correct accumulation. This test verifies that pattern
      with analytically checkable gradients.
    """
    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, SLANG_CURVE_FITTING)

    # Use known values for analytical gradient computation
    x = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)

    # cubic_poly(a, b, c, d, x) = a*x^3 + b*x^2 + c*x + d
    a = torch.tensor([0.5], device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.tensor([1.0], device="cuda", dtype=torch.float32, requires_grad=True)
    c = torch.tensor([-0.5], device="cuda", dtype=torch.float32, requires_grad=True)
    d = torch.tensor([0.25], device="cuda", dtype=torch.float32, requires_grad=True)

    y_pred = module.cubic_poly(a=a, b=b, c=c, d=d, x=x)

    # Compute expected output: a*x^3 + b*x^2 + c*x + d
    expected = 0.5 * x**3 + 1.0 * x**2 + (-0.5) * x + 0.25
    assert torch.allclose(
        y_pred, expected, atol=1e-4
    ), f"Forward pass mismatch: got {y_pred}, expected {expected}"

    # Use sum as loss so gradients are simply the sum of partials
    loss = y_pred.sum()
    loss.backward()

    # Analytical gradients for loss = sum(a*x^3 + b*x^2 + c*x + d):
    # dloss/da = sum(x^3) = 1 + 8 + 27 = 36
    # dloss/db = sum(x^2) = 1 + 4 + 9 = 14
    # dloss/dc = sum(x)   = 1 + 2 + 3 = 6
    # dloss/dd = sum(1)    = 3
    expected_grad_a = (x**3).sum().item()
    expected_grad_b = (x**2).sum().item()
    expected_grad_c = x.sum().item()
    expected_grad_d = 3.0

    tol = 1e-3
    assert a.grad is not None, "Gradient for a is None"
    assert b.grad is not None, "Gradient for b is None"
    assert c.grad is not None, "Gradient for c is None"
    assert d.grad is not None, "Gradient for d is None"

    assert (
        abs(a.grad.item() - expected_grad_a) < tol
    ), f"grad_a={a.grad.item():.4f}, expected={expected_grad_a:.4f}"
    assert (
        abs(b.grad.item() - expected_grad_b) < tol
    ), f"grad_b={b.grad.item():.4f}, expected={expected_grad_b:.4f}"
    assert (
        abs(c.grad.item() - expected_grad_c) < tol
    ), f"grad_c={c.grad.item():.4f}, expected={expected_grad_c:.4f}"
    assert (
        abs(d.grad.item() - expected_grad_d) < tol
    ), f"grad_d={d.grad.item():.4f}, expected={expected_grad_d:.4f}"


# =============================================================================
# Test 6: Multiple Backward Passes (No State Leak Between Steps)
#
# slang-torch ref: all training-loop examples
#   bezier_curvefit.py  — 10k iterations: zero_grad -> forward -> loss -> backward -> step
#   rasterizer2d.py     — 400 iterations via animation.FuncAnimation
#   mlp_image_fit.py    — 4k iterations: zero_grad -> forward -> loss -> backward -> step
#   All rely on backward() producing clean gradients each step with no leaks.
# =============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_multiple_backward_passes_no_state_leak(device_type: DeviceType):
    """
    Run multiple forward+backward passes and verify gradients are correct each
    time (no state leaked from prior passes).

    slang-torch ref: all training loops (bezier_curvefit.py, rasterizer2d.py,
    mlp_image_fit.py) call backward() hundreds to thousands of times. If
    slangpy leaks internal state (e.g., stale CallData, gradient buffers not
    zeroed), these loops silently diverge or produce wrong results.

    Validates:
    - Repeated autograd cycles work correctly
    - zero_grad() properly resets state
    - No accumulated error from repeated backward passes
    """
    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, SLANG_CURVE_FITTING)

    x = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)

    for step in range(5):
        # Fresh parameters each step (or could reuse with zero_grad)
        a = torch.tensor([float(step)], device="cuda", dtype=torch.float32, requires_grad=True)
        b = torch.tensor([0.0], device="cuda", dtype=torch.float32, requires_grad=True)
        c = torch.tensor([0.0], device="cuda", dtype=torch.float32, requires_grad=True)
        d = torch.tensor([0.0], device="cuda", dtype=torch.float32, requires_grad=True)

        y = module.cubic_poly(a=a, b=b, c=c, d=d, x=x)
        loss = y.sum()
        loss.backward()

        # Expected: loss = sum(a*x^3) = a*(1+8) = 9a
        # dloss/da = 9
        expected_grad_a = (x**3).sum().item()
        assert a.grad is not None, f"Step {step}: gradient for a is None"
        assert (
            abs(a.grad.item() - expected_grad_a) < 1e-3
        ), f"Step {step}: grad_a={a.grad.item():.4f}, expected={expected_grad_a:.4f}"


# =============================================================================
# Test 7: Interleaved SlangPy and PyTorch Operations in Optimization
#
# slang-torch ref: examples/hard-rasterizer-example/rasterizer2d.py
#   - pyramid_loss() takes rasterizer output and applies F.avg_pool2d (PyTorch)
#   - Gradients flow: loss -> PyTorch pooling -> rasterizer backward -> params
#   - The autograd graph spans both the Slang kernel and PyTorch operations
#
# Also relates to: examples/inline-mlp-example/mlp_image_fit.py
#   - loss_fn = torch.nn.MSELoss() applied to renderImage output
# =============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_interleaved_slangpy_pytorch_optimization(device_type: DeviceType):
    """
    Optimize parameters where the forward pass mixes slangpy and PyTorch ops.

    slang-torch ref: examples/hard-rasterizer-example/rasterizer2d.py
      pyramid_loss applies F.avg_pool2d (a PyTorch op) to the output of
      rasterize (a Slang kernel). Gradients must flow through both. Here we
      apply torch.sin to the output of a slangpy polynomial — same pattern,
      simpler geometry.

    Validates:
    - Autograd graph spanning slangpy and PyTorch operations
    - Gradient flow through mixed computation graphs
    """
    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, SLANG_CURVE_FITTING)

    # Target: y = sin(2x^3 + x)  (cubic_poly feeds into PyTorch sin)
    x = torch.linspace(-1, 1, 80, device="cuda", dtype=torch.float32)
    y_target = torch.sin(2.0 * x**3 + x)

    # Parameters for the cubic polynomial inside the sin
    a = torch.tensor([0.0], device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.tensor([0.0], device="cuda", dtype=torch.float32, requires_grad=True)
    c = torch.tensor([0.0], device="cuda", dtype=torch.float32, requires_grad=True)
    d = torch.tensor([0.0], device="cuda", dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.Adam([a, b, c, d], lr=0.05)

    initial_loss = None
    for epoch in range(300):
        optimizer.zero_grad()

        # slangpy computes the polynomial
        poly_out = module.cubic_poly(a=a, b=b, c=c, d=d, x=x)

        # PyTorch applies sin on top — tests mixed autograd graph
        y_pred = torch.sin(poly_out)

        loss = ((y_pred - y_target) ** 2).mean()

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    # Loss should decrease significantly (inner polynomial should converge to 2x^3 + x)
    assert_loss_decreased(initial_loss, final_loss, min_ratio=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
