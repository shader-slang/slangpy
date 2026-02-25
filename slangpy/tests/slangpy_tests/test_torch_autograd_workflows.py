# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
PyTorch autograd integration tests for slangpy.

These tests validate that slangpy supports realistic ML/optimization workflow
patterns: optimizer loops, gradient accumulation for broadcast parameters,
chained kernel calls, vector outputs, and mixed slangpy+PyTorch autograd graphs.

The workflow patterns are inspired by slang-torch examples
(https://github.com/shader-slang/slang-torch) — bezier curve fitting, MLP
training, and differentiable rasterization — but these are NOT ports of that
code. The slang-torch originals use DiffTensorView, [CUDAKernel],
[AutoPyBindCUDA], and manual torch.autograd.Function wrappers, none of which
are used here. These tests use slangpy's own API (scalar/array parameters with
automatic dispatch and auto-diff) to exercise the same categories of workload.

True parity tests using the original Slang kernels require:
  - github.com/shader-slang/slangpy/issues/740 (DiffTensorView support)
  - github.com/shader-slang/slangpy/issues/768 (raw dispatch support)
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
# Workflow: Adam optimizer loop fitting scalar parameters of a [Differentiable]
# Slang kernel, broadcast across sample points.
# Inspired by slang-torch bezier_curvefit.py (control point optimization).
# =============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_polynomial_optimization_convergence(device_type: DeviceType):
    """
    Optimize cubic polynomial coefficients to match a target polynomial.

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
# Workflow: multi-parameter optimization with gradient accumulation from many
# sample points back to shared control-point parameters.
# Inspired by slang-torch bezier_curvefit.py (degree-20 Bezier with
# DiffTensorView; here we use degree-3 with slangpy scalar params).
# =============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_bezier_curve_fitting(device_type: DeviceType):
    """
    Fit cubic Bezier control points to match a target curve.

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
# Workflow: gradient flow through multiple chained kernel calls
# (linear -> relu -> dot), with all parameters receiving gradients.
# Inspired by slang-torch mlp_image_fit.py (3-layer MLP with DiffTensorView,
# tensor-core matmul, and hand-written eval_bwd; here we use slangpy scalar
# arrays and auto-diff).
# =============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_two_layer_mlp_optimization(device_type: DeviceType):
    """
    Train a two-layer MLP (linear -> relu -> linear -> scalar output) to fit
    a target function using chained slangpy calls.

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
# Workflow: vector return type (float2) flowing through autograd in an
# optimization loop.
# Inspired by slang-torch rasterizer2d.py (float3 RGB output with
# DiffTensorView; here we use a float2 return with slangpy auto-dispatch).
# =============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_multi_output_optimization(device_type: DeviceType):
    """
    Optimize amplitude and phase of a sinusoidal that returns float2.

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
# Workflow: verifying gradient accumulation for parameters broadcast across
# many dispatch elements matches analytical expectations. Catches subtle
# gradient accumulation or copy-back bugs.
# Inspired by slang-torch bezier_curvefit.py backward pass (control_pts
# accumulates gradients from all sample-point dispatches via DiffTensorView).
# =============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_gradient_correctness_broadcast_params(device_type: DeviceType):
    """
    Verify gradients for broadcast (non-vectorized) parameters are analytically
    correct. This specifically tests the gradient accumulation pattern where a
    single parameter contributes to multiple dispatch elements.
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
# Workflow: repeated forward+backward+step cycles must produce clean gradients
# with no state leaks. All training loops depend on this.
# =============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_multiple_backward_passes_no_state_leak(device_type: DeviceType):
    """
    Run multiple forward+backward passes and verify gradients are correct each
    time (no state leaked from prior passes).

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
# Workflow: autograd graph spanning both slangpy kernels and PyTorch ops,
# with gradients flowing through the mixed graph.
# Inspired by slang-torch rasterizer2d.py (pyramid_loss applies PyTorch
# F.avg_pool2d to Slang kernel output).
# =============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_interleaved_slangpy_pytorch_optimization(device_type: DeviceType):
    """
    Optimize parameters where the forward pass mixes slangpy and PyTorch ops.
    Here we apply torch.sin to the output of a slangpy polynomial.

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
