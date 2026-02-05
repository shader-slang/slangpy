# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
PyTorch gradient parity tests for SlangPy.

These tests verify that SlangPy implementations produce identical gradients
to their PyTorch equivalents by:
1. Creating a control PyTorch Sequential with standard nodes
2. Creating a test Sequential where one node is replaced with SlangPy
3. Running one training step on both with identical inputs/weights
4. Comparing gradients from the SlangPy node against the PyTorch node

This approach can be generalized to test multiple node types and input slicing.
"""

import pytest
import sys
from typing import Any, Callable, Optional

from slangpy import DeviceType
from slangpy.testing import helpers

try:
    import torch
    import torch.nn as nn
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)

# Type alias for loss functions: takes (output, target) -> scalar loss
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

if sys.platform == "darwin":
    pytest.skip("PyTorch requires CUDA, that is not available on macOS", allow_module_level=True)


def compare_gradients(
    name: str,
    slangpy_grad: torch.Tensor,
    pytorch_grad: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-4,
):
    """Compare gradients from SlangPy and PyTorch implementations."""
    assert (
        slangpy_grad.shape == pytorch_grad.shape
    ), f"{name}: Shape mismatch - SlangPy {slangpy_grad.shape} vs PyTorch {pytorch_grad.shape}"

    if not torch.allclose(slangpy_grad, pytorch_grad, rtol=rtol, atol=atol):
        max_diff = torch.max(torch.abs(slangpy_grad - pytorch_grad)).item()
        mean_diff = torch.mean(torch.abs(slangpy_grad - pytorch_grad)).item()
        raise AssertionError(
            f"{name}: Gradient mismatch - max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}\n"
            f"SlangPy grad:\n{slangpy_grad}\n\nPyTorch grad:\n{pytorch_grad}"
        )


# =============================================================================
# Slang Implementations of Standard Operations
# =============================================================================

# All activation functions in one module for convenience
SLANG_ACTIVATIONS = """
[Differentiable]
float slang_relu(float x) {
    return max(0.0f, x);
}

[Differentiable]
float slang_leaky_relu(float x) {
    float alpha = 0.01f;  // Standard leaky ReLU slope
    return x > 0.0f ? x : alpha * x;
}

[Differentiable]
float slang_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

[Differentiable]
float slang_tanh(float x) {
    return tanh(x);
}

[Differentiable]
float slang_silu(float x) {
    // SiLU/Swish: x * sigmoid(x)
    return x / (1.0f + exp(-x));
}

[Differentiable]
float slang_softplus(float x) {
    // Softplus: log(1 + exp(x))
    // Use numerically stable version for large x
    if (x > 20.0f) {
        return x;
    }
    return log(1.0f + exp(x));
}

[Differentiable]
float slang_square(float x) {
    return x * x;
}
"""


# =============================================================================
# SlangPy Module Wrappers
# =============================================================================


class SlangpyActivation(nn.Module):
    """Generic SlangPy activation wrapper that can call any named function."""

    def __init__(self, slang_module: Any, func_name: str):
        super().__init__()
        self.slang_module = slang_module
        self.func_name = func_name
        # Get the function from the module
        self.func = getattr(slang_module, func_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)


class PyTorchSquare(nn.Module):
    """PyTorch square operation for comparison."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


# =============================================================================
# Activation Function Specifications for Parametrized Testing
# =============================================================================


class ActivationSpec:
    """Specification for an activation function to test."""

    def __init__(
        self,
        name: str,
        slang_func_name: str,
        pytorch_module_factory: Callable[[], nn.Module],
        rtol: float = 1e-4,
        atol: float = 1e-4,
    ):
        super().__init__()
        self.name = name
        self.slang_func_name = slang_func_name
        self.pytorch_module_factory = pytorch_module_factory
        self.rtol = rtol
        self.atol = atol

    def __repr__(self) -> str:
        return self.name


# Define all activation functions to test
ACTIVATION_SPECS = [
    ActivationSpec("relu", "slang_relu", nn.ReLU),
    ActivationSpec("leaky_relu", "slang_leaky_relu", lambda: nn.LeakyReLU(0.01)),
    ActivationSpec("sigmoid", "slang_sigmoid", nn.Sigmoid),
    ActivationSpec("tanh", "slang_tanh", nn.Tanh),
    ActivationSpec("silu", "slang_silu", nn.SiLU),
    ActivationSpec("softplus", "slang_softplus", nn.Softplus, rtol=1e-3, atol=1e-3),
]


# =============================================================================
# Test Infrastructure
# =============================================================================


def default_mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Default MSE loss function."""
    return ((output - target) ** 2).mean()


def compute_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    loss_fn: LossFn = default_mse_loss,
) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """
    Run forward + backward pass and return all gradients.

    Args:
        model: The neural network model
        x: Input tensor (should have requires_grad=True to get input gradients)
        target: Target tensor for loss computation
        loss_fn: Loss function taking (output, target) -> scalar loss

    Returns:
        - Dict mapping parameter names to their gradients
        - Input gradient (if input requires_grad)
    """
    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()

    # Forward pass
    output = model(x)

    # Compute loss using provided loss function
    loss = loss_fn(output, target)

    # Backward pass
    loss.backward()

    # Collect parameter gradients
    param_gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_gradients[name] = param.grad.clone()

    # Get input gradient
    input_grad = x.grad.clone() if x.grad is not None else None

    return param_gradients, input_grad


def copy_model_weights(src: nn.Module, dst: nn.Module):
    """Copy weights from src model to dst model (for layers that have weights)."""
    src_params = dict(src.named_parameters())
    dst_params = dict(dst.named_parameters())

    with torch.no_grad():
        for name, param in dst_params.items():
            if name in src_params:
                param.copy_(src_params[name])


def run_gradient_parity_test(
    control_model: nn.Module,
    test_model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    control_loss_fn: LossFn = default_mse_loss,
    test_loss_fn: Optional[LossFn] = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
):
    """
    Compare gradients between control and test setups.

    This is the main test helper. It:
    1. Runs forward + backward on both models
    2. Compares input gradients
    3. Compares all parameter gradients

    Args:
        control_model: The reference PyTorch model
        test_model: The model with SlangPy component(s)
        x: Input tensor (will be cloned for each model)
        target: Target tensor for loss computation
        control_loss_fn: Loss function for control model
        test_loss_fn: Loss function for test model (defaults to control_loss_fn)
        rtol: Relative tolerance for gradient comparison
        atol: Absolute tolerance for gradient comparison
    """
    if test_loss_fn is None:
        test_loss_fn = control_loss_fn

    # Create separate input tensors for separate backward passes
    x_control = x.clone().detach().requires_grad_(True)
    x_test = x.clone().detach().requires_grad_(True)

    # Compute gradients for both
    control_grads, control_input_grad = compute_gradients(
        control_model, x_control, target, control_loss_fn
    )
    test_grads, test_input_grad = compute_gradients(test_model, x_test, target, test_loss_fn)

    # Compare input gradients
    if control_input_grad is not None and test_input_grad is not None:
        compare_gradients("Input gradient", test_input_grad, control_input_grad, rtol, atol)

    # Compare all parameter gradients
    for name in control_grads:
        if name in test_grads:
            compare_gradients(
                f"Parameter '{name}'", test_grads[name], control_grads[name], rtol, atol
            )


# =============================================================================
# SlangPy Loss Function Implementations
# =============================================================================

SLANG_MSE_LOSS = """
[Differentiable]
float mse_element(float prediction, float target) {
    float diff = prediction - target;
    return diff * diff;
}
"""


class SlangpyMSELoss:
    """SlangPy-based MSE loss that can be used as a loss function."""

    def __init__(self, slang_module: Any):
        super().__init__()
        self.slang_module = slang_module

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Element-wise squared difference, then mean
        return self.slang_module.mse_element(output, target).mean()


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("activation_spec", ACTIVATION_SPECS, ids=lambda s: s.name)
def test_activation_gradient_parity(device_type: DeviceType, activation_spec: ActivationSpec):
    """
    Compare gradients when an activation is replaced with SlangPy implementation.

    Control: Linear(8, 16) -> PyTorchActivation -> Linear(16, 4)
    Test:    Linear(8, 16) -> SlangpyActivation -> Linear(16, 4)

    Parametrized over multiple activation functions.
    """
    device = helpers.get_torch_device(device_type)
    slang_module = helpers.create_module(device, SLANG_ACTIVATIONS)

    in_features = 8
    hidden_features = 16
    out_features = 4
    batch_size = 32

    # Create control model (pure PyTorch)
    control_model = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        activation_spec.pytorch_module_factory(),
        nn.Linear(hidden_features, out_features),
    ).cuda()

    # Create test model with SlangPy activation
    test_model = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        SlangpyActivation(slang_module, activation_spec.slang_func_name),
        nn.Linear(hidden_features, out_features),
    ).cuda()

    # Copy weights from control to test model
    copy_model_weights(control_model, test_model)

    # Create test inputs
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, device="cuda")
    target = torch.randn(batch_size, out_features, device="cuda")

    # Run comparison with activation-specific tolerances
    run_gradient_parity_test(
        control_model,
        test_model,
        x,
        target,
        rtol=activation_spec.rtol,
        atol=activation_spec.atol,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mse_loss_gradient_parity(device_type: DeviceType):
    """
    Compare gradients when MSE loss is replaced with SlangPy implementation.

    Both use: Linear(8, 16) -> ReLU -> Linear(16, 4)
    Control loss: PyTorch MSE
    Test loss: SlangPy MSE
    """
    device = helpers.get_torch_device(device_type)
    slang_module = helpers.create_module(device, SLANG_MSE_LOSS)

    in_features = 8
    hidden_features = 16
    out_features = 4
    batch_size = 32

    # Create identical models for both
    control_model = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, out_features),
    ).cuda()

    test_model = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, out_features),
    ).cuda()

    # Copy weights so models are identical
    copy_model_weights(control_model, test_model)

    # Create test inputs
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, device="cuda")
    target = torch.randn(batch_size, out_features, device="cuda")

    # Define loss functions
    pytorch_mse = lambda output, tgt: nn.functional.mse_loss(output, tgt)
    slangpy_mse = SlangpyMSELoss(slang_module)

    # Run comparison with different loss functions
    run_gradient_parity_test(
        control_model,
        test_model,
        x,
        target,
        control_loss_fn=pytorch_mse,
        test_loss_fn=slangpy_mse,
    )


# =============================================================================
# Slicing Tests
# =============================================================================


class SliceEveryOther(nn.Module):
    """Slice module that takes every other element along feature dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, ::2]


class SliceFirstHalf(nn.Module):
    """Slice module that takes the first half of features."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[1] // 2
        return x[:, :half]


class SliceSpec:
    """Specification for a slice operation to test."""

    def __init__(self, name: str, module_factory: Callable[[], nn.Module], is_contiguous: bool):
        super().__init__()
        self.name = name
        self.module_factory = module_factory
        self.is_contiguous = is_contiguous

    def __repr__(self) -> str:
        return self.name


SLICE_SPECS = [
    SliceSpec("strided", SliceEveryOther, is_contiguous=False),
    SliceSpec("contiguous", SliceFirstHalf, is_contiguous=True),
]

SLICE_POSITIONS = ["before_activation", "after_activation"]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("slice_spec", SLICE_SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("slice_position", SLICE_POSITIONS)
def test_slice_gradient_parity(
    device_type: DeviceType,
    slice_spec: SliceSpec,
    slice_position: str,
):
    """
    Test that SlangPy correctly handles tensor slices.

    Parameterized over:
    - Slice type: strided (non-contiguous) vs contiguous
    - Slice position: before or after the activation

    This addresses Chris's concern about "clever slicing functionality".
    Tests both that SlangPy can handle sliced inputs AND that gradients
    back-propagate correctly through slicing after SlangPy operations.
    """
    device = helpers.get_torch_device(device_type)
    slang_module = helpers.create_module(device, SLANG_ACTIVATIONS)

    in_features = 8
    hidden_features = 16
    post_slice_features = hidden_features // 2  # Both slice types halve the features
    out_features = 4
    batch_size = 32

    # Build models based on slice position
    if slice_position == "before_activation":
        # Linear -> Slice -> Activation -> Linear
        # Tests: SlangPy receiving sliced (possibly non-contiguous) tensor
        control_model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            slice_spec.module_factory(),
            nn.ReLU(),
            nn.Linear(post_slice_features, out_features),
        ).cuda()

        test_model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            slice_spec.module_factory(),
            SlangpyActivation(slang_module, "slang_relu"),
            nn.Linear(post_slice_features, out_features),
        ).cuda()
    else:
        # Linear -> Activation -> Slice -> Linear
        # Tests: Gradients back-propagating through slice after SlangPy
        control_model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            slice_spec.module_factory(),
            nn.Linear(post_slice_features, out_features),
        ).cuda()

        test_model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            SlangpyActivation(slang_module, "slang_relu"),
            slice_spec.module_factory(),
            nn.Linear(post_slice_features, out_features),
        ).cuda()

    # Copy weights
    copy_model_weights(control_model, test_model)

    # Create test inputs
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, device="cuda")
    target = torch.randn(batch_size, out_features, device="cuda")

    # Run comparison
    run_gradient_parity_test(control_model, test_model, x, target)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_transposed_tensor_gradient_parity(device_type: DeviceType):
    """
    Test that SlangPy correctly handles transposed (non-contiguous) tensors.

    Structure:
        Input (batch, seq, features) -> Transpose -> Activation -> Transpose back

    Transpose creates a view with non-trivial strides.
    """
    device = helpers.get_torch_device(device_type)
    slang_module = helpers.create_module(device, SLANG_ACTIVATIONS)

    batch_size = 16
    seq_len = 8
    features = 4

    # We'll build this manually since transpose doesn't fit Sequential well
    torch.manual_seed(42)
    x_control = torch.randn(batch_size, seq_len, features, device="cuda", requires_grad=True)
    x_test = x_control.clone().detach().requires_grad_(True)
    target = torch.randn(batch_size, seq_len, features, device="cuda")

    # Control: transpose -> ReLU -> transpose back
    x_control_t = x_control.transpose(1, 2)  # (batch, features, seq)
    y_control_t = torch.relu(x_control_t)
    y_control = y_control_t.transpose(1, 2)  # (batch, seq, features)
    loss_control = ((y_control - target) ** 2).mean()
    loss_control.backward()

    # Test: transpose -> SlangPy ReLU -> transpose back
    x_test_t = x_test.transpose(1, 2)
    y_test_t = slang_module.slang_relu(x_test_t)
    y_test = y_test_t.transpose(1, 2)
    loss_test = ((y_test - target) ** 2).mean()
    loss_test.backward()

    # Compare input gradients
    assert x_control.grad is not None and x_test.grad is not None
    compare_gradients("Transposed tensor input gradient", x_test.grad, x_control.grad)


# =============================================================================
# Multi-Kernel Sequence Tests
# =============================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_consecutive_slangpy_ops(device_type: DeviceType):
    """
    Test consecutive SlangPy operations in a sequence.

    Structure:
        Linear -> SlangPy(relu) -> SlangPy(sigmoid) -> Linear

    Control:
        Linear -> PyTorch(ReLU) -> PyTorch(Sigmoid) -> Linear

    Tests that gradients correctly flow through multiple consecutive SlangPy ops.
    """
    device = helpers.get_torch_device(device_type)
    slang_module = helpers.create_module(device, SLANG_ACTIVATIONS)

    in_features = 8
    hidden_features = 16
    out_features = 4
    batch_size = 32

    # Control: PyTorch -> PyTorch
    control_model = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.ReLU(),
        nn.Sigmoid(),
        nn.Linear(hidden_features, out_features),
    ).cuda()

    # Test: SlangPy -> SlangPy
    test_model = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        SlangpyActivation(slang_module, "slang_relu"),
        SlangpyActivation(slang_module, "slang_sigmoid"),
        nn.Linear(hidden_features, out_features),
    ).cuda()

    copy_model_weights(control_model, test_model)

    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, device="cuda")
    target = torch.randn(batch_size, out_features, device="cuda")

    run_gradient_parity_test(control_model, test_model, x, target)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_interleaved_slangpy_pytorch_ops(device_type: DeviceType):
    """
    Test interleaved SlangPy and PyTorch operations.

    Structure:
        Linear -> SlangPy(relu) -> PyTorch(Tanh) -> SlangPy(sigmoid) -> Linear

    Control:
        Linear -> PyTorch(ReLU) -> PyTorch(Tanh) -> PyTorch(Sigmoid) -> Linear

    Tests that gradients correctly flow through interleaved SlangPy/PyTorch ops.
    """
    device = helpers.get_torch_device(device_type)
    slang_module = helpers.create_module(device, SLANG_ACTIVATIONS)

    in_features = 8
    hidden_features = 16
    out_features = 4
    batch_size = 32

    # Control: all PyTorch
    control_model = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.ReLU(),
        nn.Tanh(),
        nn.Sigmoid(),
        nn.Linear(hidden_features, out_features),
    ).cuda()

    # Test: SlangPy -> PyTorch -> SlangPy
    test_model = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        SlangpyActivation(slang_module, "slang_relu"),
        nn.Tanh(),  # PyTorch in the middle
        SlangpyActivation(slang_module, "slang_sigmoid"),
        nn.Linear(hidden_features, out_features),
    ).cuda()

    copy_model_weights(control_model, test_model)

    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, device="cuda")
    target = torch.randn(batch_size, out_features, device="cuda")

    run_gradient_parity_test(control_model, test_model, x, target)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_three_consecutive_slangpy_ops(device_type: DeviceType):
    """
    Test three consecutive SlangPy operations.

    Structure:
        Linear -> SlangPy(relu) -> SlangPy(tanh) -> SlangPy(sigmoid) -> Linear

    Tests gradient flow through a longer chain of SlangPy operations.
    """
    device = helpers.get_torch_device(device_type)
    slang_module = helpers.create_module(device, SLANG_ACTIVATIONS)

    in_features = 8
    hidden_features = 16
    out_features = 4
    batch_size = 32

    # Control: all PyTorch
    control_model = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.ReLU(),
        nn.Tanh(),
        nn.Sigmoid(),
        nn.Linear(hidden_features, out_features),
    ).cuda()

    # Test: three consecutive SlangPy ops
    test_model = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        SlangpyActivation(slang_module, "slang_relu"),
        SlangpyActivation(slang_module, "slang_tanh"),
        SlangpyActivation(slang_module, "slang_sigmoid"),
        nn.Linear(hidden_features, out_features),
    ).cuda()

    copy_model_weights(control_model, test_model)

    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, device="cuda")
    target = torch.randn(batch_size, out_features, device="cuda")

    run_gradient_parity_test(control_model, test_model, x, target)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
