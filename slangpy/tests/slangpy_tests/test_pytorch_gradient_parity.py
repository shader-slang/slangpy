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

# ReLU: max(0, x)
SLANG_RELU = """
[Differentiable]
float relu(float x) {
    return max(0.0f, x);
}
"""

# Leaky ReLU: x if x > 0 else alpha * x
SLANG_LEAKY_RELU = """
[Differentiable]
float leaky_relu(float x) {
    float alpha = 0.01f;  // Standard leaky ReLU slope
    return x > 0.0f ? x : alpha * x;
}
"""

# Square (simple differentiable operation)
SLANG_SQUARE = """
[Differentiable]
float square(float x) {
    return x * x;
}
"""


# =============================================================================
# SlangPy Module Wrappers
# =============================================================================


class SlangpyReLU(nn.Module):
    """SlangPy-based ReLU that can be used in nn.Sequential."""

    def __init__(self, slang_module: Any):
        super().__init__()
        self.slang_module = slang_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.slang_module.relu(x)


class SlangpyLeakyReLU(nn.Module):
    """SlangPy-based Leaky ReLU that can be used in nn.Sequential."""

    def __init__(self, slang_module: Any):
        super().__init__()
        self.slang_module = slang_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.slang_module.leaky_relu(x)


class SlangpySquare(nn.Module):
    """SlangPy-based square operation that can be used in nn.Sequential."""

    def __init__(self, slang_module: Any):
        super().__init__()
        self.slang_module = slang_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.slang_module.square(x)


class PyTorchSquare(nn.Module):
    """PyTorch square operation for comparison."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


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
def test_relu_gradient_parity(device_type: DeviceType):
    """
    Compare gradients when ReLU is replaced with SlangPy implementation.

    Control: Linear(8, 16) -> ReLU -> Linear(16, 4)
    Test:    Linear(8, 16) -> SlangpyReLU -> Linear(16, 4)
    """
    device = helpers.get_torch_device(device_type)
    slang_module = helpers.create_module(device, SLANG_RELU)

    in_features = 8
    hidden_features = 16
    out_features = 4
    batch_size = 32

    # Create control model (pure PyTorch)
    control_model = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, out_features),
    ).cuda()

    # Create test model with SlangPy ReLU
    test_model = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        SlangpyReLU(slang_module),
        nn.Linear(hidden_features, out_features),
    ).cuda()

    # Copy weights from control to test model
    copy_model_weights(control_model, test_model)

    # Create test inputs
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, device="cuda")
    target = torch.randn(batch_size, out_features, device="cuda")

    # Run comparison
    run_gradient_parity_test(control_model, test_model, x, target)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
