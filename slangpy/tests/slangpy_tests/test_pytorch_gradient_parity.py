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

from slangpy import DeviceType
from slangpy.testing import helpers

try:
    import torch
    import torch.nn as nn
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)

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

    def __init__(self, slang_module):
        super().__init__()
        self.slang_module = slang_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.slang_module.relu(x)


class SlangpyLeakyReLU(nn.Module):
    """SlangPy-based Leaky ReLU that can be used in nn.Sequential."""

    def __init__(self, slang_module):
        super().__init__()
        self.slang_module = slang_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.slang_module.leaky_relu(x)


class SlangpySquare(nn.Module):
    """SlangPy-based square operation that can be used in nn.Sequential."""

    def __init__(self, slang_module):
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


def run_training_step(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Run one training step and return gradients for all parameters.

    Returns:
        - Dict mapping parameter names to their gradients
        - Input gradient (if input requires_grad)
    """
    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()

    # Forward pass
    output = model(x)

    # Simple MSE loss
    loss = ((output - target) ** 2).mean()

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


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_relu_gradient_parity(device_type: DeviceType):
    """
    Compare gradients when ReLU is replaced with SlangPy implementation.

    Control: Linear(8, 16) -> ReLU -> Linear(16, 4)
    Test:    Linear(8, 16) -> SlangpyReLU -> Linear(16, 4)

    We verify that:
    1. Input gradients match
    2. All layer weight/bias gradients match
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

    # Create identical inputs for both models
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, device="cuda")
    target = torch.randn(batch_size, out_features, device="cuda")

    # Need separate input tensors for separate backward passes
    x_control = x.clone().detach().requires_grad_(True)
    x_test = x.clone().detach().requires_grad_(True)

    # Run training step on both models
    control_grads, control_input_grad = run_training_step(control_model, x_control, target)
    test_grads, test_input_grad = run_training_step(test_model, x_test, target)

    # Compare input gradients
    assert control_input_grad is not None and test_input_grad is not None
    compare_gradients("Input gradient", test_input_grad, control_input_grad)

    # Compare all parameter gradients
    for name in control_grads:
        compare_gradients(f"Parameter '{name}'", test_grads[name], control_grads[name])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
