# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys
from pathlib import Path

from slangpy import DeviceType
from slangpy.core.module import Module
from slangpy.testing import helpers

# Skip all tests in this file if running on MacOS
if sys.platform == "darwin":
    pytest.skip("TensorView requires CUDA, not available on macOS", allow_module_level=True)

# TensorView only works with CUDA device type
DEVICE_TYPES = [DeviceType.cuda] if DeviceType.cuda in helpers.DEFAULT_DEVICE_TYPES else []
if not DEVICE_TYPES:
    pytest.skip("TensorView requires CUDA device type", allow_module_level=True)


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_tensorview_copy(device_type: DeviceType):
    """Test copy_tensorview function from test_tensorview.slang."""
    device = helpers.get_torch_device(device_type)

    module = Module.load_from_file(
        device,
        str(Path(__file__).parent / "test_tensorview.slang"),
    )

    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=torch.float32)
    output_tensor = torch.zeros(5, device="cuda", dtype=torch.float32)

    module.copy_tensorview(input_tensor, output_tensor)
    torch.cuda.synchronize()

    assert torch.allclose(
        input_tensor, output_tensor
    ), f"Expected {input_tensor}, got {output_tensor}"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_tensorview_add(device_type: DeviceType):
    """Test add_tensorview function from test_tensorview.slang."""
    device = helpers.get_torch_device(device_type)

    module = Module.load_from_file(
        device,
        str(Path(__file__).parent / "test_tensorview.slang"),
    )

    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=torch.float32)
    b = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], device="cuda", dtype=torch.float32)
    output_tensor = torch.zeros(5, device="cuda", dtype=torch.float32)

    module.add_tensorview(a, b, output_tensor)
    torch.cuda.synchronize()

    expected = a + b
    assert torch.allclose(expected, output_tensor), f"Expected {expected}, got {output_tensor}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
