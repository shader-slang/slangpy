# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for the PyTorch tensor bridge.

This tests both the native C API (via slangpy_torch) and the Python fallback
for extracting PyTorch tensor metadata. Tests are run in both modes using
the torch_bridge_mode fixture.
"""

import pytest
import sys

try:
    import torch
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)

if sys.platform == "darwin":
    pytest.skip("PyTorch requires CUDA, that is not available on macOS", allow_module_level=True)

import slangpy


class TestTorchBridgeAvailability:
    """Test bridge availability detection."""

    def test_is_torch_bridge_available(self):
        """Test that is_torch_bridge_available returns True (either native or fallback)."""
        # With the fallback implementation, the bridge should always be available
        # when torch is installed
        result = slangpy.is_torch_bridge_available()
        assert isinstance(result, bool)
        assert result is True  # Should be True since torch is installed

    def test_fallback_toggle(self):
        """Test that we can toggle between native and fallback modes."""
        original = slangpy.is_torch_bridge_using_fallback()

        try:
            # Force fallback mode
            slangpy.set_torch_bridge_python_fallback(True)
            assert slangpy.is_torch_bridge_using_fallback() is True

            # Disable forced fallback (may still be fallback if native unavailable)
            slangpy.set_torch_bridge_python_fallback(False)
            # Result depends on whether slangpy_torch is installed
        finally:
            # Restore original state
            slangpy.set_torch_bridge_python_fallback(original)

    def test_is_torch_tensor_with_tensor(self, torch_bridge_mode: str):
        """Test is_torch_tensor correctly identifies PyTorch tensors."""
        t = torch.zeros(4, 3, 2)
        assert slangpy.is_torch_tensor(t) is True

    def test_is_torch_tensor_with_non_tensor(self, torch_bridge_mode: str):
        """Test is_torch_tensor correctly rejects non-tensors."""
        assert slangpy.is_torch_tensor([1, 2, 3]) is False
        assert slangpy.is_torch_tensor("hello") is False
        assert slangpy.is_torch_tensor(42) is False
        # Note: None is not accepted by the function (nanobind rejects it)


class TestTorchTensorExtraction:
    """Test tensor metadata extraction in both native and fallback modes."""

    @pytest.fixture(autouse=True)
    def setup_bridge_mode(self, torch_bridge_mode: str):
        """Automatically use torch_bridge_mode fixture for all tests in this class."""
        self.mode = torch_bridge_mode

    def test_extract_cpu_tensor(self):
        """Test extraction of CPU tensor metadata."""
        t = torch.zeros(4, 3, 2, dtype=torch.float32)
        info = slangpy.extract_torch_tensor_info(t)

        assert info["shape"] == (4, 3, 2)
        assert info["strides"] == (6, 2, 1)
        assert info["ndim"] == 3
        assert info["device_type"] == 0  # CPU
        assert info["device_index"] == -1  # CPU has no index
        assert info["element_size"] == 4
        assert info["numel"] == 24
        assert info["is_contiguous"] is True
        assert info["is_cuda"] is False
        assert info["requires_grad"] is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_extract_cuda_tensor(self):
        """Test extraction of CUDA tensor metadata."""
        t = torch.zeros(8, 4, device="cuda:0", dtype=torch.float32)
        info = slangpy.extract_torch_tensor_info(t)

        assert info["shape"] == (8, 4)
        assert info["strides"] == (4, 1)
        assert info["ndim"] == 2
        assert info["device_type"] == 1  # CUDA
        assert info["device_index"] == 0
        assert info["element_size"] == 4
        assert info["numel"] == 32
        assert info["is_contiguous"] is True
        assert info["is_cuda"] is True
        assert info["data_ptr"] != 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_extract_cuda_stream(self):
        """Test extraction of CUDA stream from tensor on non-default stream."""
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        try:
            t = torch.zeros(4, 4, device="cuda:0")
            info = slangpy.extract_torch_tensor_info(t)
            assert info["is_cuda"] is True
        finally:
            # Reset to default stream
            torch.cuda.set_stream(torch.cuda.default_stream())

    def test_extract_different_dtypes(self):
        """Test extraction of tensors with different data types."""
        dtypes_and_sizes = [
            (torch.float16, 2),
            (torch.float32, 4),
            (torch.float64, 8),
            (torch.int8, 1),
            (torch.int16, 2),
            (torch.int32, 4),
            (torch.int64, 8),
            (torch.uint8, 1),
            (torch.bool, 1),
        ]

        for dtype, expected_size in dtypes_and_sizes:
            t = torch.zeros(10, dtype=dtype)
            info = slangpy.extract_torch_tensor_info(t)
            assert info["element_size"] == expected_size, f"Failed for {dtype}"
            assert info["numel"] == 10

    def test_extract_non_contiguous_tensor(self):
        """Test extraction of non-contiguous tensor (transposed)."""
        t = torch.zeros(4, 3)
        t_transposed = t.T  # Transpose makes it non-contiguous

        info = slangpy.extract_torch_tensor_info(t_transposed)

        assert info["shape"] == (3, 4)
        assert info["strides"] == (1, 3)  # Non-contiguous strides
        assert info["is_contiguous"] is False

    def test_extract_tensor_with_storage_offset(self):
        """Test extraction of tensor with non-zero storage offset."""
        t = torch.zeros(10, dtype=torch.float32)
        t_slice = t[2:8]  # Slice creates storage offset

        info = slangpy.extract_torch_tensor_info(t_slice)

        assert info["shape"] == (6,)
        assert info["numel"] == 6
        assert info["storage_offset"] == 2

    def test_extract_tensor_with_grad(self):
        """Test extraction of tensor requiring gradients."""
        t = torch.zeros(4, 4, requires_grad=True)
        info = slangpy.extract_torch_tensor_info(t)

        assert info["requires_grad"] is True

    def test_extract_0d_tensor(self):
        """Test extraction of 0-dimensional (scalar) tensor."""
        t = torch.tensor(42.0)
        info = slangpy.extract_torch_tensor_info(t)

        assert info["shape"] == ()
        assert info["strides"] == ()
        assert info["ndim"] == 0
        assert info["numel"] == 1

    def test_extract_1d_tensor(self):
        """Test extraction of 1-dimensional tensor."""
        t = torch.zeros(100)
        info = slangpy.extract_torch_tensor_info(t)

        assert info["shape"] == (100,)
        assert info["strides"] == (1,)
        assert info["ndim"] == 1

    def test_extract_high_dimensional_tensor(self):
        """Test extraction of high-dimensional tensor."""
        t = torch.zeros(2, 3, 4, 5, 6)
        info = slangpy.extract_torch_tensor_info(t)

        assert info["shape"] == (2, 3, 4, 5, 6)
        assert info["ndim"] == 5
        assert info["numel"] == 2 * 3 * 4 * 5 * 6

    def test_extract_non_tensor_raises(self):
        """Test that extracting non-tensor raises ValueError."""
        with pytest.raises(ValueError, match="not a PyTorch tensor"):
            slangpy.extract_torch_tensor_info([1, 2, 3])

    def test_data_ptr_is_valid(self):
        """Test that data_ptr points to valid memory."""
        t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        info = slangpy.extract_torch_tensor_info(t)

        # data_ptr should be non-zero for a tensor with data
        assert info["data_ptr"] != 0

        # Create a view from data_ptr using numpy and verify values
        import ctypes
        import numpy as np

        ptr = ctypes.cast(info["data_ptr"], ctypes.POINTER(ctypes.c_float))
        arr = np.ctypeslib.as_array(ptr, shape=(4,))

        assert arr[0] == 1.0
        assert arr[1] == 2.0
        assert arr[2] == 3.0
        assert arr[3] == 4.0

    def test_extract_tensor_signature(self):
        """Test extraction of tensor signature."""
        t = torch.zeros(4, 4, dtype=torch.float32)
        signature = slangpy.extract_torch_tensor_signature(t)
        assert signature == "[D2,S6]"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTorchBridgeCopy:
    """Test copy_to_buffer and copy_from_buffer in both native and fallback modes."""

    @pytest.fixture(autouse=True)
    def setup_bridge_mode(self, torch_bridge_mode: str):
        """Automatically use torch_bridge_mode fixture for all tests in this class."""
        self.mode = torch_bridge_mode

    def test_copy_to_buffer_contiguous(self):
        """Test copying a contiguous CUDA tensor to a raw CUDA buffer."""
        from slangpy.torchintegration.bridge_fallback import copy_to_buffer

        src = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda")
        buf = torch.zeros(4, dtype=torch.float32, device="cuda")
        byte_size = src.numel() * src.element_size()

        copy_to_buffer(src, buf.data_ptr(), byte_size)
        assert torch.equal(buf, src)

    def test_copy_to_buffer_non_contiguous(self):
        """Test copying a non-contiguous (transposed) CUDA tensor to a buffer."""
        from slangpy.torchintegration.bridge_fallback import copy_to_buffer

        base = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
        src = base.T  # shape (4,3), non-contiguous
        assert not src.is_contiguous()

        buf = torch.zeros(src.numel(), dtype=torch.float32, device="cuda")
        byte_size = src.numel() * src.element_size()

        copy_to_buffer(src, buf.data_ptr(), byte_size)
        # The buffer should contain data in C-contiguous order of the transposed view.
        expected = src.contiguous().view(-1)
        assert torch.equal(buf, expected)

    def test_copy_from_buffer_contiguous(self):
        """Test copying from a raw CUDA buffer into a contiguous tensor."""
        from slangpy.torchintegration.bridge_fallback import copy_from_buffer

        buf = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32, device="cuda")
        dest = torch.zeros(4, dtype=torch.float32, device="cuda")
        byte_size = buf.numel() * buf.element_size()

        copy_from_buffer(dest, buf.data_ptr(), byte_size)
        assert torch.equal(dest, buf)

    def test_copy_from_buffer_non_contiguous(self):
        """Test copying from a buffer into a non-contiguous destination tensor."""
        from slangpy.torchintegration.bridge_fallback import copy_from_buffer

        buf = torch.arange(12, dtype=torch.float32, device="cuda")
        base = torch.zeros(4, 3, dtype=torch.float32, device="cuda")
        dest = base.T  # shape (3,4), non-contiguous
        assert not dest.is_contiguous()

        byte_size = dest.numel() * dest.element_size()
        copy_from_buffer(dest, buf.data_ptr(), byte_size)

        expected = buf.view(dest.shape)
        assert torch.equal(dest, expected)

    def test_copy_roundtrip(self):
        """Test data survives a copy_to_buffer → copy_from_buffer round-trip."""
        from slangpy.torchintegration.bridge_fallback import (
            copy_from_buffer,
            copy_to_buffer,
        )

        src = torch.randn(8, 5, dtype=torch.float32, device="cuda")
        buf = torch.zeros(src.numel(), dtype=torch.float32, device="cuda")
        byte_size = src.numel() * src.element_size()

        copy_to_buffer(src, buf.data_ptr(), byte_size)

        dest = torch.zeros_like(src)
        copy_from_buffer(dest, buf.data_ptr(), byte_size)

        assert torch.equal(dest, src)

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.int8,
            torch.int32,
            torch.int64,
            torch.uint8,
        ],
    )
    def test_copy_roundtrip_dtypes(self, dtype: torch.dtype):
        """Test round-trip copy with various dtypes."""
        from slangpy.torchintegration.bridge_fallback import (
            copy_from_buffer,
            copy_to_buffer,
        )

        if dtype.is_floating_point:
            src = torch.randn(16, dtype=dtype, device="cuda")
        else:
            src = torch.randint(0, 100, (16,), dtype=dtype, device="cuda")

        buf = torch.zeros(src.numel(), dtype=dtype, device="cuda")
        byte_size = src.numel() * src.element_size()

        copy_to_buffer(src, buf.data_ptr(), byte_size)
        dest = torch.zeros_like(src)
        copy_from_buffer(dest, buf.data_ptr(), byte_size)

        assert torch.equal(dest, src)

    def test_copy_multidimensional(self):
        """Test round-trip copy with a multi-dimensional tensor."""
        from slangpy.torchintegration.bridge_fallback import (
            copy_from_buffer,
            copy_to_buffer,
        )

        src = torch.randn(3, 4, 5, dtype=torch.float32, device="cuda")
        buf = torch.zeros(src.numel(), dtype=torch.float32, device="cuda")
        byte_size = src.numel() * src.element_size()

        copy_to_buffer(src, buf.data_ptr(), byte_size)
        dest = torch.zeros_like(src)
        copy_from_buffer(dest, buf.data_ptr(), byte_size)

        assert torch.equal(dest, src)

    def test_copy_to_buffer_rejects_cpu_tensor(self):
        """Test that copy_to_buffer raises for CPU tensors."""
        from slangpy.torchintegration.bridge_fallback import copy_to_buffer

        src = torch.tensor([1.0, 2.0], dtype=torch.float32)  # CPU
        buf = torch.zeros(2, dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError, match="CUDA"):
            copy_to_buffer(src, buf.data_ptr(), 8)

    def test_copy_from_buffer_rejects_cpu_tensor(self):
        """Test that copy_from_buffer raises for CPU tensors."""
        from slangpy.torchintegration.bridge_fallback import copy_from_buffer

        buf = torch.zeros(2, dtype=torch.float32, device="cuda")
        dest = torch.zeros(2, dtype=torch.float32)  # CPU
        with pytest.raises(RuntimeError, match="CUDA"):
            copy_from_buffer(dest, buf.data_ptr(), 8)

    def test_copy_to_buffer_rejects_small_dest(self):
        """Test that copy_to_buffer raises when destination is too small."""
        from slangpy.torchintegration.bridge_fallback import copy_to_buffer

        src = torch.randn(100, dtype=torch.float32, device="cuda")
        buf = torch.zeros(10, dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError, match="too small"):
            copy_to_buffer(src, buf.data_ptr(), buf.numel() * buf.element_size())

    def test_copy_from_buffer_rejects_small_src(self):
        """Test that copy_from_buffer raises when source is too small."""
        from slangpy.torchintegration.bridge_fallback import copy_from_buffer

        buf = torch.zeros(10, dtype=torch.float32, device="cuda")
        dest = torch.zeros(100, dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError, match="too small"):
            copy_from_buffer(dest, buf.data_ptr(), buf.numel() * buf.element_size())

    def test_copy_from_buffer_no_grad(self):
        """Test that copy_from_buffer works on tensors with requires_grad=True."""
        from slangpy.torchintegration.bridge_fallback import copy_from_buffer

        buf = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda")
        dest = torch.zeros(3, dtype=torch.float32, device="cuda", requires_grad=True)
        byte_size = buf.numel() * buf.element_size()

        # Should not raise despite requires_grad=True
        copy_from_buffer(dest, buf.data_ptr(), byte_size)
        assert torch.equal(dest.detach(), buf)
