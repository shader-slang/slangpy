# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for the PyTorch tensor bridge (slangpy_torch).

This tests the fast native C API for extracting PyTorch tensor metadata.
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
        """Test that is_torch_bridge_available returns True when slangpy_torch is installed."""
        # If we got past the imports, torch is available so slangpy_torch should be loadable
        # (assuming it was installed via pip install src/slangpy_torch)
        result = slangpy.is_torch_bridge_available()
        # Just check that it returns a boolean
        assert isinstance(result, bool)

    def test_is_torch_tensor_with_tensor(self):
        """Test is_torch_tensor correctly identifies PyTorch tensors."""
        if not slangpy.is_torch_bridge_available():
            pytest.skip("slangpy_torch not installed")

        t = torch.zeros(4, 3, 2)
        assert slangpy.is_torch_tensor(t) is True

    def test_is_torch_tensor_with_non_tensor(self):
        """Test is_torch_tensor correctly rejects non-tensors."""
        if not slangpy.is_torch_bridge_available():
            pytest.skip("slangpy_torch not installed")

        assert slangpy.is_torch_tensor([1, 2, 3]) is False
        assert slangpy.is_torch_tensor("hello") is False
        assert slangpy.is_torch_tensor(42) is False
        # Note: None is not accepted by the function (nanobind rejects it)


class TestTorchTensorExtraction:
    """Test tensor metadata extraction."""

    @pytest.fixture(autouse=True)
    def check_bridge_available(self):
        if not slangpy.is_torch_bridge_available():
            pytest.skip("slangpy_torch not installed")

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
        assert info["cuda_stream"] == 0  # No stream for CPU

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

            # cuda_stream should be non-zero for a non-default stream
            assert info["cuda_stream"] != 0
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
