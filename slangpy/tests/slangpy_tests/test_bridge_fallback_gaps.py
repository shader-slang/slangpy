# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Tests targeting coverage gaps in torchintegration/bridge_fallback.py.

Exercises error/guard paths:
- extract_tensor_info with non-tensor
- get_signature with non-tensor
- copy_to_buffer / copy_from_buffer with non-CUDA and undersized buffers
- _CudaBufferView with unsupported dtype
- create_empty_tensor with bad scalar type code
- create_zeros_like with non-tensor
- _get_cuda_stream with non-CUDA tensor
"""

import sys
import pytest

if sys.platform == "darwin":
    pytest.skip("Torch tests require CUDA", allow_module_level=True)

try:
    import torch
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)

from slangpy.torchintegration import bridge_fallback as bf


class TestExtractTensorInfo:
    def test_non_tensor_raises(self):
        with pytest.raises(ValueError, match="not a PyTorch tensor"):
            bf.extract_tensor_info("not a tensor")

    def test_cpu_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        info = bf.extract_tensor_info(t)
        assert info["ndim"] == 1
        assert info["shape"] == (3,)
        assert info["scalar_type"] == 6  # float32
        assert info["is_cuda"] is False
        assert info["device_index"] == -1
        assert info["numel"] == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensor(self):
        t = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)
        info = bf.extract_tensor_info(t)
        assert info["is_cuda"] is True
        assert info["device_index"] is not None
        assert info["device_type"] == 1


class TestGetSignature:
    def test_non_tensor_returns_none(self):
        result = bf.get_signature("not a tensor")
        assert result is None

    def test_float32_1d(self):
        t = torch.tensor([1.0], dtype=torch.float32)
        sig = bf.get_signature(t)
        assert sig == "[D1,S6]"

    def test_int32_2d(self):
        t = torch.zeros((3, 4), dtype=torch.int32)
        sig = bf.get_signature(t)
        assert sig == "[D2,S3]"


class TestCudaBufferView:
    def test_unsupported_dtype_raises(self):
        with pytest.raises(RuntimeError, match="Unsupported dtype"):
            bf._CudaBufferView(0x1000, (10,), torch.complex128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_valid_dtype(self):
        t = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        view = bf._CudaBufferView(t.data_ptr(), (1,), torch.float32)
        iface = view.__cuda_array_interface__
        assert iface["shape"] == (1,)
        assert iface["typestr"] == "<f4"
        assert iface["version"] == 3


class TestCopyToBuffer:
    def test_non_cuda_raises(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        with pytest.raises(RuntimeError, match="CUDA"):
            bf.copy_to_buffer(t, 0x1000, 1024)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_undersized_buffer_raises(self):
        t = torch.ones(100, device="cuda", dtype=torch.float32)
        dest = torch.zeros(1, device="cuda", dtype=torch.float32)
        with pytest.raises(RuntimeError, match="too small"):
            bf.copy_to_buffer(t, dest.data_ptr(), 4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_copy_round_trip(self):
        src = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
        dest = torch.zeros(3, device="cuda", dtype=torch.float32)
        bf.copy_to_buffer(src, dest.data_ptr(), dest.numel() * dest.element_size())
        assert torch.allclose(src, dest)


class TestCopyFromBuffer:
    def test_non_cuda_raises(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        with pytest.raises(RuntimeError, match="CUDA"):
            bf.copy_from_buffer(t, 0x1000, 1024)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_undersized_buffer_raises(self):
        t = torch.ones(100, device="cuda", dtype=torch.float32)
        src = torch.zeros(1, device="cuda", dtype=torch.float32)
        with pytest.raises(RuntimeError, match="too small"):
            bf.copy_from_buffer(t, src.data_ptr(), 4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_copy_round_trip(self):
        src = torch.tensor([4.0, 5.0, 6.0], device="cuda", dtype=torch.float32)
        dest = torch.zeros(3, device="cuda", dtype=torch.float32)
        bf.copy_from_buffer(dest, src.data_ptr(), src.numel() * src.element_size())
        assert torch.allclose(src, dest)


class TestCreateEmptyTensor:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_valid_float32(self):
        t = bf.create_empty_tensor([2, 3], scalar_type=6)
        assert t.shape == (2, 3)
        assert t.dtype == torch.float32
        assert t.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_unsupported_scalar_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported scalar type"):
            bf.create_empty_tensor([4], scalar_type=999)


class TestCreateZerosLike:
    def test_non_tensor_raises(self):
        with pytest.raises(ValueError, match="not a PyTorch tensor"):
            bf.create_zeros_like("not a tensor")

    def test_valid(self):
        t = torch.ones(5, dtype=torch.float32)
        z = bf.create_zeros_like(t)
        assert z.shape == (5,)
        assert torch.all(z == 0.0)


class TestGetCudaStream:
    def test_non_cuda_returns_zero(self):
        t = torch.tensor([1.0], dtype=torch.float32)
        assert bf._get_cuda_stream(t) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_returns_nonzero(self):
        t = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        stream_ptr = bf._get_cuda_stream(t)
        assert isinstance(stream_ptr, int)


class TestIsTensor:
    def test_true_for_tensor(self):
        assert bf.is_tensor(torch.tensor([1.0])) is True

    def test_false_for_non_tensor(self):
        assert bf.is_tensor("not a tensor") is False
        assert bf.is_tensor(42) is False
