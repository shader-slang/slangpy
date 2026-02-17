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
from slangpy import DeviceType
from slangpy.testing import helpers

DEVICE_TYPES = helpers.DEFAULT_DEVICE_TYPES.copy()
# Metal does not support torch integration
if DeviceType.metal in DEVICE_TYPES:
    DEVICE_TYPES.remove(DeviceType.metal)


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
    """Tests for copy_torch_tensor_to_buffer / copy_buffer_to_torch_tensor.

    These exercise the public slangpy APIs using real Buffer objects.
    Every test uses the torch_bridge_mode fixture so it runs in both
    native and fallback modes, verifying that the fallback produces
    identical results to the native path.
    """

    @pytest.fixture(autouse=True)
    def setup_bridge_mode(self, torch_bridge_mode: str):
        """Automatically use torch_bridge_mode fixture for all tests."""
        self.mode = torch_bridge_mode

    def _make_buffer(self, device: slangpy.Device, byte_size: int) -> slangpy.Buffer:
        """Create a shared buffer suitable for tensor ↔ buffer copies."""
        return device.create_buffer(
            size=byte_size,
            usage=slangpy.BufferUsage.unordered_access
            | slangpy.BufferUsage.shader_resource
            | slangpy.BufferUsage.shared,
        )

    # ------------------------------------------------------------------
    # Basic copy tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("device_type", DEVICE_TYPES)
    def test_copy_tensor_to_buffer(self, device_type: DeviceType):
        """Test copying a contiguous CUDA tensor to a buffer."""
        import numpy as np

        device = helpers.get_torch_device(device_type)
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda")
        buffer = self._make_buffer(device, tensor.numel() * tensor.element_size())

        slangpy.copy_torch_tensor_to_buffer(tensor, buffer)
        device.sync_to_cuda()

        result = np.frombuffer(buffer.to_numpy().tobytes(), dtype=np.float32)
        expected = tensor.cpu().numpy()
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("device_type", DEVICE_TYPES)
    def test_copy_buffer_to_tensor(self, device_type: DeviceType):
        """Test copying from a buffer into a contiguous CUDA tensor."""
        import numpy as np

        device = helpers.get_torch_device(device_type)
        test_values = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        buffer = self._make_buffer(device, test_values.nbytes)
        buffer.copy_from_numpy(test_values)
        device.sync_to_device()

        tensor = torch.zeros(4, dtype=torch.float32, device="cuda")
        slangpy.copy_buffer_to_torch_tensor(buffer, tensor)

        result = tensor.cpu().numpy()
        assert np.allclose(result, test_values)

    # ------------------------------------------------------------------
    # Non-contiguous tensors
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("device_type", DEVICE_TYPES)
    def test_copy_noncontiguous_tensor_to_buffer(self, device_type: DeviceType):
        """Test that non-contiguous (transposed) tensors are handled correctly."""
        import numpy as np

        device = helpers.get_torch_device(device_type)
        base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, device="cuda")
        tensor = base.T  # shape (3,2), non-contiguous
        assert not tensor.is_contiguous()

        buffer = self._make_buffer(device, tensor.numel() * tensor.element_size())
        slangpy.copy_torch_tensor_to_buffer(tensor, buffer)
        device.sync_to_cuda()

        result = np.frombuffer(buffer.to_numpy().tobytes(), dtype=np.float32)
        expected = tensor.contiguous().cpu().numpy().flatten()
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("device_type", DEVICE_TYPES)
    def test_copy_buffer_to_noncontiguous_tensor(self, device_type: DeviceType):
        """Test copying from a buffer into a non-contiguous destination tensor."""
        import numpy as np

        device = helpers.get_torch_device(device_type)
        test_values = np.arange(12, dtype=np.float32)
        buffer = self._make_buffer(device, test_values.nbytes)
        buffer.copy_from_numpy(test_values)
        device.sync_to_device()

        base = torch.zeros(4, 3, dtype=torch.float32, device="cuda")
        tensor = base.T  # shape (3,4), non-contiguous
        assert not tensor.is_contiguous()

        slangpy.copy_buffer_to_torch_tensor(buffer, tensor)

        expected = torch.from_numpy(test_values).view(tensor.shape)
        assert torch.equal(tensor, expected.to("cuda"))

    # ------------------------------------------------------------------
    # Round-trip
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("device_type", DEVICE_TYPES)
    def test_roundtrip(self, device_type: DeviceType):
        """Test data survives tensor → buffer → tensor round-trip."""
        device = helpers.get_torch_device(device_type)
        src = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5], dtype=torch.float32, device="cuda")
        buffer = self._make_buffer(device, src.numel() * src.element_size())

        slangpy.copy_torch_tensor_to_buffer(src, buffer)

        dst = torch.zeros_like(src)
        slangpy.copy_buffer_to_torch_tensor(buffer, dst)

        assert torch.allclose(src, dst)

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ],
    )
    @pytest.mark.parametrize("device_type", DEVICE_TYPES)
    def test_roundtrip_dtypes(self, device_type: DeviceType, dtype: torch.dtype):
        """Test round-trip with various dtypes."""
        device = helpers.get_torch_device(device_type)
        if dtype.is_floating_point:
            src = torch.randn(16, dtype=dtype, device="cuda")
        else:
            src = torch.randint(0, 100, (16,), dtype=dtype, device="cuda")

        buffer = self._make_buffer(device, src.numel() * src.element_size())
        slangpy.copy_torch_tensor_to_buffer(src, buffer)

        dst = torch.zeros_like(src)
        slangpy.copy_buffer_to_torch_tensor(buffer, dst)

        assert torch.equal(dst, src)

    @pytest.mark.parametrize("device_type", DEVICE_TYPES)
    def test_roundtrip_multidimensional(self, device_type: DeviceType):
        """Test round-trip with a multi-dimensional tensor."""
        device = helpers.get_torch_device(device_type)
        src = torch.randn(3, 4, 5, dtype=torch.float32, device="cuda")
        buffer = self._make_buffer(device, src.numel() * src.element_size())

        slangpy.copy_torch_tensor_to_buffer(src, buffer)

        dst = torch.zeros_like(src)
        slangpy.copy_buffer_to_torch_tensor(buffer, dst)

        assert torch.equal(dst, src)

    @pytest.mark.parametrize("device_type", DEVICE_TYPES)
    def test_roundtrip_with_storage_offset(self, device_type: DeviceType):
        """Test round-trip on a tensor slice with non-zero storage offset."""
        device = helpers.get_torch_device(device_type)
        base = torch.arange(20, dtype=torch.float32, device="cuda")
        src = base[5:15]  # storage_offset = 5
        assert src.storage_offset() != 0

        buffer = self._make_buffer(device, src.numel() * src.element_size())
        slangpy.copy_torch_tensor_to_buffer(src, buffer)

        dst = torch.zeros_like(src)
        slangpy.copy_buffer_to_torch_tensor(buffer, dst)

        assert torch.equal(dst, src)

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("device_type", DEVICE_TYPES)
    def test_copy_rejects_cpu_tensor(self, device_type: DeviceType):
        """Test that copying a CPU tensor raises an error."""
        device = helpers.get_torch_device(device_type)
        cpu_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)  # CPU
        buffer = self._make_buffer(device, cpu_tensor.numel() * cpu_tensor.element_size())

        with pytest.raises(RuntimeError, match="CUDA|cuda"):
            slangpy.copy_torch_tensor_to_buffer(cpu_tensor, buffer)

    @pytest.mark.parametrize("device_type", DEVICE_TYPES)
    def test_copy_rejects_small_buffer(self, device_type: DeviceType):
        """Test that copying to a buffer that is too small raises an error."""
        device = helpers.get_torch_device(device_type)
        tensor = torch.randn(100, dtype=torch.float32, device="cuda")
        buffer = self._make_buffer(device, 10 * 4)  # only 10 floats

        with pytest.raises(RuntimeError, match="[Bb]uffer.*small|[Ss]ize"):
            slangpy.copy_torch_tensor_to_buffer(tensor, buffer)

    # ------------------------------------------------------------------
    # Gradient handling
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("device_type", DEVICE_TYPES)
    def test_copy_buffer_to_grad_tensor(self, device_type: DeviceType):
        """Test that copy works on tensors with requires_grad=True."""
        import numpy as np

        device = helpers.get_torch_device(device_type)
        test_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buffer = self._make_buffer(device, test_values.nbytes)
        buffer.copy_from_numpy(test_values)
        device.sync_to_device()

        tensor = torch.zeros(3, dtype=torch.float32, device="cuda", requires_grad=True)
        slangpy.copy_buffer_to_torch_tensor(buffer, tensor)

        assert torch.allclose(tensor.detach(), torch.tensor(test_values, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTorchBridgeCreateEmptyTensor:
    """Tests for create_torch_empty_tensor via the TorchBridge.

    Verifies that tensors created through the bridge (both native and
    fallback modes) have the correct shape, dtype, and device.
    """

    @pytest.fixture(autouse=True)
    def setup_bridge_mode(self, torch_bridge_mode: str):
        """Automatically use torch_bridge_mode fixture for all tests."""
        self.mode = torch_bridge_mode

    # TENSOR_BRIDGE_SCALAR_* codes from tensor_bridge_api.h
    SCALAR_UINT8 = 0
    SCALAR_INT8 = 1
    SCALAR_INT16 = 2
    SCALAR_INT32 = 3
    SCALAR_INT64 = 4
    SCALAR_FLOAT16 = 5
    SCALAR_FLOAT32 = 6
    SCALAR_FLOAT64 = 7

    _DTYPE_PARAMS = [
        (SCALAR_UINT8, torch.uint8),
        (SCALAR_INT8, torch.int8),
        (SCALAR_INT16, torch.int16),
        (SCALAR_INT32, torch.int32),
        (SCALAR_INT64, torch.int64),
        (SCALAR_FLOAT16, torch.float16),
        (SCALAR_FLOAT32, torch.float32),
        (SCALAR_FLOAT64, torch.float64),
    ]

    def test_create_1d(self):
        """Test creating a simple 1D float32 tensor."""
        t = slangpy.create_torch_empty_tensor([16], self.SCALAR_FLOAT32)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (16,)
        assert t.dtype == torch.float32
        assert t.is_cuda

    def test_create_multidimensional(self):
        """Test creating a multi-dimensional tensor."""
        t = slangpy.create_torch_empty_tensor([2, 3, 4], self.SCALAR_FLOAT32)
        assert t.shape == (2, 3, 4)
        assert t.numel() == 24
        assert t.is_contiguous()
        assert t.is_cuda

    def test_create_scalar(self):
        """Test creating a 0-dimensional (scalar) tensor."""
        t = slangpy.create_torch_empty_tensor([], self.SCALAR_FLOAT32)
        assert t.shape == ()
        assert t.ndim == 0
        assert t.numel() == 1

    @pytest.mark.parametrize("scalar_type,expected_dtype", _DTYPE_PARAMS)
    def test_create_dtypes(self, scalar_type: int, expected_dtype: torch.dtype):
        """Test that all supported scalar types produce the correct torch dtype."""
        t = slangpy.create_torch_empty_tensor([8], scalar_type)
        assert t.dtype == expected_dtype
        assert t.shape == (8,)
        assert t.is_cuda

    def test_created_tensor_is_writable(self):
        """Test that the created tensor can be written to and read back."""
        t = slangpy.create_torch_empty_tensor([4], self.SCALAR_FLOAT32)
        t.fill_(42.0)
        assert torch.all(t == 42.0)

    def test_metadata_roundtrip(self):
        """Test that extract_torch_tensor_info works on a bridge-created tensor."""
        t = slangpy.create_torch_empty_tensor([3, 5], self.SCALAR_FLOAT32)
        info = slangpy.extract_torch_tensor_info(t)
        assert info["shape"] == (3, 5)
        assert info["ndim"] == 2
        assert info["numel"] == 15
        assert info["element_size"] == 4
        assert info["is_cuda"] is True
        assert info["is_contiguous"] is True
