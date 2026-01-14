# slangpy_torch_bridge

Fast PyTorch tensor access bridge for slangpy.

This is a minimal PyTorch native extension that provides zero-overhead access
to PyTorch tensor internals (data pointer, shape, strides) without going through
Python APIs like `__dlpack__` or `__cuda_array_interface__`.

## Why?

When accessing PyTorch tensor data from a library that doesn't link to libtorch:

| Method | Overhead per call |
|--------|------------------|
| `tensor.data_ptr()` | ~0.09 µs |
| `tensor.__cuda_array_interface__` | ~1.5 µs |
| `tensor.__dlpack__()` | ~1.2 µs |
| **This bridge** | ~0.01 µs |

For hot paths where microseconds matter, this 100x improvement is significant.

## Installation

```bash
pip install ./src/slangpy_torch_bridge
```

## Usage

```python
import torch
import slangpy_torch_bridge as bridge

tensor = torch.randn(1024, 1024, device='cuda')

# Fast extraction
info = bridge.extract_info(tensor)
print(f"Data pointer: {info.data_ptr}")
print(f"Shape: {info.shape_tuple}")
print(f"Strides: {info.strides_tuple}")
```

## Integration with slangpy

The main slangpy library can import this bridge when PyTorch tensors are detected,
allowing it to access tensor data without the overhead of Python protocols.
