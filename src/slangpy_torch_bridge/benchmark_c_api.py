# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Benchmark demonstrating the actual C function pointer overhead.

This simulates what happens when your native code calls the bridge
function pointer directly with a PyObject*.
"""

import torch
import slangpy_torch_bridge as bridge
import time
import ctypes

# Create test tensor
t = torch.randn(1024, 1024)

print("=" * 60)
print("TENSOR BRIDGE C API BENCHMARK")
print("=" * 60)
print()

# Get the function pointer
extract_fn_ptr = bridge.get_extract_fn_ptr()
print(f"Extract function pointer: 0x{extract_fn_ptr:016x}")
print(f"TensorBridgeInfo struct size: {bridge.INFO_STRUCT_SIZE} bytes")
print()

# Set up ctypes to call the C function directly
# This simulates what your native code does with the function pointer


# Define the struct in ctypes
class TensorBridgeInfo(ctypes.Structure):
    _fields_ = [
        ("data_ptr", ctypes.c_void_p),
        ("shape", ctypes.c_int64 * 12),
        ("strides", ctypes.c_int64 * 12),
        ("ndim", ctypes.c_int32),
        ("device_type", ctypes.c_int32),
        ("device_index", ctypes.c_int32),
        ("scalar_type", ctypes.c_int32),
        ("element_size", ctypes.c_int32),
        ("numel", ctypes.c_int64),
        ("storage_offset", ctypes.c_int64),
        ("flags", ctypes.c_uint32),  # bitfield
    ]


# Define the function signature
# int tensor_bridge_extract(void* py_obj, TensorBridgeInfo* out)
ExtractFn = ctypes.CFUNCTYPE(
    ctypes.c_int,  # return type
    ctypes.c_void_p,  # PyObject* (as void*)
    ctypes.POINTER(TensorBridgeInfo),  # TensorBridgeInfo*
)

# Cast the function pointer
extract_fn = ExtractFn(extract_fn_ptr)

# Allocate the output struct
info = TensorBridgeInfo()

# Get PyObject* as an integer (this is what id() returns in CPython)
tensor_ptr = id(t)

print("=== Direct C Function Call Test ===")
# Call the function directly
result = extract_fn(tensor_ptr, ctypes.byref(info))
print(f"Return code: {result}")
print(f"Data pointer: 0x{info.data_ptr:016x}")
print(f"Expected:     0x{t.data_ptr():016x}")
print(f"Match: {info.data_ptr == t.data_ptr()}")
print(f"ndim: {info.ndim}")
print(f"numel: {info.numel}")
print(f"shape: {list(info.shape[:info.ndim])}")
print(f"strides: {list(info.strides[:info.ndim])}")
print()

# Benchmark the direct C function call via ctypes
print("=== Benchmarks ===")
print()

# Warm up
for _ in range(1000):
    extract_fn(tensor_ptr, ctypes.byref(info))

iterations = 1000000

# Benchmark direct C call via ctypes
start = time.perf_counter()
for _ in range(iterations):
    extract_fn(tensor_ptr, ctypes.byref(info))
end = time.perf_counter()
c_call_time_ns = (end - start) * 1e9 / iterations
print(f"Direct C function call (via ctypes):   {c_call_time_ns:.1f} ns")
print("  ^ This is what your native code would experience")
print()

# Compare with Python API calls
iterations_py = 100000

start = time.perf_counter()
for _ in range(iterations_py):
    ptr = t.data_ptr()
    shape = t.shape
    strides = t.stride()
end = time.perf_counter()
python_time_ns = (end - start) * 1e9 / iterations_py
print(f"Python API (data_ptr + shape + stride): {python_time_ns:.1f} ns")

start = time.perf_counter()
for _ in range(iterations_py):
    cai = t.__cuda_array_interface__ if t.is_cuda else None
end = time.perf_counter()
# Skip if not CUDA

start = time.perf_counter()
for _ in range(iterations_py):
    result = bridge.test_c_api(t)
end = time.perf_counter()
pybind_time_ns = (end - start) * 1e9 / iterations_py
print(f"bridge.test_c_api() (pybind11 wrapper): {pybind_time_ns:.1f} ns")

print()
print("=== Summary ===")
print(f"Direct C call: {c_call_time_ns:.1f} ns")
print(f"Python API:    {python_time_ns:.1f} ns  ({python_time_ns/c_call_time_ns:.1f}x slower)")
print(f"Pybind11:      {pybind_time_ns:.1f} ns  ({pybind_time_ns/c_call_time_ns:.1f}x slower)")
print()
print("NOTE: The ctypes overhead adds ~50-100ns. Your actual C++ code")
print("calling the function pointer directly will be even faster.")
