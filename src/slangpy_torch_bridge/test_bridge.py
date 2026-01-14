# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import slangpy_torch_bridge as bridge
import time

# Create test tensor (CPU to avoid CUDA issues with pip install)
t = torch.randn(1024, 1024)

# Test the C API works via the test function
print("=== Testing C API ===")
result = bridge.test_c_api(t)
print(f"Success: {result['success']}")
print(f"Data ptr: {result['data_ptr']}")
print(f"Shape: {result['shape']}")
print(f"Ndim: {result['ndim']}")
print(f"Is CUDA: {result['is_cuda']}")
print()

# Verify against PyTorch
print("=== Verification ===")
print(f"data_ptr matches: {result['data_ptr'] == t.data_ptr()}")
print(f"shape matches: {result['shape'] == t.shape}")
print()

# Get the function pointers
print("=== Function Pointers ===")
api_ptr = bridge.get_api_ptr()
extract_fn_ptr = bridge.get_extract_fn_ptr()
print(f"API ptr: 0x{api_ptr:016x}")
print(f"Extract fn ptr: 0x{extract_fn_ptr:016x}")
print(f"Struct size: {bridge.INFO_STRUCT_SIZE} bytes")
print()

# Benchmark
iterations = 100000

print("=== Benchmarks ===")

# Benchmark the C API test function (goes through pybind11)
start = time.perf_counter()
for _ in range(iterations):
    result = bridge.test_c_api(t)
end = time.perf_counter()
c_api_time = (end - start) * 1e6 / iterations
print(f"test_c_api() (C API via pybind11): {c_api_time:.3f} us per call")

# Benchmark extract_info (pybind11 dict creation)
start = time.perf_counter()
for _ in range(iterations):
    info = bridge.extract_info(t)
end = time.perf_counter()
extract_time = (end - start) * 1e6 / iterations
print(f"extract_info() (dict creation): {extract_time:.3f} us per call")

# Benchmark Python data_ptr() + shape + stride
start = time.perf_counter()
for _ in range(iterations):
    ptr = t.data_ptr()
    shape = t.shape
    strides = t.stride()
end = time.perf_counter()
python_time = (end - start) * 1e6 / iterations
print(f"data_ptr() + shape + stride(): {python_time:.3f} us per call")

# Benchmark __dlpack__
iterations_dlpack = 10000
start = time.perf_counter()
for _ in range(iterations_dlpack):
    capsule = t.__dlpack__()
end = time.perf_counter()
dlpack_time = (end - start) * 1e6 / iterations_dlpack
print(f"__dlpack__(): {dlpack_time:.3f} us per call")
