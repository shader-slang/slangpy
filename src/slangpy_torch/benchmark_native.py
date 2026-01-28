# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# pyright: reportAttributeAccessIssue=false

"""
Benchmark the ACTUAL C function call overhead from native code.

This uses the benchmark functions built into the bridge module,
which measure the C function call in a tight loop without any
Python involvement.
"""

import torch
import slangpy_torch as bridge

# Create test tensor
device = "cuda" if torch.cuda.is_available() else "cpu"
t = torch.randn(1024, 1024, device=device)

print("=" * 70)
print("TENSOR BRIDGE - NATIVE C FUNCTION CALL BENCHMARK")
print("=" * 70)
print(f"Device: {device}")
print(f"Tensor shape: {t.shape}")
print(f"TensorBridgeInfo struct size: {bridge.INFO_STRUCT_SIZE} bytes")
print()

iterations = 1000000

# Benchmark 1: The C API function call (what slangpy_ext would use)
print("Running C API benchmark...")
result = bridge.benchmark_c_api(t, iterations)
c_api_ns = result["per_call_ns"]
print(f"  tensor_bridge_extract() via PyObject*: {c_api_ns:.1f} ns/call")
print(
    f"  Verified: data_ptr=0x{result['data_ptr']:x}, ndim={result['ndim']}, numel={result['numel']}"
)
print()

# Benchmark 2: Direct libtorch access (theoretical minimum)
print("Running libtorch baseline benchmark...")
result = bridge.benchmark_libtorch(t, iterations)
libtorch_ns = result["per_call_ns"]
print(f"  Direct libtorch access (data_ptr+sizes+strides+numel): {libtorch_ns:.1f} ns/call")
print()

# Benchmark 3: Python API for comparison
import time

iterations_py = 100000

print("Running Python API benchmark...")
start = time.perf_counter()
for _ in range(iterations_py):
    ptr = t.data_ptr()
    shape = t.shape
    strides = t.stride()
    numel = t.numel()
end = time.perf_counter()
python_ns = (end - start) * 1e9 / iterations_py
print(f"  Python API (data_ptr+shape+stride+numel): {python_ns:.1f} ns/call")
print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Direct libtorch (baseline):     {libtorch_ns:7.1f} ns")
print(
    f"C API via PyObject*:            {c_api_ns:7.1f} ns  (+{c_api_ns - libtorch_ns:.1f} ns overhead)"
)
print(
    f"Python API:                     {python_ns:7.1f} ns  ({python_ns/c_api_ns:.1f}x slower than C API)"
)
print()
print("The C API overhead over direct libtorch is the cost of:")
print("  1. THPVariable_Check() - type check")
print("  2. THPVariable_Unpack() - get C++ tensor from PyObject")
print("  3. Copying shape/strides to the output struct")
print()
print("This is what your slangpy_ext code will experience when calling")
print("the function pointer with an nb::handle.ptr().")
