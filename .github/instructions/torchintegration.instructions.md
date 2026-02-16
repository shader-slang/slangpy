---
description: These instructions should be loaded whenever the context refers to torch
---

# PyTorch Integration

SlangPy allows users to pass `torch.Tensor` objects directly to Slang GPU functions. Torch tensors are detected automatically — no wrappers required. The system supports forward dispatch, autograd backward differentiation, and CUDA stream synchronization.

## Architecture Overview

The torch integration spans four layers:

```
User code: module.add(torch_tensor_a, torch_tensor_b)
    │
    ▼
Phase 1 (C++): Signature building — TorchBridge extracts tensor metadata (~28ns native)
    │
    ▼
Phase 2 (Python, once): TorchTensorMarshall created, kernel generated, CallData cached
    │
    ▼
Phase 3 (C++): Dispatch — tensor data bound to shader, CUDA stream sync, kernel launched
    │
    ▼
(if requires_grad) Autograd: TorchAutoGradHook.backward() → function.bwds() kernel
```

## Two Operating Modes

The system has a TorchBridge singleton (`src/slangpy_ext/utils/torch_bridge.h`) that provides tensor metadata extraction in two modes:

1. **Native mode** (fast, ~28ns): Uses the separate `slangpy_torch` extension package, which links against libtorch's C++ API. Installed via `pip install slangpy-torch --no-build-isolation`.
2. **Python fallback mode** (~350ns): Uses `slangpy/torchintegration/bridge_fallback.py` with standard PyTorch Python APIs. Always available when torch is installed.

The fallback cannot perform CUDA memory copies (`copy_to_buffer`/`copy_from_buffer` raise `RuntimeError`). These are only needed for non-CUDA backends (D3D12/Vulkan) using interop buffers.

For testing, `TorchBridge::set_force_python_fallback(true)` forces fallback mode even when native is available.

## Native extension development

When developing the native extension, install it as editable:
```
cd .\src\slangpy_torch\
pip install --editable . --no-build-isolation
```

## File Map

### Bridge Layer (tensor metadata extraction)
| File | Purpose |
|------|---------|
| `src/slangpy_torch/tensor_bridge_api.h` | Shared C API header — `TensorBridgeInfo` struct, `TensorBridgeAPI` function pointer table, result codes. Used by both `slangpy_torch` and `slangpy_ext`. |
| `src/slangpy_torch/torch_bridge_impl.cpp` | Native implementation using libtorch C++ API (`THPVariable_Unpack`). Compiled as a separate pybind11 module. |
| `src/slangpy_torch/setup.py` | Build script for the `slangpy-torch` pip package. Must use `--no-build-isolation` for ABI compatibility. |
| `src/slangpy_ext/utils/torch_bridge.h` | `TorchBridge` singleton — lazy-initializes native or fallback mode, provides `extract()`, `is_tensor()`, `get_signature()`, `copy_to_buffer()`, `copy_from_buffer()`, `call_torch_autograd_hook()`. |
| `slangpy/torchintegration/bridge_fallback.py` | Python fallback implementations: `extract_tensor_info()`, `get_signature()`, `get_current_cuda_stream()`. Memory copy functions raise `RuntimeError`. |

### Marshalling Layer (type resolution + shader binding)
| File | Purpose |
|------|---------|
| `slangpy/torchintegration/torchtensormarshall.py` | `TorchTensorMarshall` — Python-side marshall for `torch.Tensor`. Handles type resolution (`resolve_types`), dimensionality (`resolve_dimensionality`), and kernel code generation (`gen_calldata`). Registered in `PYTHON_TYPES[torch.Tensor]`. |
| `src/slangpy_ext/utils/slangpytorchtensor.h` | `NativeTorchTensorMarshall` (C++) — base class providing native `get_shape()` and `write_shader_cursor_pre_dispatch()`. Also defines `NativeTorchTensorDiffPair` for autograd. |
| `src/slangpy_ext/utils/slangpytorchtensor.cpp` | Native implementation — tensor field writing, interop buffer management, output tensor creation, CUDA pointer binding. |

### Autograd Layer (backward differentiation)
| File | Purpose |
|------|---------|
| `slangpy/torchintegration/autogradhook.py` | `TorchAutoGradHook(torch.autograd.Function)` — wraps SlangPy dispatch in PyTorch's autograd graph. Forward runs kernel; backward lazily generates a `bwds` CallData with real gradient tensors. |
| `slangpy/torchintegration/detection.py` | `detect_torch_tensors()` — scans args/kwargs for `torch.Tensor` and `NativeTorchTensorDiffPair` objects, returns `(has_torch, requires_autograd)`. |
| `slangpy/core/calldata.py` | `torch_autograd_hook()` (module-level function) — entry point called from C++ when autograd is active. Wraps tensors in `NativeTorchTensorDiffPair`, invokes `TorchAutoGradHook.apply()`. Also: `find_torch_tensors()` walks args/kwargs and assigns `is_input` based on binding access patterns. |
| `slangpy/core/function.py` | `FunctionNode.bwds` property — returns a `FunctionNodeBwds` that generates backward-derivative kernels (`CallMode.bwds`). |

### Dispatch Integration (C++)
| File | Purpose |
|------|---------|
| `src/slangpy_ext/utils/slangpyfunction.cpp` | `NativeFunctionNode::call` — checks `call_data->is_torch_integration()` to sync CUDA streams, and `call_data->is_torch_autograd()` to route through `TorchBridge::call_torch_autograd_hook()`. |
| `src/slangpy_ext/utils/slangpy.cpp` | `get_value_signature()` — fast path for torch tensors using `TorchBridge::get_signature()` (~15ns). Prefixes with `"torch\n"` in the cache key. |

## Key Data Structures

### `TensorBridgeInfo` (C struct, `tensor_bridge_api.h`)
Contains all tensor metadata: `data_ptr`, `shape`/`strides` (caller-provided buffers), `ndim`, `scalar_type` (c10::ScalarType enum), `device_type`/`device_index`, `element_size`, `numel`, `storage_offset`, and flags (`is_contiguous`, `is_cuda`, `requires_grad`).

### `NativeTorchTensorDiffPair` (C++, `slangpytorchtensor.h`)
Pairs a primal tensor with a gradient tensor for autograd. Fields: `primal`, `grad`, `index` (position in saved tensors list), `is_input` (determines grad direction).

### `TorchTensorMarshall` (Python, `torchtensormarshall.py`)
Inherits from `NativeTorchTensorMarshall`. Constructed by `create_torch_tensor_marshall()` factory, which handles three cases:
- `torch.Tensor` → no gradient support
- `NativeTorchTensorDiffPair` with `is_input=True` → creates `d_out` marshall for writing gradients
- `NativeTorchTensorDiffPair` with `is_input=False` → creates `d_in` marshall for reading gradients
- `ReturnContext` → creates output marshall from Slang return type

### Tensor Signature Format
`"[Dn,Sm]"` — `n` = ndim, `m` = c10::ScalarType enum value. Example: `[D2,S6]` = 2D float32.

## Dispatch Flow (CUDA)

1. **Signature lookup** — `get_value_signature()` calls `TorchBridge::get_signature()` → cache key built
2. **Cache hit** → skip to step 5
3. **Cache miss** → `CallData.__init__()` detects torch tensors, imports `torchtensormarshall`, creates `TorchTensorMarshall` per tensor arg
4. **Kernel generation** — type resolution, vectorization, Slang code generation + compilation
5. **CUDA stream sync** — if `is_torch_integration()`, gets PyTorch's current CUDA stream and passes it to the dispatch
6. **Shader cursor writing** — `NativeTorchTensorMarshall::write_shader_cursor_pre_dispatch()`:
   - CUDA backend: writes tensor's `data_ptr()` directly as device address (zero-copy)
   - Non-CUDA backend (D3D12/Vulkan): allocates interop buffer, copies data via `copy_to_buffer()`, writes buffer address
7. **Dispatch** — kernel launched on the synchronized stream
8. **Readback** (non-CUDA only) — `copy_from_buffer()` copies results back to torch tensor

## Autograd Flow

When any input `torch.Tensor` has `requires_grad=True`:

1. `CallData.__init__` sets `torch_autograd = True`
2. At dispatch, C++ routes to `call_torch_autograd_hook()` → Python `torch_autograd_hook()`
3. `find_torch_tensors()` wraps each tensor in `NativeTorchTensorDiffPair`:
   - Read-only bindings → `is_input=True` (gradients written to these in backward)
   - Write-only bindings → `is_input=False` (gradients read from these in backward)
   - Read-write bindings → error (ambiguous grad direction)
4. `TorchAutoGradHook.forward()` runs the forward kernel, saves input tensors for backward
5. On `loss.backward()`, `TorchAutoGradHook.backward()`:
   - Restores tensor references from saved tensors
   - Creates zero gradient tensors for inputs, assigns upstream gradients to outputs
   - Calls `function.bwds(*args, **kwargs)` which generates a backward-derivative kernel
   - Returns computed gradients

## Supported Torch dtypes

| torch dtype | Slang ScalarType | c10 code |
|-------------|------------------|----------|
| `torch.int8` | `int8` | 1 |
| `torch.int16` | `int16` | 2 |
| `torch.int32` | `int32` | 3 |
| `torch.int64` | `int64` | 4 |
| `torch.uint8` | `uint8` | 0 |
| `torch.float16` | `float16` | 5 |
| `torch.float32` | `float32` | 6 |
| `torch.float64` | `float64` | 7 |

## Tests

| Test file | What it covers |
|-----------|----------------|
| `slangpy/tests/utils/test_torch_bridge.py` | Low-level bridge: metadata extraction (CPU/CUDA, various dtypes, non-contiguous, transposed, storage offsets, 0-D to high-D), signature extraction, `is_tensor` detection. Runs in both native and fallback modes via fixture. |
| `slangpy/tests/slangpy_tests/test_torchintegration.py` | End-to-end: `add`/`polynomial` functions with scalars/vectors/arrays/generics, extra vectorization dims, return modes (return/pass/out), autograd forward+backward, cache reuse, structured types. |
| `slangpy/tests/slangpy_tests/test_torchbuffers.py` | CUDA stream synchronization: shared vs non-shared contexts, custom streams, race condition detection with large buffers (100M elements). |

## Common Patterns for Making Changes

### Adding a new torch dtype
1. Add mapping in `_torch_to_scalar_type`, `_torch_to_data_type` (in `torchtensormarshall.py`)
2. Add case in `NativeTorchTensorMarshall::create_output()` switch statement (`slangpytorchtensor.cpp`)
3. Add c10 mapping in `_SCALAR_TYPE_MAP` (`bridge_fallback.py`)
4. Add test case to `test_torchintegration.py` signature parametrize list

### Modifying the bridge API
1. Bump `TENSOR_BRIDGE_API_VERSION` in `tensor_bridge_api.h`
2. Add function pointer type and field to `TensorBridgeAPI` struct
3. Implement in `torch_bridge_impl.cpp` and add to `g_api` static
4. Add native method to `TorchBridge` class (`torch_bridge.h`)
5. Add Python fallback in `bridge_fallback.py`
6. Add tests in `test_torch_bridge.py` for both modes

### Adding autograd support for a new access pattern
1. Update `find_torch_tensors()` in `slangpy/core/calldata.py` — determine `is_input` from binding access
2. Update `TorchAutoGradHook.backward()` in `autogradhook.py` — handle new gradient flow direction
3. Add tests in `test_torchintegration.py`

## Platform Constraints
- **macOS**: PyTorch CUDA is not available; all torch tests are skipped on macOS
- **Metal backend**: Torch integration is not supported (removed from `DEVICE_TYPES` in tests)
- **Non-CUDA backends** (D3D12/Vulkan): Require interop buffers for data transfer; `copy_to_buffer`/`copy_from_buffer` must work (fallback mode will fail for these)
- **CUDA backend**: Zero-copy — torch tensor's device pointer is written directly to shader uniforms
