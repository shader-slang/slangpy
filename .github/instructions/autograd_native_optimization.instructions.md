---
description: Instructions for moving the PyTorch autograd dispatch hot path from Python to C++ for performance
---

# Native Autograd Dispatch Optimization

## Problem

The current autograd hot path bounces between C++ and Python multiple times per call:

```
C++ (NativeFunctionNode::call)
  → C++ (TorchBridge::call_torch_autograd_hook)
    → Python (calldata.torch_autograd_hook)              ← SLOW
      → C++ (NativeCallData::find_torch_tensors)          ← DONE: moved to C++
      → Python (TorchAutoGradHook.apply)                  ← unavoidable (PyTorch API)
        → Python (TorchAutoGradHook.forward)              ← SLOW: unpacking, bookkeeping
          → C++ (NativeCallData::call/exec)
        → Python (TorchAutoGradHook.backward)             ← SLOW: tensor restore, grad creation
          → Python (function.bwds)
            → C++ (NativeCallData::call/exec)
```

The Python overhead of `torch_autograd_hook`, `TorchAutoGradHook.forward`, and `TorchAutoGradHook.backward` adds ~10-15μs per forward+backward cycle. Previous research has shown it is not possible (or very difficult) to fully interact with the torch autograd system from C++ — `TorchAutoGradHook.apply` must remain in Python. But everything around it can be native.

## Target Architecture

```
C++ (NativeFunctionNode::call)
  → C++ (prepare_autograd + call TorchAutoGradHook.apply)  ← find tensors, build DiffPairs, extract inputs
    → Python (TorchAutoGradHook.apply)                      ← unavoidable (PyTorch API)
      → Python (TorchAutoGradHook.forward)                   ← THIN: calls C++ immediately
        → C++ (NativeCallData::autograd_forward)              ← run kernel, collect outputs, clear pairs
      → Python (TorchAutoGradHook.backward)                  ← THIN: calls C++ immediately
        → C++ (NativeCallData::autograd_backward)             ← restore tensors, create grads, call bwds
```

The only Python code involved should be:
1. A direct call from native code to `TorchAutoGradHook.apply`
2. `TorchAutoGradHook.forward` and `TorchAutoGradHook.backward` as thin wrappers calling straight into native code

## Current Code Map

| File | Current Role |
|------|-------------|
| `src/slangpy_ext/utils/slangpyfunction.cpp` | `NativeFunctionNode::call` — entry point, detects `torch_autograd`, calls `TorchBridge::call_torch_autograd_hook()` |
| `src/slangpy_ext/utils/torch_bridge.h` | `TorchBridge::call_torch_autograd_hook()` — does prep in C++ (find_torch_tensors, extract inputs), calls `TorchAutoGradHook.apply()` directly |
| `slangpy/core/calldata.py` | `torch_autograd_hook()` — Python function (no longer called by TorchBridge, can be removed in step 8) |
| `slangpy/core/calldata.py` | `CallData._build_autograd_access_list()` — builds the flat access list at build time (once per signature) |
| `src/slangpy_ext/utils/slangpy.h` | `NativeCallData` — has `find_torch_tensors()`, `autograd_access_list`, `is_torch_autograd()`, `runtime()`, etc. |
| `src/slangpy_ext/utils/slangpy.cpp` | `NativeCallData::find_torch_tensors()` / `find_torch_tensors_recurse()` — C++ implementation that walks args/kwargs and creates `NativeTorchTensorDiffPair` objects |
| `src/sgl/utils/slangpy.h` | `AutogradAccess` enum — `none`, `read`, `write`, `readwrite` |
| `slangpy/torchintegration/autogradhook.py` | `TorchAutoGradHook` — `torch.autograd.Function` subclass with `forward()` and `backward()` doing non-trivial Python work |
| `src/slangpy_ext/utils/slangpytorchtensor.h` | `NativeTorchTensorDiffPair` — pairs primal+grad tensors with `index` and `is_input` |
| `slangpy/bindings/boundvariableruntime.py` | `BoundVariableRuntime` — Python wrapper around `NativeBoundVariableRuntime`, stores access patterns |

## Completed Steps

### Step 1+2: Precompute access patterns and move `find_torch_tensors` to C++ ✅

**Design:** Instead of storing an access enum on each `NativeBoundVariableRuntime` (which would require the C++ recursive walk to mirror the binding tree structure), a flat `std::vector<AutogradAccess>` is precomputed at build time and stored on `NativeCallData`. The order of the list matches the deterministic recursive walk order through args/kwargs. At dispatch time, the C++ `find_torch_tensors` steps through the list with a simple index as it encounters tensors.

**How the access list is built (build time, Python, once per signature):**
In `CallData.build()`, after `self.runtime = BoundCallRuntime(bindings)`, if `self.torch_autograd` is True, `_build_autograd_access_list()` is called. This walks the unpacked args/kwargs in the same recursive order as `find_torch_tensors`, checking each torch tensor's binding:
- If `binding.vector_type` is an `ITensorType`: reads `binding.vector_type.access` (a `TensorAccess` enum)
- Otherwise: reads `binding.access[0]` (an `AccessType` enum)
- Maps to the corresponding `AutogradAccess` value (`read`, `write`, `readwrite`, or `none`)
- Appends to a flat list, stored via `self.autograd_access_list = access_list`

**How tensors are found (dispatch time, C++, every call):**
`NativeCallData::find_torch_tensors(nb::list args, nb::dict kwargs)` walks args then kwargs. For each element:
- If it's a `dict`: recurse into its values
- If it's a torch tensor (checked via `TorchBridge::is_tensor()`): read `m_autograd_access_list[access_idx++]`, create a `NativeTorchTensorDiffPair`, replace the tensor in the list/dict
- Otherwise: leave as-is

**Files changed:**

| File | What was added |
|------|---------------|
| `src/sgl/utils/slangpy.h` | `AutogradAccess` enum with `SGL_ENUM_INFO` / `SGL_ENUM_REGISTER` |
| `src/slangpy_ext/utils/slangpy.h` | `m_autograd_access_list` field + getter/setter on `NativeCallData`; `find_torch_tensors()` and `find_torch_tensors_recurse()` method declarations |
| `src/slangpy_ext/utils/slangpy.cpp` | Implementation of `find_torch_tensors()` and `find_torch_tensors_recurse()`; nanobind bindings for `AutogradAccess` enum, `autograd_access_list` property, and `find_torch_tensors` method |
| `slangpy/core/calldata.py` | `_build_autograd_access_list()` method; call in `build()` when `torch_autograd` is True; removed old Python `find_torch_tensors()` and `_find_torch_tensors_recurse()` methods (the native `NativeCallData::find_torch_tensors` on the base class is now used directly) |

### Step 3: Add `create_zeros_like_tensor` to TorchBridge ✅

**Problem:** `TorchAutoGradHook.backward()` calls `torch.zeros_like(pair.primal)` to create gradient tensors. Moving backward logic to C++ requires this capability natively.

**Solution:** Added `create_zeros_like_tensor()` to `TorchBridge`.

The method takes an existing tensor and returns a new zero tensor with the same shape, dtype, and device. For the native path, it uses libtorch's `torch::zeros_like()`. For the fallback, it calls `torch.zeros_like()` via cached Python function handle.

**Files changed:**

| File | What was added |
|------|---------------|
| `src/slangpy_torch/tensor_bridge_api.h` | Added `TensorBridge_CreateZerosLikeFn` function pointer typedef, added `create_zeros_like` to `TensorBridgeAPI` struct, bumped `TENSOR_BRIDGE_API_VERSION` to 7 |
| `src/slangpy_torch/torch_bridge_impl.cpp` | Implemented `tensor_bridge_create_zeros_like()` using `torch::zeros_like()` C++ API, added to `g_api` static |
| `src/slangpy_ext/utils/torch_bridge.h` | Added `create_zeros_like_tensor()` public methods (PyObject* and nb::handle overloads) with native + fallback paths, `python_create_zeros_like()` private method, cached `m_py_create_zeros_like` handle |
| `slangpy/torchintegration/bridge_fallback.py` | Added `create_zeros_like(tensor)` Python fallback calling `torch.zeros_like()` |

### Step 4: Add `autograd_forward()` to NativeCallData ✅

**Problem:** `TorchAutoGradHook.forward()` currently does significant Python work: separating pairs into input/output lists, calling the forward kernel, clearing tensor references, collecting output tensors.

**Solution:** Added C++ method `NativeCallData::autograd_forward()` encapsulating most of the forward logic.

**Implementation logic** (mirrors current `TorchAutoGradHook.forward`):
1. Separate pairs into input/output lists by checking `pair.is_input`
2. Call `this->exec(opts, nullptr, args, kwargs)` to run the forward kernel
3. Clear tensor references on all pairs (set primal/grad to None)
4. If result is not None and `_result` not in kwargs, create a new output `NativeTorchTensorDiffPair` for the result and append to pairs
5. Return `(input_tensors, output_tensors, result, pairs)`

**Files changed:**

| File | What was added |
|------|---------------|
| `src/slangpy_ext/utils/slangpy.h` | Declared `autograd_forward(ref<NativeCallRuntimeOptions> opts, nb::list args, nb::dict kwargs, nb::list pairs)` returning `nb::tuple` on `NativeCallData` |
| `src/slangpy_ext/utils/slangpy.cpp` | Implemented `autograd_forward()` — separates pairs, runs forward kernel via `exec()`, clears tensor refs, handles `_result` pair creation; exposed via nanobind binding |

### Step 5: Add `autograd_backward()` to NativeCallData ✅

**Problem:** `TorchAutoGradHook.backward()` does significant Python work: restoring tensors from saved state, creating zero gradient tensors, assigning upstream gradients, calling `function.bwds()`.

**Solution:** Added C++ method `NativeCallData::autograd_backward()` encapsulating most of the backward logic.

**Method signature:**
```cpp
nb::tuple autograd_backward(
    nb::handle function_node,   // The FunctionNode (for calling .bwds)
    nb::list pairs,             // The NativeTorchTensorDiffPair list from forward
    nb::list args,              // Saved args (with DiffPair placeholders)
    nb::dict kwargs,            // Saved kwargs (with DiffPair placeholders)
    nb::list saved_tensors,     // Input tensors from save_for_backward
    nb::tuple grad_outputs      // Upstream gradients for output tensors
);
```

**Implementation logic** (mirrors `TorchAutoGradHook.backward`):
1. Iterate over pairs. For each input pair:
   - Restore `primal` from `saved_tensors[input_idx]`
   - If `primal.requires_grad`, create `grad = torch.zeros_like(primal)` via `TorchBridge::create_zeros_like_tensor()`
   - Append grad to input_grads list
2. For each output pair:
   - Set `primal = None`
   - Set `grad = grad_outputs[output_idx]`
   - If device is not CUDA (checked via `m_device->type() != DeviceType::cuda`), make grad contiguous via `pair->grad.attr("contiguous")()`
3. Call `function_node.attr("bwds")(*args, **kwargs)` — this calls the Python `FunctionNode.bwds` property which returns a `FunctionNodeBwds`, then `__call__` dispatches through `_native_call` back into C++
4. Return tuple of input gradients

**Key detail:** The device check for contiguity uses `m_device` from `NativeCallData` directly, so no `device` parameter is needed. The `function.bwds` call is one Python→C++→Python→C++ round trip that is hard to avoid without caching the bwds function node. A future optimization could cache the bwds `NativeFunctionNode` on `NativeCallData` at build time.

**Files changed:**

| File | What was added |
|------|---------------|
| `src/slangpy_ext/utils/slangpy.h` | Declared `autograd_backward()` on `NativeCallData` |
| `src/slangpy_ext/utils/slangpy.cpp` | Implemented `autograd_backward()` — restores tensors, creates zero grads via `TorchBridge::create_zeros_like_tensor()`, assigns upstream grads, calls `function.bwds()`, handles non-CUDA contiguity; exposed via nanobind binding |

### Step 6: Rewrite `call_torch_autograd_hook()` in TorchBridge ✅

**Problem:** `TorchBridge::call_torch_autograd_hook()` called the Python function `torch_autograd_hook` in `calldata.py`, which did preparation work in Python before calling `TorchAutoGradHook.apply()`.

**Solution:** Rewrote `call_torch_autograd_hook()` to do all preparation in C++ and call `TorchAutoGradHook.apply()` directly. Added separate lazy init for the autograd hook class (`init_autograd_hook()`), replaced `m_py_torch_autograd_hook` with `m_autograd_hook_class` + `m_autograd_hook_initialized`.

**Implementation:**
1. Convert args/kwargs to mutable list/dict
2. Call `NativeCallData::find_torch_tensors()` via `call_data.attr("find_torch_tensors")`
3. Extract input tensors by iterating pairs (via attribute access to avoid circular header dependency)
4. Build options tuple, call `TorchAutoGradHook.apply(options_tuple, *inputs)`
5. Extract and return the last result element

**Note:** The pair iteration in `call_torch_autograd_hook()` uses attribute access (`pair.attr("is_input")`, `pair.attr("primal")`) rather than direct C++ cast to `NativeTorchTensorDiffPair*` because `torch_bridge.h` cannot include `slangpytorchtensor.h` (circular dependency through `slangpy.h`). This is acceptable since this code runs once per autograd call, not in a tight loop.

**Files changed:**

| File | What was changed |
|------|------------------|
| `src/slangpy_ext/utils/torch_bridge.h` | Rewrote `call_torch_autograd_hook()` to do prep in C++ and call `TorchAutoGradHook.apply()` directly; added `init_autograd_hook()` for lazy init; replaced `m_py_torch_autograd_hook` with `m_autograd_hook_class` + `m_autograd_hook_initialized`; removed autograd hook import from `init_python_fallback()`; updated `reset()` |

### Step 7: Simplify TorchAutoGradHook Python class

**Problem:** `forward()` and `backward()` currently do non-trivial Python work.

**Solution:** Make them thin wrappers that call the new C++ methods.

**File:** `slangpy/torchintegration/autogradhook.py`

**New `forward()`:**
```python
@staticmethod
def forward(ctx, options, *tensors):
    function, forwards_cd, rt_options, args, kwargs, pairs = options
    ctx.function = function
    ctx.forwards_cd = forwards_cd

    # Call native forward — handles kernel dispatch, pair bookkeeping, output collection
    input_tensors, output_tensors, result, pairs = forwards_cd.autograd_forward(
        rt_options, args, kwargs, pairs
    )

    ctx.args = args
    ctx.kwargs = kwargs
    ctx.pairs = pairs
    ctx.save_for_backward(*input_tensors)

    res = tuple(output_tensors)
    if result is not None:
        res += (result,)
    return res

@staticmethod
def backward(ctx, *grad_outputs):
    # Call native backward — handles tensor restoration, grad creation, bwds dispatch
    input_grads = ctx.forwards_cd.autograd_backward(
        ctx.function, ctx.pairs, ctx.args, ctx.kwargs,
        list(ctx.saved_tensors), grad_outputs,
        ctx.function.module.device
    )
    return (None,) + tuple(input_grads)
```

### Step 8: Clean up calldata.py

**Files:**

| File | Change |
|------|--------|
| `slangpy/core/calldata.py` | Remove `find_torch_tensors()`, `_find_torch_tensors_recurse()`, and `torch_autograd_hook()` — all moved to C++ |

The `torch_autograd_hook` function is no longer called from `TorchBridge` (which now calls `TorchAutoGradHook.apply` directly). The `find_torch_tensors` method is replaced by the native `NativeCallData::find_torch_tensors()`. These can be removed entirely or left as deprecated stubs.

## Testing

Existing tests cover all the behavior being moved:

| Test file | Coverage |
|-----------|----------|
| `slangpy/tests/slangpy_tests/test_torchintegration.py` | End-to-end autograd forward+backward, including scalars/vectors/arrays/generics, return modes, cache reuse |
| `slangpy/tests/slangpy_tests/test_torchbuffers.py` | CUDA stream synchronization, shared vs non-shared contexts |
| `slangpy/tests/utils/test_torch_bridge.py` | Low-level bridge tests (both native and fallback modes) |
| `slangpy/benchmarks/test_benchmark_autograd.py` | Performance benchmarks for four autograd approaches |

Run the full test suite after each step to catch regressions:
```bash
pytest slangpy/tests/slangpy_tests/test_torchintegration.py -v
pytest slangpy/tests/slangpy_tests/test_torchbuffers.py -v
pytest slangpy/tests/utils/test_torch_bridge.py -v
```

## Build

After C++ changes, rebuild:
```bash
cmake --build --preset windows-msvc-debug
```

## Expected Performance Impact

| Source | Before | After |
|--------|--------|-------|
| `torch_autograd_hook` Python function | ~1-2μs | Eliminated (C++) |
| `find_torch_tensors` Python walk | ~2-5μs per tensor | Eliminated (C++) |
| `TorchAutoGradHook.forward` Python bookkeeping | ~1-2μs | ~200-500ns (thin wrapper) |
| `TorchAutoGradHook.backward` Python bookkeeping | ~3-5μs | ~200-500ns (thin wrapper) |
| **Total Python overhead per fwd+bwd cycle** | **~10-15μs** | **~1-2μs** |

## Dependency Order

```
Steps 1+2 (access list + find_torch_tensors in C++) ─── DONE ───┐
Step 3 (create_zeros_like) ─────────────────────────── DONE ────┤
  └→ Step 5 (autograd_backward) ───────────────────── DONE ────┤
Step 4 (autograd_forward) ─────────────────────────── DONE ────┤
                                                                 │
Step 6 (rewrite call_torch_autograd_hook) ──────────── DONE ────┘
  └→ Step 7 (simplify TorchAutoGradHook)
      └→ Step 8 (clean up calldata.py)
```

Step 7 is the next step to implement. Steps 7-8 are sequential.

## Files Changed Summary

### Already changed (Steps 1-6)

| File | Changes |
|------|---------|
| `src/sgl/utils/slangpy.h` | Added `AutogradAccess` enum (`none`, `read`, `write`, `readwrite`) |
| `src/slangpy_ext/utils/slangpy.h` | Added `m_autograd_access_list` + getter/setter, `find_torch_tensors()`, `find_torch_tensors_recurse()`, `autograd_forward()`, `autograd_backward()` to `NativeCallData` |
| `src/slangpy_ext/utils/slangpy.cpp` | Implemented `find_torch_tensors()` / `find_torch_tensors_recurse()` / `autograd_forward()` / `autograd_backward()`; nanobind bindings for `AutogradAccess`, `autograd_access_list`, `find_torch_tensors`, `autograd_forward`, `autograd_backward` |
| `slangpy/core/calldata.py` | Added `_build_autograd_access_list()`; call site in `build()` after `self.runtime` is set; removed old Python `find_torch_tensors()` / `_find_torch_tensors_recurse()` |
| `src/slangpy_torch/tensor_bridge_api.h` | Added `TensorBridge_CreateZerosLikeFn` typedef, `create_zeros_like` in `TensorBridgeAPI` struct, bumped `TENSOR_BRIDGE_API_VERSION` to 7 |
| `src/slangpy_torch/torch_bridge_impl.cpp` | Implemented `tensor_bridge_create_zeros_like()`, added to `g_api` |
| `src/slangpy_ext/utils/torch_bridge.h` | Added `create_zeros_like_tensor()`; rewrote `call_torch_autograd_hook()` to do prep in C++ and call `TorchAutoGradHook.apply()` directly; added `init_autograd_hook()`, `m_autograd_hook_class`, `m_autograd_hook_initialized`; removed `m_py_torch_autograd_hook` |
| `slangpy/torchintegration/bridge_fallback.py` | Added `create_zeros_like()` fallback |

### Still to change (Steps 7-8)

| File | Changes |
|------|---------|
| `slangpy/torchintegration/autogradhook.py` | Simplify `forward()` / `backward()` to thin wrappers calling `autograd_forward()` / `autograd_backward()` |
| `slangpy/core/calldata.py` | Remove `torch_autograd_hook()` (no longer called from TorchBridge) |
