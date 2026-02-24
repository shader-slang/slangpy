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
      → Python (CallData.find_torch_tensors)              ← SLOW: walks args, wraps in DiffPair
      → Python (TorchAutoGradHook.apply)                  ← unavoidable (PyTorch API)
        → Python (TorchAutoGradHook.forward)              ← SLOW: unpacking, bookkeeping
          → C++ (NativeCallData::call/exec)
        → Python (TorchAutoGradHook.backward)             ← SLOW: tensor restore, grad creation
          → Python (function.bwds)
            → C++ (NativeCallData::call/exec)
```

The Python overhead of `torch_autograd_hook`, `find_torch_tensors`, `TorchAutoGradHook.forward`, and `TorchAutoGradHook.backward` adds ~10-15μs per forward+backward cycle. Previous research has shown it is not possible (or very difficult) to fully interact with the torch autograd system from C++ — `TorchAutoGradHook.apply` must remain in Python. But everything around it can be native.

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
| `src/slangpy_ext/utils/torch_bridge.h` | `TorchBridge::call_torch_autograd_hook()` — lazily imports and calls Python `torch_autograd_hook` |
| `slangpy/core/calldata.py` | `torch_autograd_hook()` — Python function that calls `find_torch_tensors()` then `TorchAutoGradHook.apply()` |
| `slangpy/core/calldata.py` | `CallData.find_torch_tensors()` / `_find_torch_tensors_recurse()` — Python methods that walk args/kwargs, check access patterns, create `NativeTorchTensorDiffPair` |
| `slangpy/torchintegration/autogradhook.py` | `TorchAutoGradHook` — `torch.autograd.Function` subclass with `forward()` and `backward()` doing non-trivial Python work |
| `src/slangpy_ext/utils/slangpy.h` | `NativeCallData` — has `is_torch_autograd()`, `runtime()`, etc. |
| `src/slangpy_ext/utils/slangpytorchtensor.h` | `NativeTorchTensorDiffPair` — pairs primal+grad tensors with `index` and `is_input` |
| `slangpy/bindings/boundvariableruntime.py` | `BoundVariableRuntime` — Python wrapper around `NativeBoundVariableRuntime`, stores access patterns |

## Implementation Steps

### Step 1: Store tensor access pattern natively at build time

**Problem:** `find_torch_tensors` in Python checks `binding.vector_type` (is it an `ITensorType`? what's its `.access`?) and `binding.access[0]`. This requires Python reflection objects. Moving `find_torch_tensors` to C++ requires this information to be available natively.

**Solution:** Add a `torch_autograd_access` field to `NativeBoundVariableRuntime` — an enum value (`none`, `read`, `write`, `error_readwrite`) precomputed at build time.

**Files:**

| File | Change |
|------|--------|
| `src/slangpy_ext/utils/slangpy.h` | Add `AutogradAccess` enum and `m_autograd_access` field + getter/setter to `NativeBoundVariableRuntime` |
| `src/slangpy_ext/utils/slangpy.cpp` | Expose `autograd_access` property in nanobind bindings |
| `slangpy/bindings/boundvariableruntime.py` | In `BoundVariableRuntime.__init__`, compute autograd access from `binding.vector_type` and `binding.access`, set it on the native object |

The computation logic (currently in `_find_torch_tensors_recurse`) is:
```python
if isinstance(binding.vector_type, ITensorType):
    if binding.vector_type.access == TensorAccess.read:       → AutogradAccess.read
    elif binding.vector_type.access == TensorAccess.write:    → AutogradAccess.write
    elif binding.vector_type.access == TensorAccess.read_write: → AutogradAccess.error_readwrite
else:
    if binding.access[0] == AccessType.read:      → AutogradAccess.read
    elif binding.access[0] == AccessType.write:   → AutogradAccess.write
    elif binding.access[0] == AccessType.readwrite: → AutogradAccess.error_readwrite
```

This runs once at build time (Python), never at dispatch time.

### Step 2: Move `find_torch_tensors` to C++

**Problem:** `CallData.find_torch_tensors()` and `_find_torch_tensors_recurse()` are Python methods that walk args/kwargs recursively, detect `torch.Tensor` objects, check access patterns, and create `NativeTorchTensorDiffPair` objects.

**Solution:** Implement `NativeCallData::find_torch_tensors()` in C++.

**Files:**

| File | Change |
|------|--------|
| `src/slangpy_ext/utils/slangpy.h` | Add `find_torch_tensors(nb::list args, nb::dict kwargs)` method to `NativeCallData`, returning `nb::list` of `NativeTorchTensorDiffPair` |
| `src/slangpy_ext/utils/slangpy.cpp` | Implement using `TorchBridge::is_tensor()`, `NativeBoundVariableRuntime::autograd_access()`, and `NativeTorchTensorDiffPair` construction |

The C++ implementation walks `m_runtime->args()` / `m_runtime->kwargs()` in parallel with the actual args/kwargs. For each binding with children, it recurses into `dict` args. For torch tensors, it reads `autograd_access` from the binding, creates a `NativeTorchTensorDiffPair`, and replaces the tensor in the args/kwargs with the pair. Returns the list of all pairs.

Key details:
- Uses `TorchBridge::instance().is_tensor(obj)` for tensor detection (fast, ~28ns native)
- Reads `binding->autograd_access()` for input/output determination (pure C++ field read)
- Throws `std::runtime_error` for `AutogradAccess::error_readwrite` (same as current Python)
- Must modify args list and kwargs dict **in place** (same semantics as current Python)
- Must handle recursive `dict` arguments by following `binding->children()`

### Step 3: Add `create_zeros_like_tensor` to TorchBridge

**Problem:** `TorchAutoGradHook.backward()` calls `torch.zeros_like(pair.primal)` to create gradient tensors. Moving backward logic to C++ requires this capability natively.

**Solution:** Add `create_zeros_like_tensor()` to `TorchBridge`.

**Files:**

| File | Change |
|------|--------|
| `src/slangpy_torch/tensor_bridge_api.h` | Add `create_zeros_like` function pointer to `TensorBridgeAPI`, bump `TENSOR_BRIDGE_API_VERSION` |
| `src/slangpy_torch/torch_bridge_impl.cpp` | Implement using `torch::zeros_like()` C++ API |
| `src/slangpy_ext/utils/torch_bridge.h` | Add `create_zeros_like_tensor(PyObject* tensor)` method with native + fallback paths |
| `slangpy/torchintegration/bridge_fallback.py` | Add `create_zeros_like(tensor)` Python fallback calling `torch.zeros_like()` |

The method takes an existing tensor and returns a new zero tensor with the same shape, dtype, and device. For the native path, it uses libtorch's `torch::zeros_like()`. For the fallback, it calls `torch.zeros_like()` via cached Python function handle.

**Note:** An alternative is `create_empty_tensor` (already exists) + a separate zero-fill, but `zeros_like` is a single allocation+fill that PyTorch optimizes internally.

### Step 4: Add `autograd_forward()` to NativeCallData

**Problem:** `TorchAutoGradHook.forward()` currently does significant Python work: separating pairs into input/output lists, calling the forward kernel, clearing tensor references, collecting output tensors.

**Solution:** New C++ method `NativeCallData::autograd_forward()` encapsulating most of the forward logic.

**Files:**

| File | Change |
|------|--------|
| `src/slangpy_ext/utils/slangpy.h` | Declare `autograd_forward()` on `NativeCallData` |
| `src/slangpy_ext/utils/slangpy.cpp` | Implement + expose via nanobind |

**Method signature:**
```cpp
/// Run the forward kernel and prepare data for autograd.
/// Returns a tuple of (input_tensors, output_tensors, result, pairs).
/// - input_tensors: list of primal tensors from input pairs (for save_for_backward)
/// - output_tensors: list of primal tensors from output pairs (returned to autograd)
/// - result: the kernel result (or None)
/// - pairs: updated pairs list (with _result pair appended if needed)
nb::tuple autograd_forward(
    ref<NativeCallRuntimeOptions> opts,
    nb::list args,
    nb::dict kwargs,
    nb::list pairs
);
```

**Implementation logic** (mirrors current `TorchAutoGradHook.forward`):
1. Separate pairs into input/output lists by checking `pair.is_input`
2. Call `this->call(opts, args, kwargs)` to run the forward kernel
3. Clear tensor references on all pairs via `pair.clear_tensors()`
4. If result is not None and `_result` not in kwargs, create a new output `NativeTorchTensorDiffPair` for the result and append to pairs
5. Return `(input_tensors, output_tensors, result, pairs)`

### Step 5: Add `autograd_backward()` to NativeCallData

**Problem:** `TorchAutoGradHook.backward()` does significant Python work: restoring tensors from saved state, creating zero gradient tensors, assigning upstream gradients, calling `function.bwds()`.

**Solution:** New C++ method `NativeCallData::autograd_backward()`.

**Files:**

| File | Change |
|------|--------|
| `src/slangpy_ext/utils/slangpy.h` | Declare `autograd_backward()` on `NativeCallData` |
| `src/slangpy_ext/utils/slangpy.cpp` | Implement + expose via nanobind |

**Method signature:**
```cpp
/// Run the backward pass: restore tensors, create gradients, call bwds kernel.
/// Returns tuple of input gradients (matching order of input tensors from forward).
nb::tuple autograd_backward(
    nb::handle function_node,   // The FunctionNode (for calling .bwds)
    nb::list pairs,             // The NativeTorchTensorDiffPair list from forward
    nb::list args,              // Saved args (with DiffPair placeholders)
    nb::dict kwargs,            // Saved kwargs (with DiffPair placeholders)
    nb::list saved_tensors,     // Input tensors from save_for_backward
    nb::tuple grad_outputs,     // Upstream gradients for output tensors
    nb::handle device           // Device for contiguity check
);
```

**Implementation logic** (mirrors current `TorchAutoGradHook.backward`):
1. Iterate over pairs. For each input pair:
   - Restore `primal` from `saved_tensors[input_idx]`
   - If `primal.requires_grad`, create `grad = torch.zeros_like(primal)` via `TorchBridge::create_zeros_like_tensor()`
   - Append grad to input_grads list
2. For each output pair:
   - Set `primal = None`
   - Set `grad = grad_outputs[output_idx]`
   - If device is not CUDA, make grad contiguous (via a `contiguous()` call — need to add to TorchBridge or call through Python)
3. Call `function_node.attr("bwds")(*args, **kwargs)` — this calls the Python `FunctionNode.bwds` property which returns a `FunctionNodeBwds`, then `__call__` dispatches through `_native_call` back into C++
4. Return tuple of input gradients

**Key detail on calling `bwds`:** The `function.bwds` property creates `FunctionNodeBwds(self)` which is a Python object. Calling it goes through `FunctionNode.__call__` → `FunctionNode.call` → `_native_call`. This is one Python→C++→Python→C++ round trip that is hard to avoid without caching the bwds function node. For now, using `function_node.attr("bwds")(*args, **kwargs)` from C++ is acceptable — it's a single Python call per backward pass. A future optimization could cache the bwds `NativeFunctionNode` on `NativeCallData` at build time.

### Step 6: Rewrite `call_torch_autograd_hook()` in TorchBridge

**Problem:** Currently `TorchBridge::call_torch_autograd_hook()` calls the Python function `torch_autograd_hook` in `calldata.py`, which then does preparation work in Python before calling `TorchAutoGradHook.apply()`.

**Solution:** Do the preparation work in C++ and call `TorchAutoGradHook.apply()` directly.

**Files:**

| File | Change |
|------|--------|
| `src/slangpy_ext/utils/torch_bridge.h` | Rewrite `call_torch_autograd_hook()`, cache `TorchAutoGradHook` class, remove `m_py_torch_autograd_hook` |
| `src/slangpy_ext/utils/slangpyfunction.cpp` | No changes needed — already calls `TorchBridge::call_torch_autograd_hook()` |

**New implementation of `call_torch_autograd_hook()`:**
```cpp
nb::object call_torch_autograd_hook(
    nb::handle function_node,
    nb::handle call_data,        // NativeCallData*
    nb::handle options,          // NativeCallRuntimeOptions*
    nb::args args,
    nb::kwargs kwargs
) const
{
    init_autograd_hook();  // Lazy-init cached TorchAutoGradHook class

    // 1. Convert args to mutable list, kwargs to mutable dict
    nb::list args_list;
    for (auto arg : args) args_list.append(arg);
    nb::dict kwargs_dict;
    for (auto [k, v] : kwargs) kwargs_dict[k] = v;

    // 2. Call NativeCallData::find_torch_tensors (C++)
    NativeCallData* cd = nb::cast<NativeCallData*>(call_data);
    nb::list pairs = cd->find_torch_tensors(args_list, kwargs_dict);

    // 3. Extract input tensors
    nb::list inputs;
    for (auto pair_obj : pairs) {
        auto* pair = nb::cast<NativeTorchTensorDiffPair*>(pair_obj);
        if (pair->is_input)
            inputs.append(pair->primal);
    }

    // 4. Build options tuple and call TorchAutoGradHook.apply
    nb::tuple options_tuple = nb::make_tuple(
        function_node, call_data, options, args_list, kwargs_dict, pairs
    );
    nb::object results = m_autograd_hook_class.attr("apply")(options_tuple, *nb::cast<nb::args>(inputs));

    // 5. Extract result
    if (!results.is_none() && nb::len(results) > 0) {
        return results[nb::int_(nb::len(results) - 1)];
    }
    return nb::none();
}
```

**Lazy init:**
```cpp
void init_autograd_hook() const {
    if (m_autograd_hook_initialized) return;
    nb::module_ hook_module = nb::module_::import_("slangpy.torchintegration.autogradhook");
    m_autograd_hook_class = hook_module.attr("TorchAutoGradHook");
    m_autograd_hook_initialized = true;
}
```

Add to TorchBridge private members:
```cpp
mutable bool m_autograd_hook_initialized = false;
mutable nb::object m_autograd_hook_class;
```

And clean up in `reset()`:
```cpp
m_autograd_hook_initialized = false;
m_autograd_hook_class.reset();
```

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

Steps can be done in this order given their dependencies:

```
Step 1 (autograd_access field)
  └→ Step 2 (find_torch_tensors in C++) ─────────────┐
Step 3 (create_zeros_like)                            │
  └→ Step 5 (autograd_backward) ──────────────────────┤
Step 4 (autograd_forward) ────────────────────────────┤
                                                      │
Step 6 (rewrite call_torch_autograd_hook) ←───────────┘
  └→ Step 7 (simplify TorchAutoGradHook)
      └→ Step 8 (clean up calldata.py)
```

Steps 1, 3, and 4 are independent of each other and can be done in parallel. Steps 2, 5 depend on 1 and 3 respectively. Steps 6-8 are sequential and depend on all prior steps.

## Files Changed Summary

| File | Changes |
|------|---------|
| `src/slangpy_ext/utils/slangpy.h` | Add `AutogradAccess` enum to `NativeBoundVariableRuntime`, add `find_torch_tensors()` / `autograd_forward()` / `autograd_backward()` to `NativeCallData` |
| `src/slangpy_ext/utils/slangpy.cpp` | Implement new methods, expose via nanobind |
| `src/slangpy_ext/utils/torch_bridge.h` | Rewrite `call_torch_autograd_hook()`, add `create_zeros_like_tensor()`, cache `TorchAutoGradHook` class |
| `src/slangpy_torch/tensor_bridge_api.h` | Add `create_zeros_like` function pointer, bump API version |
| `src/slangpy_torch/torch_bridge_impl.cpp` | Implement `create_zeros_like` |
| `slangpy/torchintegration/bridge_fallback.py` | Add `create_zeros_like` fallback |
| `slangpy/torchintegration/autogradhook.py` | Simplify `forward()` / `backward()` to thin wrappers |
| `slangpy/bindings/boundvariableruntime.py` | Compute and set `autograd_access` at build time |
| `slangpy/core/calldata.py` | Remove `find_torch_tensors()`, `_find_torch_tensors_recurse()`, `torch_autograd_hook()` |
