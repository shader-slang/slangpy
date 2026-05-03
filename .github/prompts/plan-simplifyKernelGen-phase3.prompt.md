## Phase 3: Direct Compute Kernel Invocation

**Goal**: When the user's Slang function is ALREADY a `[shader("compute")]` entry point (or can trivially be one), skip kernel generation entirely and dispatch the pre-written shader directly.

**Parent plan**: [plan-simplifyKernelGen.prompt.md](plan-simplifyKernelGen.prompt.md)

---

### Step 3.1: Detection

In the function resolution phase, detect when the target Slang function:
- Has `[shader("compute")]` attribute
- Has parameter types that SlangPy can bind directly (uniforms, buffers, textures)
- Has explicit thread count specified by the user (already supported via `function.set_thread_count()`)

---

### Step 3.2: Direct dispatch path

When eligible:
- Skip Phase 2 (kernel generation) entirely
- Create a `ComputePipeline` directly from the user's shader
- Map Python arguments to entry point parameters using the type marshalling but without code generation
- Dispatch directly

---

### Step 3.3: Argument binding

Leverage Phase 2's per-argument binding infrastructure — the same cursor write logic that writes individual uniform params would write to the pre-written shader's entry point params.

---

### Step 3.4: Tests

**Gating test** — assert CURRENT behavior so it breaks when Phase 3 is implemented:

| Test | Slang Source | Args | Asserts (current behavior) | Breaks when |
|------|-------------|------|---------------------------|-------------|
| `test_gate_compute_shader_generates_wrapper` | Source with `[shader("compute")] void my_kernel(...)` function, test calls a helper function in the same module | N/A | SlangPy generates its own `compute_main` wrapper; user's `[shader("compute")]` is ignored | Step 3.1 |

**Post-implementation tests** — should pass AFTER Phase 3 is complete:

- `test_phase3_direct_dispatch`: dispatch a pre-written `[shader("compute")]` kernel directly, verify no wrapper generated
- `test_phase3_requires_thread_count`: verify error when thread count not specified
- `test_phase3_scalar_params`: verify scalar uniform params bind correctly
- `test_phase3_buffer_params`: verify `RWStructuredBuffer` params bind correctly
- `test_phase3_texture_params`: verify texture params bind correctly
