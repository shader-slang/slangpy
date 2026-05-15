# Plan: Replace ShaderModel with String-Based Profiles

## TL;DR
Replace the `ShaderModel` enum throughout SGL with string-based Slang profiles (e.g. `"sm_6_6"`, `"glsl_450"`, `"spirv_1_5"`). The device will expose a list of supported profiles derived from RHI capabilities. Clean break — no deprecated aliases.

## Context
- `ShaderModel` is a D3D12-centric enum (sm_6_0 through sm_6_7) used as a proxy for Slang target profiles
- Vulkan in slang-rhi already "emulates" shader models by mapping Vulkan features to SM levels (approximate)
- Slang's `findProfile()` natively accepts strings like `"sm_6_6"`, `"glsl_450"`, `"spirv_1_5"` etc.
- slang-rhi capabilities already contain profile-like atoms: `_sm_6_X`, `_spirv_1_X`, `_GLSL_4XX`, `metallib_X_X`, `_cuda_sm_X_X`
- Device already has `capabilities()` returning `std::vector<std::string>` and `has_capability()` for string-based checks

## Phase 1: C++ Core Changes

### Step 1: Add profile support to Device
**Files:** `src/sgl/device/device.h`, `src/sgl/device/device.cpp`

- Add `std::vector<std::string> m_supported_profiles` member
- Add `const std::vector<std::string>& supported_profiles() const`
- Add `bool has_profile(std::string_view profile) const`
- In constructor, after querying RHI capabilities, derive supported profiles by filtering capabilities that match known profile patterns (`_sm_6_*`, `_spirv_1_*`, `_GLSL_*`, `metallib_*`, `_cuda_sm_*`)
- Remove `m_supported_shader_model` member and `supported_shader_model()` accessor
- Remove `ShaderModel` from `Device::to_string()` output, replace with profiles list
- Add `std::string default_profile() const` — returns the highest/best profile for the device type (replaces the auto-select logic)

### Step 2: Replace ShaderModel in SlangCompilerOptions
**Files:** `src/sgl/device/shader.h`, `src/sgl/device/shader.cpp`

- In `SlangCompilerOptions`: replace `ShaderModel shader_model{ShaderModel::unknown}` with `std::string profile`
- In `SlangSession::create_session()`:
  - When `profile` is empty, call `device->default_profile()` to get auto-selected profile
  - Validate profile against `device->supported_profiles()` (or `has_profile()`)
  - Pass profile string directly to `global_session->findProfile()`
  - Keep the SM 6.7 ray payload workaround as a temporary profile-string check (if `profile == "sm_6_7"`, downgrade to `"sm_6_6"`)
  - Remove `get_shader_model_major_version()` / `get_shader_model_minor_version()` calls; parse profile string for `__SHADER_TARGET_MAJOR`/`__SHADER_TARGET_MINOR` defines (or remove those defines if no `.slang` files use them — confirmed none do)
  - Remove the CUDA special-case comment since profiles handle CUDA natively (`"cuda_sm_7_0"` etc.)

### Step 3: Remove ShaderModel enum
**Files:** `src/sgl/device/types.h`

- Delete the `ShaderModel` enum, `SGL_ENUM_INFO`, `SGL_ENUM_REGISTER`, `get_shader_model_major_version()`, `get_shader_model_minor_version()`

## Phase 2: Python Bindings

### Step 4: Update nanobind bindings
**Files:** `src/slangpy_ext/device/types.cpp`, `src/slangpy_ext/device/shader.cpp`, `src/slangpy_ext/device/device.cpp`, `src/slangpy_ext/py_doc.h`

- Remove `ShaderModel` enum binding from `types.cpp`
- In `shader.cpp`: change `SGL_DICT_TO_DESC_FIELD(shader_model, ShaderModel)` to `SGL_DICT_TO_DESC_FIELD(profile, std::string)` and update `.def_rw("shader_model", ...)` → `.def_rw("profile", ...)`
- In `device.cpp`: replace `supported_shader_model` property with `supported_profiles` (list) and `default_profile` (string) properties
- Remove ShaderModel-related entries from `py_doc.h`

### Step 5: Update `__init__.pyi` type stubs
**File:** `slangpy/__init__.pyi`

- Remove `ShaderModel` class
- Update `SlangCompilerOptions.shader_model` → `SlangCompilerOptions.profile` (type: `str`)
- Update `SlangCompilerOptionsDict` entry
- Replace `Device.supported_shader_model` with `Device.supported_profiles` (type: `list[str]`) and `Device.default_profile` (type: `str`)

## Phase 3: Python Layer

### Step 6: Update test helpers
**File:** `slangpy/testing/helpers.py`

- `dispatch_compute()`: replace `shader_model: spy.ShaderModel = spy.ShaderModel.sm_6_6` parameter with `profile: str = "sm_6_6"`
- Replace the Metal/CUDA special-case (`shader_model = spy.ShaderModel.sm_6_0`) — instead use `device.default_profile()` when profile not suitable for device type
- Replace `shader_model > device.supported_shader_model` skip logic with `profile not in device.supported_profiles`
- Replace `compiler_options["shader_model"]` with `compiler_options["profile"]`

### Step 7: Update tests
**Files:**
- `slangpy/tests/device/slang/test_float16.py`
- `slangpy/tests/device/slang/test_float64.py`
- `slangpy/tests/device/slang/test_uint64.py`

- Replace `spy.ShaderModel.sm_6_X` parametrize values with profile strings (`"sm_6_2"`, `"sm_6_3"`, etc.)
- Update function signatures from `shader_model: spy.ShaderModel` to `profile: str`
- Update calls to `dispatch_compute(shader_model=...)` → `dispatch_compute(profile=...)`

### Step 8: Update samples/docs
**Files:**
- `samples/tutorials/compute_shader.ipynb` — update `supported_shader_model` reference in output
- `docs/generated/api.rst` — will auto-regenerate
- Any other samples referencing `ShaderModel` or `shader_model`

## Phase 4: Profile Auto-Selection Logic

### Step 9: Implement `default_profile()` on Device
The logic per device type:
- **D3D12**: Return highest `sm_6_X` from `supported_profiles()` (with SM 6.7→6.6 workaround)
- **Vulkan**: Return highest `sm_6_X` (Slang accepts sm profiles for SPIRV targets too) OR consider using `spirv_1_X` — needs decision
- **Metal**: Return highest `metallib_X_X` or a reasonable default
- **CUDA**: Return highest `cuda_sm_X_X` or leave empty (Slang handles it)
- **CPU/WGPU**: Return empty string (no profile needed)

**Decision needed:** For Vulkan, should `default_profile()` return `"sm_6_6"` (D3D12-style, current behavior) or a SPIRV-native profile like `"spirv_1_5"`? Current code already passes `sm_X_X` profiles for Vulkan via `findProfile()`, so keeping `sm_X_X` preserves behavior.

## Relevant Files

- `src/sgl/device/types.h` — delete `ShaderModel` enum + helpers
- `src/sgl/device/device.h` — replace `m_supported_shader_model` with `m_supported_profiles`, add `supported_profiles()`, `has_profile()`, `default_profile()`
- `src/sgl/device/device.cpp` — rewrite profile detection from RHI capabilities, update `to_string()`
- `src/sgl/device/shader.h` — change `SlangCompilerOptions::shader_model` → `::profile` (string)
- `src/sgl/device/shader.cpp` — rewrite `create_session()` profile selection to use string, remove SM version helpers
- `src/slangpy_ext/device/types.cpp` — remove `ShaderModel` binding
- `src/slangpy_ext/device/shader.cpp` — update `SGL_DICT_TO_DESC_FIELD` and `.def_rw`
- `src/slangpy_ext/device/device.cpp` — replace `supported_shader_model` property
- `src/slangpy_ext/py_doc.h` — remove ShaderModel doc entries
- `slangpy/__init__.pyi` — update stubs
- `slangpy/testing/helpers.py` — update `dispatch_compute()` signature
- `slangpy/tests/device/slang/test_float16.py` — update parametrize + types
- `slangpy/tests/device/slang/test_float64.py` — update parametrize + types
- `slangpy/tests/device/slang/test_uint64.py` — update parametrize + types

## Verification

1. Build: `cmake --build --preset windows-msvc-debug`
2. Run Python tests: `pytest slangpy/tests -v`
3. Run specific affected tests: `pytest slangpy/tests/device/slang/test_float16.py slangpy/tests/device/slang/test_float64.py slangpy/tests/device/slang/test_uint64.py -v`
4. Run pre-commit: `pre-commit run --all-files`
5. Verify `device.supported_profiles` returns correct profiles for D3D12 and Vulkan devices
6. Verify `device.default_profile()` selects appropriate profile per backend
7. Verify `SlangCompilerOptions(profile="sm_6_6")` works correctly
8. Verify empty profile auto-selects correctly
9. Verify invalid profile string raises appropriate error

## Decisions
- **Clean break**: `ShaderModel` enum removed entirely, no deprecated aliases
- **String-based profiles**: matching Slang's `findProfile()` input format
- **Profile validation**: SGL validates profile against `supported_profiles()` before passing to Slang
- **Auto-select**: empty profile string triggers auto-selection of highest suitable profile
- **SM 6.7 workaround**: kept as temporary logic, translated to profile-string comparison
- **`__SHADER_TARGET_MAJOR`/`__SHADER_TARGET_MINOR` defines**: can be removed (no `.slang` files in the repo use them), or kept by parsing the profile string

## Further Considerations
1. **Vulkan default profile**: For Vulkan, should `default_profile()` return `"sm_6_X"` (preserving current behavior) or a native SPIRV profile? Recommendation: keep `"sm_6_X"` for now since that's what Slang uses and current code already does this.
2. **Profile ordering**: For `supported_profiles()`, we need a defined ordering to determine "highest". For SM profiles this is natural; for mixed profile types (spirv + sm), ordering is less clear. Recommendation: store profiles grouped by type in the order they're discovered, and have `default_profile()` implement per-backend logic.
3. **`__SHADER_TARGET_MAJOR`/`__SHADER_TARGET_MINOR` macro defines**: These are currently injected into every Slang session but no shader code in the repo uses them. They could be removed to simplify, or kept for user convenience. Recommendation: remove them since they're D3D12-centric and no code depends on them.
