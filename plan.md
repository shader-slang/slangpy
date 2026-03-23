# Plan: Refactor SGL Shader API to Align with Slang IComponentType

## TL;DR

Introduce a `SlangComponentType` base class in SGL that mirrors Slang's `IComponentType` interface (minimal subset: `layout`, `specialize`, `link`). Make `SlangModule`, `SlangEntryPoint`, and the new `SlangTypeConformance` inherit from it. Add `create_composite_component_type()` on `SlangSession`. Keep `ShaderProgram` separate (HAS-A, not IS-A) for backward compatibility. All existing APIs remain unchanged — new classes sit alongside them.

---

## Background: Current vs Target Mapping

### Current SGL ↔ Slang mapping:
| Slang Interface | SGL Class | Notes |
|---|---|---|
| `ISession` | `SlangSession` | Full wrapper |
| `IModule : IComponentType` | `SlangModule` | No base class, bespoke API |
| `IEntryPoint : IComponentType` | `SlangEntryPoint` | No base class, wraps `IComponentType*` |
| `ITypeConformance : IComponentType` | *(none)* | Created transiently inside `SlangEntryPoint::init()` |
| `IComponentType` (composite) | *(none)* | Hidden inside `ShaderProgram::link()` |
| `IComponentType` (linked) | `ShaderProgram` | Also holds `rhi::IShaderProgram` |

### Target mapping after refactor:
| Slang Interface | SGL Class | Inherits |
|---|---|---|
| `IComponentType` | **`SlangComponentType`** (new) | `Object` |
| `IModule` | `SlangModule` | `SlangComponentType` |
| `IEntryPoint` | `SlangEntryPoint` | `SlangComponentType` |
| `ITypeConformance` | **`SlangTypeConformance`** (new) | `SlangComponentType` |
| composite `IComponentType` | Return as `SlangComponentType` | from `SlangSession::create_composite_component_type()` |
| linked `IComponentType` | Return as `SlangComponentType` | from `SlangComponentType::link()` |
| linked + rhi program | `ShaderProgram` | `DeviceChild` (unchanged, HAS-A `SlangComponentType` internally) |

---

## Phase 1: C++ Core — New `SlangComponentType` Base Class

### Step 1.1: Define `SlangComponentType` in `src/sgl/device/shader.h`
- New class `SlangComponentType : public Object`
- Wraps `Slang::ComPtr<slang::IComponentType>`
- Holds a back-reference to `SlangSession*` (breakable to avoid ref cycles)
- Minimal public API:
  - `session()` → `SlangSession*`
  - `layout(target_index = 0)` → `ref<const ProgramLayout>` — calls `IComponentType::getLayout()`
  - `specialize(span<SpecializationArg>)` → `ref<SlangComponentType>` — calls `IComponentType::specialize()`, resolves string types via session
  - `link()` → `ref<SlangComponentType>` — calls `IComponentType::link()`
  - `link_with_options(SlangLinkOptions)` → `ref<SlangComponentType>` — calls `IComponentType::linkWithOptions()`
  - `specialization_param_count()` → `int`
  - `entry_point_count()` → `int` — reflects composed entry points
  - `slang_component_type()` → `slang::IComponentType*` — raw access

- Internal data:
  - `Slang::ComPtr<slang::IComponentType> m_component_type`
  - `breakable_ref<SlangSession> m_session`

### Step 1.2: Make `SlangModule` inherit from `SlangComponentType`
- Change base from `Object` to `SlangComponentType`
- `SlangModule` passes its `slang::IModule*` (which IS-A `slang::IComponentType*`) to the `SlangComponentType` base
- `SlangModule::layout()` can now optionally delegate to the base, or keep its current implementation (both call `getLayout()`)
- Existing `SlangModule` API unchanged — `entry_points()`, `entry_point()`, `has_entry_point()`, `module_decl()`, `slang_module()` all stay
- Add `slang_module()` as the typed accessor, keep `slang_component_type()` from the base for generic access

### Step 1.3: Make `SlangEntryPoint` inherit from `SlangComponentType`
- Change base from `Object` to `SlangComponentType`
- `SlangEntryPoint` passes its `slang::IComponentType*` to the base
- Existing API unchanged: `name()`, `stage()`, `layout()`, `rename()`, `with_name()`, `specialize()`, `module()`
- `SlangEntryPoint::specialize()` continues to return `ref<SlangEntryPoint>` (more specific), but the base `SlangComponentType::specialize()` returns `ref<SlangComponentType>` — these coexist via override/method name differentiation or by making the entry-point `specialize()` call the base and wrap the result

### Step 1.4: Add `SlangTypeConformance` class
- New class `SlangTypeConformance : public SlangComponentType`
- Wraps result of `ISession::createTypeConformanceComponentType()`
- Minimal API:
  - Inherits `specialize()`, `link()`, `layout()` from base
  - Constructor takes session + type + interface + optional id override
- Created via `SlangSession::create_type_conformance(type, interface, id_override)` (new factory method)
- The existing transient conformance logic in `SlangEntryPoint::init()` can remain as-is for backward compat

### Step 1.5: Add `create_composite_component_type()` to `SlangSession`
- New method: `ref<SlangComponentType> create_composite_component_type(span<ref<SlangComponentType>> components)`
- Calls `ISession::createCompositeComponentType()`
- Returns a `SlangComponentType` wrapping the composite
- This gives users manual control over component ordering and composition

### Step 1.6: Update forward declarations in `fwd.h`
- Remove stale forward declarations for `SlangComponentType` (repurpose) and `SlangGlobalScope` (remove if unused)
- Add forward declaration for `SlangTypeConformance`

### Step 1.7: Adapt the `SlangSessionBuild` system
- The build system uses `SlangModuleData`, `SlangEntryPointData`, `ShaderProgramData` — these can stay as-is
- For `SlangComponentType` base, the `m_component_type` pointer can be set during `store_built_data()`, populated from the data structs
- Hot-reload path: `SlangSession::recreate_session()` rebuilds modules and programs. The new base class doesn't add new rebuild complexity since it delegates to modules/entry-points which already participate in the build

### Step 1.8: Keep `ShaderProgram` unchanged (HAS-A approach)
- `ShaderProgram` continues to store `Slang::ComPtr<slang::IComponentType>` for the linked program in `ShaderProgramData`
- Optionally add a convenience accessor: `ref<SlangComponentType> linked_component_type()` that wraps the internal `linked_program` in a `SlangComponentType` for users who want to inspect it generically
- The existing `link_program()` API on `SlangSession` remains unchanged
- `ShaderProgramDesc` still takes `vector<ref<SlangModule>>` and `vector<ref<SlangEntryPoint>>` — but could optionally be extended to also accept `vector<ref<SlangComponentType>>` for forward flexibility

---

## Phase 2: Nanobind Python Bindings

### Step 2.1: Bind `SlangComponentType` in `src/slangpy_ext/device/shader.cpp`
- New Python class `SlangComponentType`
- Expose: `session`, `layout`, `specialize`, `link`, `link_with_options`, `specialization_param_count`, `entry_point_count`
- Use nanobind inheritance so `SlangModule`, `SlangEntryPoint`, `SlangTypeConformance` inherit from it

### Step 2.2: Update `SlangModule` binding
- Change base class from `Object` to `SlangComponentType` in nanobind registration
- All existing properties/methods remain bound

### Step 2.3: Update `SlangEntryPoint` binding
- Change base class from `Object` to `SlangComponentType` in nanobind registration
- All existing properties/methods remain bound

### Step 2.4: Bind `SlangTypeConformance`
- New Python class inheriting from `SlangComponentType`
- Bind appropriate constructors

### Step 2.5: Bind new `SlangSession` methods
- `create_composite_component_type(components)` → `SlangComponentType`
- `create_type_conformance(type_name, interface_name, id_override)` → `SlangTypeConformance`

---

## Phase 3: Tests

### Step 3.1: New test file `slangpy/tests/test_component_type.py`
- Test `SlangComponentType.layout()` returns valid `ProgramLayout`
- Test `SlangComponentType.specialize()` with generic parameters
- Test `SlangComponentType.link()` produces a linked component type
- Test `SlangSession.create_composite_component_type()` with module + entry point
- Test `SlangTypeConformance` creation and composition
- Test that `isinstance(module, SlangComponentType)` is `True`
- Test that `isinstance(entry_point, SlangComponentType)` is `True`

### Step 3.2: Validate existing tests pass
- Run full test suite: `pytest slangpy/tests -v` to confirm no regressions
- The refactor should be purely additive — no existing API changes

---

## Phase 4: Pre-commit & Cleanup

### Step 4.1: Run pre-commit
- `pre-commit run --all-files`
- Fix any formatting issues

---

## Relevant Files

### New files:
- (None — all changes in existing files, unless `SlangComponentType` warrants its own header, which is unlikely given the current pattern of putting all shader classes in `shader.h`)

### Modified files:
- `src/sgl/device/shader.h` — Add `SlangComponentType` class, `SlangTypeConformance` class, change inheritance of `SlangModule` and `SlangEntryPoint`, add new `SlangSession` methods
- `src/sgl/device/shader.cpp` — Implement `SlangComponentType` methods, `SlangTypeConformance`, `create_composite_component_type()`, `create_type_conformance()`, adapt `SlangModule`/`SlangEntryPoint` constructors to init base
- `src/sgl/device/fwd.h` — Update forward declarations (repurpose `SlangComponentType`, remove `SlangGlobalScope`, add `SlangTypeConformance`)
- `src/slangpy_ext/device/shader.cpp` — Add nanobind bindings for new classes, update inheritance for existing bindings
- `slangpy/__init__.py` — Export new types
- `slangpy/__init__.pyi` — Add stubs for new types
- `slangpy/tests/` — New test file for component type API

### Reference files (patterns to follow):
- `src/sgl/device/shader.h` lines 370-436 (`SlangModule`) — follow this pattern for new classes
- `src/sgl/device/shader.cpp` lines 1022-1060 (`SlangEntryPoint::init` with type conformance) — reference for `SlangTypeConformance` creation
- `src/sgl/device/shader.cpp` lines 1214-1330 (`ShaderProgram::link`) — reference for `createCompositeComponentType` + `linkWithOptions` calls
- `src/slangpy_ext/device/shader.cpp` lines 233-265 (existing nanobind class bindings) — pattern for new bindings

---

## Verification

1. Build: `cmake --build --preset windows-msvc-debug`
2. Run existing tests: `pytest slangpy/tests -v` — expect zero regressions
3. Run new tests: `pytest slangpy/tests/test_component_type.py -v`
4. Run pre-commit: `pre-commit run --all-files`
5. Manual smoke test: Create a composite component type from Python, specialize it, link it, and verify layout access

---

## Decisions

- **Minimal API**: `SlangComponentType` exposes `layout`, `specialize`, `link`, `link_with_options`, `specialization_param_count`, `entry_point_count` only. Code retrieval methods (`getEntryPointCode`, `getTargetCode`) are NOT exposed initially — they can be added later if needed.
- **SlangTypeConformance**: First-class wrapper, created via `SlangSession::create_type_conformance()`. Existing transient logic in `SlangEntryPoint::init()` remains for backward compat.
- **Composite creation**: `SlangSession::create_composite_component_type()` exposed to give users manual composition control.
- **ShaderProgram**: Unchanged externally (HAS-A). Optionally gains `linked_component_type()` accessor.
- **Backward compatibility**: All existing Python/C++ APIs preserved. New classes are purely additive.
- **Hot reload**: `SlangComponentType` instances returned from `specialize()`, `link()`, `create_composite_component_type()` are ephemeral/unmanaged by the session — they don't participate in hot-reload rebuilds. Only `SlangModule` and `ShaderProgram` (which are session-registered) participate in hot-reload.

## Further Considerations

1. **Entry point specialize return type**: `SlangEntryPoint::specialize()` currently returns `ref<SlangEntryPoint>`. The base `SlangComponentType::specialize()` would return `ref<SlangComponentType>`. The entry-point override can wrap the result back in a `SlangEntryPoint` (current behavior) while the generic base version returns the generic type. Alternatively, only one version exists and callers downcast. Recommendation: keep the specific `SlangEntryPoint::specialize()` and have it call through to base internally.

2. **`SlangComponentType` as abstract vs concrete**: It should be **concrete** (not abstract) since composites and linked results are plain `IComponentType*` with no further subclass-specific behavior. Only `SlangModule`, `SlangEntryPoint`, `SlangTypeConformance` add specialized methods.

3. **Thread safety**: `SlangComponentType` wrapping COM pointers follows the same thread-safety model as the existing wrappers — no additional synchronization needed. The Slang session is not thread-safe for mutation, so `specialize()` and `link()` follow the same constraints.
