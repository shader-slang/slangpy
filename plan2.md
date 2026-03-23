# Plan: Modernize SlangComponentType Wrappers and Hot Reload

## TL;DR

Two tasks: (1) Identify old code paths that directly call raw `slang::IComponentType`/`slang::ISession` APIs instead of using the new `SlangComponentType`/`SlangSession` wrappers, and refactor them. (2) Extend the hot reload system to support the new `SlangComponentType`-based APIs (type conformances, composites, specialized/linked component types) so they survive reload like modules and programs do.

---

## Part 1: Consolidate Raw Slang API Calls Through Wrappers

### Current State

The `SlangComponentType` class already wraps many `slang::IComponentType` operations:
- `specialize()` → wraps `IComponentType::specialize()`
- `link()` → wraps `IComponentType::link()`
- `link_with_options()` → wraps `IComponentType::linkWithOptions()`
- `layout()` → wraps `IComponentType::getLayout()`

`SlangSession` wraps session-level operations:
- `create_type_conformance()` → wraps `ISession::createTypeConformanceComponentType()`
- `create_composite_component_type()` → wraps `ISession::createCompositeComponentType()`
- `load_module()`, `load_module_from_source()`, etc.

### Sites Still Using Raw Slang APIs

#### A. `SlangEntryPoint::init()` (shader.cpp ~L1168-L1326)

This method does significant work directly on raw Slang APIs:

1. **Type conformance creation** (L1239): Calls `slang_session->createTypeConformanceComponentType()` directly instead of going through `SlangSession::create_type_conformance()`
2. **Composite creation** (L1260): Calls `slang_session->createCompositeComponentType()` directly instead of `SlangSession::create_composite_component_type()`
3. **Specialization** (L1304): Calls `data->slang_entry_point->specialize()` directly instead of `SlangComponentType::specialize()`
4. **Type resolution via layout** (L1236, L1282): Calls `slang_module->getLayout()` and `layout->findTypeByName()` directly
5. **Entry point discovery** (L1180): Calls `slang_module->findEntryPointByName()` directly
6. **Entry point layout query** (L1316): Calls `data->slang_entry_point->getLayout()->getEntryPointByIndex()` directly

#### B. `ShaderProgram::link()` (shader.cpp ~L1444-1554)

1. **Composite creation** (L1465): Calls `session->createCompositeComponentType()` directly
2. **Link with options** (L1513): Calls `composed_program->linkWithOptions()` directly

#### C. `SlangModule::load()` (shader.cpp ~L1004-1062)

1. **Module loading** (L1015, L1028): Calls `slang_session->loadModule()` and `slang_session->loadModuleFromSourceString()` directly — these are internal operations with no equivalent wrapper method (the wrapper IS `SlangSession::load_module`)

#### D. `HotReload::update_watched_paths_for_session()` (hot_reload.cpp ~L106-L116)

1. **Module enumeration** (L106): Calls `slang_session->getLoadedModuleCount()` and `getLoadedModule()` directly
2. **Dependency tracking** (L110): Calls `slang_module->getDependencyFileCount()` and `getDependencyFilePath()` directly

#### E. `SlangSession::create_type_conformance()` (shader.cpp ~L633-677)

1. **Type lookup** (L643-651): Iterates registered modules calling `slang_module->getLayout()->findTypeByName()` directly

#### F. `SlangComponentType::specialize()` (shader.cpp ~L881-923)

1. **Type lookup** (L883-908): Calls `m_component_type->getLayout()` and `layout->findTypeByName()` directly — this is the wrapper implementation itself, so some raw access is expected.

### Proposed Changes for Part 1

**Steps:**

1. **Refactor `SlangEntryPoint::init()`** to use `SlangSession` wrapper methods for type conformance and composite creation where possible. This is the biggest win — the init method duplicates logic already in `SlangSession::create_type_conformance()` and `create_composite_component_type()`.
   - Challenge: `init()` operates on `SlangSessionBuild` data (pre-commit build), not the live session. The wrapper methods operate on the live session. Need to either:
     - (a) Add build-aware variants of these methods, or
     - (b) Accept that build-phase code necessarily works at a lower level
   - **Recommendation**: Option (b) — the build phase is internal implementation detail. The duplication is acceptable because the build operates on temporary Slang objects that haven't been stored yet. Wrapping them would add complexity for no user-facing benefit.

2. **Refactor `ShaderProgram::link()`** — same situation as above: build-phase code. Keep raw API usage since it operates on build-temporary data.

3. **Add module-level helper methods** to `SlangModule` for operations that are currently raw:
   - `find_type_by_name(name)` → wraps `slang_module->getLayout()->findTypeByName()`
   - Consider whether entry point discovery (`findEntryPointByName`) needs a cleaner wrapper (it somewhat exists via `module.entry_point(name)`)

4. **Refactor `HotReload::update_watched_paths_for_session()`** to go through `SlangSession`/`SlangModule` wrappers instead of directly querying `slang::ISession` for loaded modules and dependencies. This requires adding:
   - `SlangSession::loaded_modules()` or exposing dependency info through the module wrapper
   - `SlangModule::dependency_file_paths()` → wraps `slang_module->getDependencyFileCount()/getDependencyFilePath()`

5. **Add `SlangSession::find_type_by_name()`** to centralize the type lookup pattern used in `create_type_conformance()` and `SlangComponentType::specialize()`.

---

## Part 2: Hot Reload Support for New SlangComponentType APIs

### Current Hot Reload Architecture

**What survives hot reload (registered with session/modules):**
- `SlangSession` — container persists, internal `SlangSessionData` rebuilt
- `SlangModule` — registered in `SlangSession::m_registered_modules`, rebuilt via `load()`/`store_built_data()`
- `SlangEntryPoint` — registered in `SlangModule::m_registered_entry_points`, rebuilt via `init()`/`store_built_data()`
- `ShaderProgram` — registered in `SlangSession::m_registered_programs`, rebuilt via `link()`/`store_built_data()`
- `Pipeline` (compute/raytracing) — registered in `ShaderProgram::m_registered_pipelines`, notified via `notify_program_reloaded()`

**What does NOT survive hot reload:**
- `SlangComponentType` instances from `specialize()`, `link()`, `link_with_options()` — become stale (dangling `slang::IComponentType*`)
- `SlangTypeConformance` instances from `create_type_conformance()` — become stale
- Composite `SlangComponentType` instances from `create_composite_component_type()` — become stale
- All reflection objects — explicitly invalidated

### The Problem

Users can create `SlangTypeConformance`, specialized `SlangComponentType`, and composite `SlangComponentType` objects through the public API. These hold raw `slang::IComponentType*` pointers that become invalid after hot reload. Unlike modules and programs, there is no registration/rebuild mechanism for these objects.

### Proposed Design

**Core Principle**: Follow the existing pattern — store enough descriptor data to recreate the object, register with session, rebuild during `recreate_session()`.

**Steps:**

1. **Add `SlangTypeConformance` registration to `SlangSession`**
   - Add `std::set<SlangTypeConformance*> m_registered_type_conformances` to `SlangSession`
   - Register/unregister in `SlangTypeConformance` constructor/destructor
   - `SlangTypeConformance` already stores its `TypeConformance` descriptor (interface_name, type_name, id) — enough to rebuild
   - Add `SlangTypeConformance::rebuild(SlangSessionBuild&)` method
   - Add `SlangTypeConformance::store_built_data(SlangSessionBuild&)` method
   - Update `SlangSessionBuild` to include type conformance build data
   - Update `recreate_session()` to rebuild type conformances after modules

2. **Add "standalone" `SlangComponentType` tracking**
   - For specialized/linked/composite component types returned to users, these are harder because they don't have descriptors that can be replayed.
   - Options:
     - **(a) Record creation recipe**: Store the operation (specialize/link/compose) and arguments as a descriptor. On reload, replay the recipe. This is complex because compositions can be nested.
     - **(b) Invalidation pattern**: Follow the reflection model — mark these objects as invalid after reload. Users query `is_valid()` and recreate as needed.
     - **(c) Callback pattern**: Register a rebuild callback with the session. On reload, the callback is called and the user provides the new component type.
   - **Recommendation**: Option (b) for generic `SlangComponentType` instances. These are typically intermediate objects used during shader compilation, not long-lived. Adding `is_valid()` (similar to reflection objects) is sufficient.
   - **Alternative for common cases**: If specialize/link/compose are used in common patterns that should auto-rebuild (e.g., type conformances composed with entry points), handle those at the SlangEntryPoint/ShaderProgram level where rebuild logic already exists.

3. **Update `SlangSessionBuild` struct**
   ```
   struct SlangSessionBuild {
       ref<SlangSessionData> session;
       std::map<const SlangModule*, ref<SlangModuleData>> modules;
       std::map<const ShaderProgram*, ref<ShaderProgramData>> programs;
       std::map<const SlangEntryPoint*, ref<SlangEntryPointData>> entry_points;
       // NEW:
       std::map<const SlangTypeConformance*, Slang::ComPtr<slang::IComponentType>> type_conformances;
   };
   ```

4. **Update `recreate_session()` rebuild order**
   ```
   create_session(build);
   // Rebuild modules (and their entry points)
   for (auto module : m_registered_modules) module->load(build);
   // NEW: Rebuild type conformances (need modules loaded for type lookup)
   for (auto tc : m_registered_type_conformances) tc->rebuild(build);
   // Rebuild programs (need modules + entry points + type conformances)
   for (auto program : m_registered_programs) program->link(build);
   // Store everything on success
   ...
   ```

5. **Add invalidation to standalone `SlangComponentType`**
   - Add `bool m_valid{true}` to `SlangComponentType`
   - Add `is_valid()` method
   - In `SlangSession::recreate_session()`, invalidate all "unregistered" component types (those not tracked as module/entrypoint/program/conformance)
   - Alternatively: track all created `SlangComponentType` instances via weak references and invalidate on reload
   - **Simpler approach**: Since `SlangComponentType` holds a `breakable_ref<SlangSession>`, and standalone instances are created via session methods that return new objects — just document that these don't survive hot reload, and add `is_valid()` that checks if the underlying `slang::IComponentType` is still usable.

6. **Add Python-side `is_valid` property** to `SlangComponentType` binding

7. **Add hot reload tests** in `slangpy/tests/device/`:
   - Test that `SlangTypeConformance` survives hot reload
   - Test that standalone `SlangComponentType` from specialize/link/compose is properly invalidated
   - Test that programs using type conformances survive hot reload

### Relevant Files to Modify

- `src/sgl/device/shader.h` — Add registration, rebuild methods, `is_valid()`, update `SlangSessionBuild`
- `src/sgl/device/shader.cpp` — Implement rebuild logic for type conformances, invalidation for standalone component types, refactor raw API calls
- `src/sgl/device/hot_reload.cpp` — Refactor `update_watched_paths_for_session()` to use wrappers
- `src/sgl/device/hot_reload.h` — Minor: add any new wrapper types if needed
- `src/slangpy_ext/device/shader.cpp` — Add `is_valid` property binding, add `dependency_file_paths` binding
- `slangpy/tests/device/test_component_type.py` — Add hot reload tests for type conformances

---

## Verification

1. Build: `cmake --build --preset windows-msvc-debug`
2. Run existing tests: `pytest slangpy/tests/device/test_component_type.py -v`
3. Run new hot reload tests (once written)
4. Run full test suite: `pytest slangpy/tests -v`
5. Run pre-commit: `pre-commit run --all-files`

## Decisions

- Build-phase code (inside `SlangEntryPoint::init()`, `ShaderProgram::link()`, `SlangModule::load()`) will continue using raw Slang APIs since it operates on temporary build objects. Only user-facing and cross-module code should go through wrappers.
- `SlangTypeConformance` gets full hot reload support (registration + rebuild).
- Standalone `SlangComponentType` (from specialize/link/compose) gets invalidation-only support (`is_valid()`), not auto-rebuild, matching the reflection object pattern.
- `HotReload::update_watched_paths_for_session()` should be refactored to use wrapper methods via new `SlangModule::dependency_file_paths()` API.

## Further Considerations

1. **Should `ShaderProgram` also store `SlangTypeConformance` refs?** Currently programs built with type conformances bake them into the entry point during `SlangEntryPoint::init()`. If type conformances are separately tracked, should programs also reference them for rebuild ordering? Recommendation: No — entry points already store `TypeConformance` descriptors and rebuild them internally.

2. **Nested composition tracking**: If a user creates `composite = session.create_composite_component_type([conformance1, conformance2, entry_point])`, should the composite auto-rebuild? Recommendation: No — use invalidation pattern. Composites are typically intermediate objects in a shader compilation pipeline.

3. **`SlangComponentType::specialize()` result tracking**: The `specialize()` method on the base class creates a new `SlangComponentType`. Should these be tracked? Recommendation: No — `SlangEntryPoint::specialize()` already has its own tracking via module registration. The base-class `specialize()` is for advanced use cases and should use invalidation.
