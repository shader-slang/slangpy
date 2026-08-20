# Extend weak references to low-level reflection caches

This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

SlangPy now has a native `weak_ref<T>` primitive and uses it for high-level reflection `Layout` caches, but two older reflection caches still retain their values strongly and the low-level wrapper registry still observes objects through raw pointers. After this work, a `SlangModule` or `Device` cache will preserve wrapper identity only while callers keep the wrapper alive, and the global low-level reflection registry will safely lock live wrappers rather than reconstructing strong references from unverified raw pointers.

The work is split into two commits. The first changes the module and device layout caches. The second changes the global wrapper registry and synchronizes concurrent access. Existing required ownership links, especially `BaseReflectionObject::m_owner`, remain strong so Slang reflection pointers keep their originating component alive.

## Progress

- [x] (2026-08-20) Scanned project-owned native caches and observer relationships for weak-reference candidates.
- [x] (2026-08-20) Chose the module layout cache, device built-in layout cache, and global low-level reflection registry as the first follow-up migrations.
- [x] (2026-08-20) Converted module and device layout caches to `weak_ref` and added ownership/identity regressions.
- [x] (2026-08-20) Built, tested, and formatted the independently committable cache conversion milestone.
- [ ] Convert the global reflection registry to synchronized weak entries and add registry regressions.
- [ ] Build, test, format, and commit the registry conversion independently.

## Surprises and Discoveries

- Observation: `SlangModule::m_cached_layout` and the cached `ProgramLayout` directly own each other.
  Evidence: `SlangModule::layout()` stores a strong `ref<const ProgramLayout>`, while `ProgramLayout::from_slang()` stores the module as the wrapper's strong reflection owner.

- Observation: the built-in high-level layout participates in a longer device ownership loop.
  Evidence: `Device` owns `refl::Layout`, which owns a low-level `ProgramLayout`, whose module owner reaches the module's session and then the device.

- Observation: the global low-level reflection registry reconstructs `ref<T>` directly from cached raw pointers.
  Evidence: `src/sgl/device/reflection.cpp::create_reflection_type_from_slang_type()` stores `const BaseReflectionObject*` and constructs a new strong reference from it on cache hits without synchronization.

## Decision Log

- Decision: Keep cache identity while a wrapper is externally alive, but allow it to expire when only the cache observes it.
  Rationale: This matches the high-level reflection cache behavior and removes ownership loops without disabling useful identity caching.
  Date/Author: 2026-08-20, Codex and user.

- Decision: Keep `BaseReflectionObject::m_owner` strong.
  Rationale: That member is required ownership, not a cache: it preserves the Slang component that owns the raw reflection pointer.
  Date/Author: 2026-08-20, Codex.

- Decision: Store both a raw identity pointer and a native weak reference in each global registry entry.
  Rationale: The weak reference makes promotion safe. The raw value is used only for identity when a wrapper destructor unregisters, preventing an older destructor from erasing a replacement entry for a reused Slang pointer.
  Date/Author: 2026-08-20, Codex.

## Outcomes and Retrospective

The first milestone removes both remaining strong layout-cache edges while preserving live cache identity and hot-reload behavior. Validation passed for the full MSVC debug build, 9 object cases with 2,724 assertions, 3 focused cache/hot-reload cases with 50 assertions, 17 lifetime cases, 100 device reflection cases with 11 expected skips, and 150 high-level reflection cases. The global registry milestone remains pending.

## Context and Orientation

`src/sgl/device/shader.h` defines `SlangModule`. Its `m_cached_layout` stores the low-level `ProgramLayout` returned by `SlangModule::layout()` in `src/sgl/device/shader.cpp`. Every low-level reflection wrapper derives from `BaseReflectionObject` in `src/sgl/device/reflection.h`; that base strongly owns the module, entry point, shader program, or shader object whose Slang component produced the reflection pointer.

`src/sgl/device/device.h` stores `Device::m_builtin_layout`, the high-level `refl::Layout` returned by `Device::builtin_layout()` in `src/sgl/device/device.cpp`. Hot reload updates a live cached layout in place so external wrappers retain identity and observe a generation increment.

`src/sgl/device/reflection.cpp` has one process-wide map from raw Slang reflection pointers to SGL wrappers. Wrapper destruction erases entries, while device close and hot reload enumerate the map and invalidate matching live wrappers. A weak registry must therefore support identity-preserving lookup, expired-entry replacement, conditional destructor unregistering, and safe enumeration.

## Plan of Work

First, change `SlangModule::m_cached_layout` and `Device::m_builtin_layout` to `weak_ref`. Update their getters to lock before returning a cache hit. Preserve explicit reset when underlying reflection becomes stale during module rebuild or device close. Update built-in hot reload to lock the wrapper and mutate it only while a temporary strong reference is held. Add native tests proving repeated live lookup returns the same wrapper and releasing the last caller-owned reference expires the cache value.

Build and run the focused object, shader, lookup, reflection, and Python lifetime/reflection tests, then run pre-commit. Commit only the cache conversion, its tests, and this plan's first-milestone state.

Second, replace the raw global reflection map with entries containing a weak wrapper and raw identity pointer. Guard lookup, unregistering, and invalidation collection with a mutex. Cache lookup locks the weak entry, erases expired entries, and constructs exactly one replacement while serialized. Destructor unregistering erases only the entry whose identity pointer matches the wrapper being destroyed. Invalidation locks live wrappers into a temporary vector under the registry mutex, erases selected entries, releases the mutex, and then invalidates wrappers.

Add native tests for live identity, expiration and recreation, and concurrent lookup of one raw Slang reflection pointer. Rebuild, run focused native and Python reflection/hot-reload tests, run pre-commit, update this plan, and commit only the registry milestone.

## Concrete Steps

Run all commands from `C:\src\slangpy`. Per `AGENTS.md`, builds and tests run outside the sandbox.

    cmake --build --preset windows-msvc-debug
    build\windows-msvc\Debug\sgl_tests.exe --test-suite=object
    build\windows-msvc\Debug\sgl_tests.exe --test-suite=device
    build\windows-msvc\Debug\sgl_tests.exe --test-suite=refl
    pytest slangpy/tests/device/test_lifetimes.py -q
    pytest slangpy/tests/device/test_reflection.py -q
    pytest slangpy/tests/slangpy_tests/test_reflection2.py -q
    pre-commit run --all-files

Use narrower doctest filters where practical, but retain the lifetime and hot-reload cases that exercise the ownership graph. Build again if formatting changes native source.

## Validation and Acceptance

For both module and built-in layouts, two lookups while a caller holds a strong reference return the identical pointer. After every caller-owned strong reference is released, a test `weak_ref` reports expiration, proving the cache does not own the wrapper. A later lookup succeeds by recreating it. Built-in hot reload continues to update a live wrapper and increment its generation.

For the registry, concurrent lookup of the same Slang pointer returns one shared live wrapper, expired wrappers are recreated, and hot reload invalidates existing wrappers without data races in registry bookkeeping. Existing Python reflection and device lifetime suites continue to pass.

## Idempotence and Recovery

All changes are source edits and tests that can be rebuilt repeatedly. The dirty `external/slang-rhi` submodule is unrelated and must not be staged, modified, or cleaned. Each commit is independently buildable; if the registry milestone exposes an issue, retain the already committed cache migration while correcting the registry separately.

## Artifacts and Notes

First milestone validation:

    cmake --build --preset windows-msvc-debug
    Exit code: 0

    build\windows-msvc\Debug\sgl_tests.exe --test-suite=object
    9 cases passed; 2,724 assertions passed.

    build\windows-msvc\Debug\sgl_tests.exe --test-case="module layout cache is non-owning,builtin lookup layout cache is non-owning,builtin lookup layout"
    3 cases passed; 50 assertions passed.

    pytest slangpy/tests/device/test_lifetimes.py -q
    17 passed.

    pytest slangpy/tests/device/test_reflection.py -q
    100 passed, 11 skipped.

    pytest slangpy/tests/slangpy_tests/test_reflection2.py -q
    150 passed.

Final commit identifiers will be recorded as the milestones complete.

## Interfaces and Dependencies

No third-party dependencies are added. The cache members become `weak_ref<const ProgramLayout>` and `weak_ref<refl::Layout>`. The global registry remains internal to `src/sgl/device/reflection.cpp` and uses standard-library mutex synchronization. The public low-level reflection owner contract remains unchanged.
