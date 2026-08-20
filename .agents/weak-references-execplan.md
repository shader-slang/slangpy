# Add native weak references to Object and use them in reflection caches

This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

SlangPy reflection objects currently form an ownership cycle: `sgl::refl::Layout` caches `Type` and `Function` objects, while each cached object strongly owns its `Layout`. The temporary fix in the working tree gives each cached object a separate `shared_ptr` lifetime token, but that duplicates lifetime machinery and adds reflection-specific bookkeeping. After this change, every `sgl::Object` can be observed through a native `weak_ref<T>` without being kept alive. `Layout` can cache `Type` and `Function` using that facility, breaking the cycle while preserving cache identity for objects that are still alive.

The behavior is demonstrated by native object tests showing that a weak reference locks while its target lives, expires after the final strong reference, survives a native-to-Python ownership transition, and is safe when locking races with final release. The reflection test demonstrates that releasing a `Type` or `Function` returns the `Layout` reference count to its baseline and that a later lookup recreates the wrapper. A follow-up hardening milestone closes two Python-boundary races found during review: ownership transfer must not invert the GIL and weak-state mutex, and a wrapper whose Python reference count has reached zero must never be promoted during C++ destruction. A Linux Clang ThreadSanitizer configuration provides ongoing data-race coverage for the native protocol.

## Progress

- [x] (2026-08-17 14:08Z) Confirmed the design decisions: universal native weak-reference support, conventional GIL-enabled CPython, no blanket Python `weakref.ref` enablement, and reflection-only migration.
- [x] (2026-08-17 14:08Z) Inspected `Object`, nanobind intrusive ownership, reference tracking, reflection caches, and the existing tests.
- [x] (2026-08-17 14:24Z) Implemented the lazy auxiliary weak state, `weak_ref<T>` API, synchronized final release, native-to-Python transition marker, and GIL runner bridge.
- [x] (2026-08-17 14:27Z) Added native tests for API behavior, inheritance conversions, native-to-Python transition, and concurrent final release.
- [x] (2026-08-17 14:24Z) Replaced the temporary reflection lifetime tokens with `weak_ref<Type>` and `weak_ref<Function>`.
- [x] (2026-08-17 14:38Z) Built successfully, passed focused native and Python reflection tests, and passed pre-commit after its formatting pass.
- [x] (2026-08-17 15:30Z) Completed a multi-specialist review and reproduced the two unsafe interleavings in the ownership protocol by code inspection.
- [x] (2026-08-17 15:50Z) Fixed native weak promotion to reserve the observed native count directly instead of calling the representation-switching `Object::inc_ref()` while holding the weak-state mutex.
- [x] (2026-08-17 15:50Z) Rejected and expired Python weak promotion when the wrapper reference count is zero, before incrementing it.
- [x] (2026-08-17 15:55Z) Added destructor-reentrancy, concurrent native-to-Python transition, and concurrent weak-state initialization regression tests using one recursive fake-GIL lock.
- [x] (2026-08-17 16:00Z) Added `SGL_ENABLE_TSAN`, Linux Clang CI coverage, sanitizer environment support for `thread`, and a CMake/CI configuration that compiles reference tracking.
- [x] (2026-08-17 16:10Z) Rebuilt and passed the default and reference-tracking object suites, focused native and Python reflection tests, sanitizer configuration checks, and pre-commit. The Linux TSan runtime execution remains delegated to the new CI job because the local host is Windows.
- [x] (2026-08-20) Removed the now-obsolete `BaseModule` destructor and `Layout::clear_caches()` cycle-breaking path after the weak cache migration, then rebuilt and passed the lifetime, object, native reflection, and Python reflection suites.

## Surprises and Discoveries

- Observation: nanobind requires its intrusive `set_self_py` callback to be `noexcept`.
  Evidence: `external/nanobind/include/nanobind/nb_attr.h` stores a `void (*)(T*, PyObject*) noexcept` callback. Allocating a CPython weak-reference object during this callback would therefore have no clean allocation-failure path.

- Observation: `Object::dec_ref()` currently deletes directly after observing native state value `3`, without a compare/exchange.
  Evidence: `src/sgl/core/object.cpp` handles `value == 3` with `delete this`. Weak promotion racing with final release requires this path to synchronize and claim the final reference before deletion.

- Observation: the existing `ref<T>` optionally assigns a unique tracking identifier and records every strong acquisition when `SGL_ENABLE_REF_TRACKING` is enabled.
  Evidence: `src/sgl/core/object.h` initializes `m_ref_id` and calls `inc_ref_tracked()` and `dec_ref_tracked()`. A successful weak lock must use the same identifier for its acquired and released strong reference.

- Observation: `set_self_py()` previously loaded the native count and later replaced it with the Python pointer without claiming the state, so a concurrent strong increment could theoretically be lost even before weak references were introduced.
  Evidence: reserving state value zero and claiming the old state with compare/exchange provides a transition marker. `inc_ref()`, `dec_ref()`, `ref_count()`, and `self_py()` wait until that short transition completes.

- Observation: the Windows Clang build re-ran CMake and rebuilt the public-header dependency graph, but completed successfully.
  Evidence: the final build completed 360 Ninja edges and linked `sgl.dll`, the nanobind extension, examples, and `sgl_tests.exe` with exit code zero.

- Observation: native weak promotion can cross from the native representation into Python reference counting while it still owns `WeakState::m_mutex`.
  Evidence: `WeakState::lock()` reads an odd state and then calls generic `Object::inc_ref()`. A concurrent `set_self_py()` can replace that state with zero and later wait for the same mutex while holding the GIL, leaving the weak-lock thread waiting for the GIL while holding the mutex.

- Observation: nanobind calls the derived C++ destructor while the wrapper reference count is already zero, but `Object::~Object()` cannot expire the weak state until the derived destructor finishes.
  Evidence: `external/nanobind/src/nb_type.cpp::inst_dealloc()` calls the registered destructor after `Py_DECREF` has entered `tp_dealloc`. A reentrant `weak_ref::lock()` from a derived destructor therefore sees the Python mode until the base destructor runs. The existing reference-count callback can reject this state while the GIL is held without modifying nanobind.

- Observation: `Layout::clear_caches()` retained two pointer-style uses after the reflection maps changed from `ref<T>` to `weak_ref<T>`, exposing that the whole method had become redundant.
  Evidence: history showed that commit `f96958c3` added the method and the `BaseModule` destructor solely to break the former strong `Layout -> Type/Function -> Layout` cycle. The weak maps remove the first strong edge, and all 17 lifetime regression tests pass after deleting that cleanup path.

## Decision Log

- Decision: Add one auxiliary-state pointer directly to every `Object`, rather than introducing a weak-referenceable subclass or inheritance decorator.
  Rationale: SlangPy and Falcor2 contain deep inheritance trees where leaf-only opt-in would be awkward. The eight-byte cost on 64-bit systems is acceptable, while a universal base capability avoids hierarchy and binding bookkeeping.
  Date/Author: 2026-08-17, Codex and user.

- Decision: Allocate the auxiliary `WeakState` lazily, only when the first native weak reference is constructed.
  Rationale: Objects that are never weakly referenced pay only for the pointer in `Object`; they do not allocate a control block.
  Date/Author: 2026-08-17, Codex and user.

- Decision: Support native `weak_ref<T>` but do not make all Python wrappers publicly weak-referenceable through Python's `weakref.ref()`.
  Rationale: Native weak promotion can be synchronized using the existing GIL-enabled Python ownership model. Avoiding blanket `nb::is_weak_referenceable()` also avoids another pointer in every nanobind wrapper.
  Date/Author: 2026-08-17, Codex and user.

- Decision: Require conventional GIL-enabled CPython and do not claim free-threaded Python support in this first implementation.
  Rationale: The GIL makes promotion of the non-owning Python wrapper pointer atomic with respect to wrapper deallocation. Free-threaded CPython would require a different promotion mechanism, most likely an actual CPython weak reference.
  Date/Author: 2026-08-17, Codex and user.

- Decision: Construct weak references only from strong `ref<T>` instances or compatible `weak_ref<T>` instances, and expose `lock()` rather than pointer-like access.
  Rationale: A strong source guarantees that lazy control-block creation cannot race with target destruction. Requiring `lock()` makes the temporary ownership needed for safe access explicit.
  Date/Author: 2026-08-17, Codex and user.

- Decision: Migrate only the `sgl::refl::Layout` caches in this change.
  Rationale: `breakable_ref`, device caches, and Falcor2 observer relationships have distinct behavior and can be considered separately after the primitive is proven.
  Date/Author: 2026-08-17, Codex and user.

- Decision: Reserve `Object::m_state == 0` while native references are transferred to Python.
  Rationale: This lets ownership transfer atomically claim the native count. Other reference operations wait for the Python pointer to be published, preventing a concurrent increment or decrement from being applied to the obsolete native count.
  Date/Author: 2026-08-17, Codex.

- Decision: Preserve reference-tracking identity by letting `weak_ref<T>` prepare an empty `ref<T>`, pass its tracking identifier into weak promotion, and then adopt exactly that acquired reference.
  Rationale: Performing an ordinary `ref<T>` construction after promotion would increment twice, while an untracked adopt path would make leak diagnostics inaccurate.
  Date/Author: 2026-08-17, Codex.

- Decision: Reserve native promotion with a compare/exchange on the exact odd `m_state` value observed under the weak-state mutex.
  Rationale: The operation either acquires a native reference without touching Python or observes that ownership changed and retries after releasing the mutex. It therefore cannot acquire the GIL while holding `WeakState::m_mutex`.
  Date/Author: 2026-08-17, Codex.

- Decision: Treat a Python wrapper reference count of zero as expired inside `lock_python()` and never call `Py_INCREF` in that state.
  Rationale: Conventional CPython holds the GIL from the decrement-to-zero through `tp_dealloc`, so the zero test and promotion are atomic with wrapper deallocation. This closes destructor-time resurrection without requiring a nanobind fork or enabling a Python weak-reference slot on every wrapper.
  Date/Author: 2026-08-17, Codex.

- Decision: Add ThreadSanitizer as a separate Linux Clang configuration rather than combining it with AddressSanitizer.
  Rationale: AddressSanitizer and ThreadSanitizer runtimes are incompatible. A focused native object-suite job gives the ownership protocol race coverage without involving GPU dispatch or an uninstrumented CPython runtime.
  Date/Author: 2026-08-17, Codex.

- Decision: Remove `Layout::clear_caches()` and the custom `BaseModule` destructor after migrating the layout caches to `weak_ref`.
  Rationale: Those APIs existed only to break the former strong ownership cycle. Retaining them would discard live reflection caches when a module wrapper is released, despite the weak maps already allowing the layout and device graph to be reclaimed naturally. `Type::clear_caches()` and `Function::clear_caches()` remain because hot reload uses them to invalidate data derived from old low-level reflection objects.
  Date/Author: 2026-08-20, Codex and user.

## Outcomes and Retrospective

Implemented universal native weak references with one additional pointer in `Object` and a lazily allocated, independently reference-counted control block. Native final release and weak promotion synchronize only when a weak state exists. Python-owned promotion runs under the conventional CPython GIL and takes a normal Python reference, without enabling Python-level `weakref.ref()` for every binding.

The reflection caches now store `weak_ref<Type>` and `weak_ref<Function>` directly. The temporary raw-pointer plus `std::weak_ptr<void>` entries and per-object `shared_ptr` lifetime tokens were removed. The regression continues to show that cache hits preserve identity while live and do not keep the `Layout` cycle alive. The initial validation did not exercise the reviewed Python-boundary races; this section will be updated with the hardening and TSan results when the follow-up milestone completes.

The hardening milestone makes native promotion representation-specific: a successful compare/exchange reserves the strong count without touching Python, while a failed compare/exchange retries after releasing the weak-state mutex. Python promotion now checks the wrapper count while the GIL is held and marks a zero-count target expired, so derived-destructor reentrancy cannot resurrect an object already in `tp_dealloc`.

Validation completed successfully: the Windows Clang debug build passed; the default `object` native suite passed 8 cases and 2,723 assertions; the separate build with object and reference tracking enabled passed 9 cases and 2,726 assertions; the two focused native reflection cases passed 38 assertions; `slangpy/tests/slangpy_tests/test_reflection2.py` passed all 150 cases across D3D12, Vulkan, and CUDA; the sanitizer workflow YAML and invalid sanitizer-combination checks passed; and every pre-commit hook passed. The new Linux Clang TSan job runs the object suite with reference tracking enabled, but its runtime result is pending CI because TSan is not available on the Windows development host. The implementation intentionally does not claim free-threaded CPython support and does not migrate other potential weak-reference users such as `breakable_ref` or Falcor2 registries.

The follow-up MSVC validation also passes after removing the obsolete cleanup path entirely: the full `windows-msvc-debug` build linked, all 17 lifetime regressions passed, the object suite passed 9 cases and 2,724 assertions, the two native reflection cases passed 38 assertions, and the Python reflection suite passed all 150 cases.

## Context and Orientation

`src/sgl/core/object.h` defines `Object`, the intrusive strong-reference wrapper `ref<T>`, and the older `breakable_ref<T>`. Native ownership is encoded in `Object::m_state`: odd values encode a strong count, while even values are aligned `PyObject*` pointers. `src/sgl/core/object.cpp` implements the reference-count operations and the transition to Python ownership. `src/slangpy_ext/core/object.cpp` installs callbacks that perform Python reference-count operations while holding the Global Interpreter Lock, abbreviated GIL.

A weak reference is a non-owning handle that can attempt to create a temporary strong reference with `lock()`. The auxiliary `WeakState` must outlive the target object when weak references remain. It therefore has its own small intrusive count: one state reference belongs to the live `Object`, and one belongs to every `weak_ref`. The control block stores the live target pointer and whether ownership is native, Python, or expired. A mutex in the lazily allocated control block synchronizes weak locking with the final strong release. Normal strong increments and non-final decrements do not take this mutex.

When an object is Python-owned, `Object::m_state` contains the non-owning Python wrapper pointer. A weak lock must acquire the GIL before the control-block mutex, verify that the object remains live, and increment the ordinary Python reference count. Python wrapper deallocation already runs with the GIL and invokes the C++ destructor; the destructor takes the control-block mutex, marks it expired, and clears its target. This ordering prevents promotion from a Python reference count of zero. `object_init_py()` will gain a callback that runs a non-throwing C function while holding the GIL, keeping `sgl` itself independent of Python headers and libraries.

`src/sgl/refl/layout.h` and `src/sgl/refl/layout.cpp` hold four caches for high-level `Type` and `Function` wrappers. The working tree currently contains a temporary `WeakCacheEntry<T>` composed of a raw pointer and `std::weak_ptr<void>`. `src/sgl/refl/type.h` and `src/sgl/refl/function.h` own matching `shared_ptr` tokens. These temporary members must be removed and the maps changed directly to `weak_ref<Type>` and `weak_ref<Function>`.

`tests/sgl/core/test_object.cpp` is the native unit-test file for intrusive ownership. `tests/sgl/refl/test_reflection.cpp` already contains the new regression test proving the caches no longer own their values. Python reflection behavior is covered by `slangpy/tests/test_reflection2.py`.

The worktree also contains changes unrelated to this implementation: the `external/slang-rhi` submodule is dirty, and `slangpy/tests/core/test_testing_helpers.py` is an earlier leak-fix test. Preserve both.

## Plan of Work

First, update `src/sgl/core/fwd.h` and `src/sgl/core/object.h` with the `weak_ref<T>` declaration and implementation. Add an atomic pointer to an opaque auxiliary state to `Object`. A weak reference stores both an adjusted `T*`, needed for correct derived-to-base conversions, and the shared auxiliary-state pointer. Its copy operations retain the state, its destructor releases the state, and `lock()` asks the state to acquire one strong reference before adopting that already-acquired reference into a result `ref<T>`. Make `weak_ref<T>` a friend of `ref<T>` so the result can preserve reference-tracking identifiers without performing a second increment.

Second, implement the opaque state in `src/sgl/core/object.cpp`. Use a lazily initialized control block with an atomic auxiliary reference count, a mutex, a mode, and the live `Object*`. Change final native release to claim state value `3` with compare/exchange. If a weak state exists, claim and expire the target while holding its mutex before deleting. Make the `Object` destructor expire any remaining state and release the object's auxiliary reference. Extend `object_init_py()` with a callback that executes a non-throwing thunk under the GIL. The Python-owned weak-lock branch invokes this callback before taking the state mutex and acquiring an ordinary tracked or untracked strong reference.

Third, extend `src/slangpy_ext/core/object.cpp` to provide the GIL runner callback. Do not add `nb::is_weak_referenceable()` to `Object`; native weak references use the intrusive wrapper pointer already stored in `m_state`, and existing explicitly Python-weak-referenceable bindings remain unchanged.

Fourth, add focused tests to `tests/sgl/core/test_object.cpp`. Test empty weak references, copying and moving, compatible derived-to-base conversion, successful locking, expiration, and lazy lifetime behavior. Add a deterministic stress loop in which one thread drops the final strong reference while another attempts to lock. Add a fake Python-wrapper counter and fake GIL callback to exercise native-to-Python transfer, weak promotion through Python reference counting, wrapper destruction, and expiration without embedding a Python interpreter in the native test executable. Keep the callbacks installed for the remainder of the native process because they point to static test functions.

Fifth, replace `Layout::WeakCacheEntry<T>` with `weak_ref<T>`, remove the `<memory>` includes, friendship used only for the lifetime token, and `m_cache_lifetime` members. Keep the cache helper and hot-reload snapshots, changing cache insertion to construct weak references from the strong objects.

Finally, build before every test phase. Run the object and reflection native tests, the focused Python reflection suite, and pre-commit. If formatting changes files, rebuild and rerun the affected tests. Do not modify or clean unrelated worktree changes.

## Concrete Steps

All commands run from `C:\src\slangpy`. CMake builds and tests must run outside the sandbox per `AGENTS.md`.

Inspect the final diff:

    git diff -- src/sgl/core/fwd.h src/sgl/core/object.h src/sgl/core/object.cpp src/slangpy_ext/core/object.cpp src/sgl/refl/layout.h src/sgl/refl/layout.cpp src/sgl/refl/type.h src/sgl/refl/function.h tests/sgl/core/test_object.cpp tests/sgl/refl/test_reflection.cpp

Build the configured Windows debug tree:

    cmake --build --preset windows-clang-debug

If that preset is unavailable, inspect `cmake --list-presets=build` and use the already configured debug preset without reconfiguring or disturbing unrelated build trees.

Run focused native tests from the produced test executable, using its existing doctest filters for the object and reflection cases. Then run:

    pytest slangpy/tests/slangpy_tests/test_reflection2.py -v
    pre-commit run --all-files

Record exact executable paths, filters, assertion counts, pytest counts, and hook results in this plan.

## Validation and Acceptance

The native object suite shows that `weak_ref<T>::lock()` returns the original object while a strong owner exists and an empty `ref<T>` after destruction. Its 200-iteration concurrency test completes without crashes, underflows, or leaked live-object counters. Its fake-Python test shows that locking increases the fake Python count, releasing the locked `ref` decreases it, and releasing the simulated wrapper destroys the target and expires the weak reference.

The native reflection regression test named `native layout caches do not own reflection objects` must pass. While a cached `Type` or `Function` exists, repeat lookup must return the identical pointer. After its scope ends, `Layout::ref_count()` must return to baseline, proving there is no `Layout -> cached value -> Layout` ownership cycle. A later lookup must still succeed by recreating the wrapper.

`slangpy/tests/slangpy_tests/test_reflection2.py` passes on D3D12, Vulkan, and CUDA, demonstrating that cache identity, specialization, and hot reload behavior remain intact. Pre-commit finishes with every hook passing.

## Idempotence and Recovery

All edits are source changes applied with patches and can be rebuilt repeatedly. The lazy weak state has no persistent external data. If a build exposes an unsafe interaction between Python transition and final release, retain the reflection lifetime-token implementation while correcting the generic primitive; do not revert unrelated files. Never reset or clean the dirty `external/slang-rhi` submodule or the unrelated untracked Python test.

## Artifacts and Notes

The working tree began with a validated temporary reflection implementation: a raw cached pointer paired with `std::weak_ptr<void>`, with a `std::shared_ptr<void>` token in each `Type` and `Function`. That implementation is the behavioral reference but should disappear from the final diff.

Final validation evidence:

    cmake --build --preset windows-clang-debug
    Exit code: 0

    build\windows-clang\Debug\sgl_tests.exe --test-suite=object
    5 test cases passed; 665 assertions passed.

    build\windows-clang\Debug\sgl_tests.exe --test-case="native layout caches do not own reflection objects,native layout hot reload"
    2 test cases passed; 38 assertions passed.

    pytest slangpy/tests/slangpy_tests/test_reflection2.py -q
    150 passed in 35.26s.

    pre-commit run --all-files
    All hooks passed.

Hardening validation evidence:

    build\windows-clang\Debug\sgl_tests.exe --test-suite=object
    8 test cases passed; 2723 assertions passed.

    build\windows-clang-reftracking\Debug\sgl_tests.exe --test-suite=object
    9 test cases passed; 2726 assertions passed.

    build\windows-clang\Debug\sgl_tests.exe --test-case="native layout caches do not own reflection objects,native layout hot reload"
    2 test cases passed; 38 assertions passed.

    pytest slangpy/tests/slangpy_tests/test_reflection2.py -q
    150 passed in 33.74s.

    pre-commit run --all-files
    All hooks passed.

The intended public interface is:

    template<typename T>
    class weak_ref {
    public:
        weak_ref() noexcept;
        weak_ref(std::nullptr_t) noexcept;

        template<typename U>
        weak_ref(const ref<U>& strong);

        ref<T> lock() const noexcept;
        bool expired() const noexcept;
        void reset() noexcept;
    };

## Interfaces and Dependencies

`src/sgl/core/fwd.h` must declare `template<typename> class weak_ref;`.

`src/sgl/core/object.h` must expose `weak_ref<T>` alongside `ref<T>` and add exactly one auxiliary pointer-sized atomic member to ordinary non-tracking `Object`. The object remains Python-optional; the header may forward-declare Python and weak-state types but must not include Python headers.

`object_init_py()` must retain its existing increment, decrement, and count callbacks and add a non-throwing GIL-runner callback. `src/slangpy_ext/core/object.cpp` is the only production caller and will implement it with `nb::gil_scoped_acquire`.

No new third-party dependency is permitted. Use the standard library atomics and mutex already available to the project.

Revision note: Initial plan created after the user approved all three design-scope questions. It incorporates the existing temporary reflection-cache changes and preserves unrelated worktree state. Updated after implementation to record the transition marker, tracking-preserving promotion, final API, and validation evidence.
