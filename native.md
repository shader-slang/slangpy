# Native SlangPy Migration Plan

## Goals

Move SlangPy toward a native-first architecture:

- Make `Tensor` a native runtime type, with as much of its data model and runtime behavior as possible in pure `sgl`.
- Keep Python as an adapter layer for Python-only conveniences, not as the owner of core runtime semantics.
- Preserve a path to native-only SlangPy-like functionality for C++ users.
- Keep the work structured, testable, and shippable in small phases.

## Boundaries

The most important boundary is simple:

- New code under `src/sgl/refl` and `src/sgl/func` must not include `nanobind.h`, use `nb::object`, use `PyObject`, include Python headers, or call Python.
- Existing core SGL Python lifetime hooks are not a precedent for this migration; do not expand them as part of native reflection or native Tensor.
- Code under `src/slangpy_ext` owns nanobind bindings, Python object adaptation, NumPy/Torch conversion, and any Python registry fallback.
- Python files under `slangpy/` should become compatibility and ergonomic layers over native implementations.

The current low-level Slang reflection wrappers already live in `src/sgl/device/reflection.h`. The missing native piece is the higher-level semantic reflection model currently implemented in `slangpy/reflection/reflectiontypes.py`.

## Naming

Use native names that do not bake in Python:

```cpp
namespace sgl::refl {
class Layout;
class Type;
class Function;
class Field;
class Parameter;
}

namespace sgl::func {
class BaseModule;
class BaseStruct;
class Tensor;
struct TensorDesc;
}
```

Terminology:

| Name | Meaning |
|------|---------|
| `sgl::ProgramLayout` | Existing low-level Slang/RHI reflection wrapper. |
| `sgl::refl::Layout` | New semantic reflection layout that wraps a low-level layout and owns SlangPy-style type/function caches. |
| `sgl::func::BaseModule` / `sgl::func::BaseStruct` | Minimal functional-layer footholds that preserve owning module/layout identity without pulling Python behavior into core SGL. |
| Built-in SGL layout | A generic per-device layout for built-in SGL support types. It should not be named or implemented as a Python/SlangPy lookup module in core SGL. |
| Python compatibility names | Temporary names such as `SlangType` and `SlangProgramLayout` that bind to native reflection during the migration. |

Python can keep exporting names such as `SlangType` and `SlangProgramLayout` for compatibility while the migration is in progress. Internally, those should become bindings or thin wrappers around `sgl::refl::Type` and `sgl::refl::Layout`.

`NativeSlangType` should eventually disappear. It exists today because the semantic `SlangType` is Python-only and native marshalling needs to call back into Python for `shape`, `derivative`, and layouts. Once `sgl::refl::Type` owns those concepts natively, `NativeSlangType` should first become a compatibility alias/shim and then be removed.

The compatibility window is Phases 2-6. During that window, names such as `SlangType` can be deprecated aliases or bindings for native reflection types. In Phase 7, Python should use the same names as native (`Layout`, `Type`, `Tensor`, etc.) and the old compatibility names should be removed rather than kept indefinitely.

The initial Python binding modules `slangpy.native_refl` and `slangpy.native_func` are also temporary migration names. They make the native/Python boundary explicit while the existing Python API still owns names such as `Module.layout` and `Struct.name`. These native classes should not grow a parallel public `native_*` field surface; their data should replace the corresponding Python fields as each native implementation becomes complete.

## Progress So Far

Recorded on 2026-05-14.

Completed:

- Created the working branch `dev/ccummings/native-slangpy`.
- Added the first native semantic reflection foothold:
  - `src/sgl/refl/layout.h`
  - `src/sgl/refl/layout.cpp`
  - `sgl::refl::Layout` wraps the existing low-level `sgl::ProgramLayout`, tracks a generation id, and refreshes on hot reload.
- Added the first native functional footholds:
  - `src/sgl/func/base_module.h`
  - `src/sgl/func/base_module.cpp`
  - `src/sgl/func/base_struct.h`
  - `src/sgl/func/base_struct.cpp`
  - `sgl::func::BaseModule` stores the composed Slang module and native reflection layout.
  - `sgl::func::BaseStruct` stores the owning base module and the minimal reflected struct identity needed by future native lookup.
- Added bindings in the new binding folders:
  - `src/slangpy_ext/refl/layout.cpp` binds `sgl::refl::Layout`.
  - `src/slangpy_ext/func/functional.cpp` binds `sgl::func::BaseModule` and `sgl::func::BaseStruct`.
- Added transitional Python binding modules:
  - `slangpy.native_refl`
  - `slangpy.native_func`
- Updated Python layering:
  - `slangpy.core.module.Module` now inherits from `slangpy.native_func.BaseModule`.
  - `slangpy.core.struct.Struct` now inherits from `slangpy.native_func.BaseStruct`.
  - Existing Python fields such as `Module.layout`, `Struct.name`, `Struct.full_name`, and `Struct.shape` remain the active Python API until native reflection can replace them outright.
- Avoided exposing duplicate `native_*` fields on the Python base classes. The native data is present for inheritance and future lookup, not as a parallel public API.
- Added tests:
  - `tests/sgl/func/test_reflection.cpp`
  - `slangpy/tests/slangpy_tests/test_native_bridge.py`
- Confirmed the native `src/sgl/refl` and `src/sgl/func` additions remain Python-free with:

```powershell
rg -n "nanobind|nb::|PyObject|Python\.h|pybind" src/sgl/refl src/sgl/func
```

Verified:

```powershell
cmake --build --preset windows-msvc-debug --target slangpy_ext sgl_tests
build\windows-msvc\Debug\sgl_tests.exe -ts=func
python -m pytest slangpy/tests/slangpy_tests/test_native_bridge.py -v
python -m pytest slangpy/tests/slangpy_tests/test_reflection2.py -v
python -m pytest slangpy/tests/slangpy_tests/test_tensor.py -v
pre-commit run --all-files
```

Current limitations:

- High-level semantic `Type`, `Function`, `Field`, `Parameter`, and full `Layout` lookup behavior are still Python-owned.
- `slangpy/reflection/lookup.py` has not moved native yet.
- `Module.layout` still returns the Python `SlangProgramLayout`, not `sgl::refl::Layout`.
- `Struct` still delegates user-facing metadata to the Python `SlangType` object.
- Tensor, NumPy marshalling, `NDBuffer`, and `StridedBufferView` are not migrated yet.
- Generated stubs and public docs have not been updated for the temporary `native_refl` / `native_func` modules.

Next recommended step:

- Start the native lookup work needed by Tensor: introduce native type specs and a `LookupContext`/built-in layout path, then teach Python adapters for `resolve_program_layout()` and `resolve_element_type()` to route through native fast paths where possible.

## Nanobind Trampoline Rule

Use nanobind trampoline classes only when Python code must override native virtual functions.

Do not create a `PyFoo : Foo` just to add Python-specific methods. Bind those methods directly with lambdas.

Good:

```cpp
nb::class_<sgl::func::Tensor, Object>(m, "NativeTensor")
    .def_prop_ro("shape", &sgl::func::Tensor::shape)
    .def(
        "to_numpy",
        [](const sgl::func::Tensor& self)
        {
            return tensor_to_numpy(self);
        }
    )
    .def(
        "__getitem__",
        [](const ref<sgl::func::Tensor>& self, nb::object index)
        {
            TensorIndex parsed = parse_python_index(index, self->shape());
            return self->view(parsed.shape, parsed.strides, parsed.offset);
        }
    );
```

Allowed only when Python must override virtual behavior:

```cpp
class TypeProvider : public Object {
public:
    virtual ref<refl::Type> try_create_type(refl::Layout* layout, const TypeReflection* refl) = 0;
};

struct PyTypeProvider : TypeProvider {
    NB_TRAMPOLINE(TypeProvider, 1);
    ref<refl::Type> try_create_type(refl::Layout* layout, const TypeReflection* refl) override
    {
        NB_OVERRIDE(try_create_type, layout, refl);
    }
};
```

## Custom Python Reflection

We still need user-extensible reflection metadata from Python, but built-in native lookups must stay fast.

Use a two-level lookup:

1. Native fast path.
   Built-in factories handle known Slang types such as scalars, vectors, arrays, resources, tensors, tensor views, `DifferentialPair`, and `Unknown`.

2. Optional provider fallback.
   `slangpy_ext` can register a provider that calls Python registries such as the current `TYPE_OVERRIDES`.

The provider fallback should be called only on cache miss and only when native built-ins cannot create the type. Built-in native types should never consult the Python provider, even on first lookup.

Example native registry:

```cpp
namespace sgl::refl {

using TypeFactory = std::function<ref<Type>(Layout&, ref<const TypeReflection>)>;

class TypeRegistry {
public:
    void register_factory(std::string name, TypeFactory factory);
    ref<Type> create_builtin(Layout& layout, ref<const TypeReflection> reflection) const;

private:
    // Prefer structural recognition from reflection kind/access/scalar metadata.
    // String maps are only for named generic families that cannot be classified structurally.
    std::unordered_map<std::string, TypeFactory> m_by_short_name;
    std::unordered_map<std::string, TypeFactory> m_by_full_name;
};

class TypeProvider : public Object {
public:
    virtual ref<Type> try_create_type(Layout& layout, ref<const TypeReflection> reflection) = 0;
};

} // namespace sgl::refl
```

Registry ownership:

- `Layout` owns type/function caches for one low-level Slang program layout.
- A generic built-in registry can be shared by `Layout` instances or owned by a native lookup service.
- `slangpy_ext` may attach an optional provider to a layout or registry, but the provider must return native data, not Python-owned runtime objects.

Example lookup order:

```cpp
ref<refl::Type> Layout::get_or_create_type(ref<const TypeReflection> reflection)
{
    if (!reflection)
        return nullptr;

    if (auto existing = find_cached(reflection.get()))
        return existing;

    ref<Type> type = m_registry.create_builtin(*this, reflection);
    if (!type && m_provider)
        type = m_provider->try_create_type(*this, reflection);
    if (!type)
        type = make_ref<UnhandledType>(ref(this), reflection);

    cache_type(type);
    return type;
}
```

## Phase 0: Guardrails And Inventory

Purpose:

- Add lightweight compile-time and review guardrails before moving code.
- Document and enforce where Python is allowed.
- Inventory deprecated `NDBuffer` and `StridedBufferView` usage so deletion is deliberate.

Implementation:

- Add a short comment block near new native headers stating that `src/sgl/refl` and `src/sgl/func` must stay Python-free.
- Keep `Shape` where it is for this migration. It already lives in `src/sgl/utils/slangpy.h`; moving it can be a later cleanup once the native reflection and Tensor APIs have settled.
- Open a tracking checklist for all public `NDBuffer` exports and generated docs updates.

Gating tests:

- Once `src/sgl/refl` and `src/sgl/func` exist, `rg -n "nanobind|nb::|PyObject|Python\\.h|pybind" src/sgl/refl src/sgl/func` returns no matches.
- Existing build still passes before behavior changes:

```powershell
cmake --build --preset windows-msvc-debug
pytest slangpy/tests/slangpy_tests/test_reflection2.py -v
pytest slangpy/tests/slangpy_tests/test_tensor.py -v
```

## Phase 1: Native Semantic Reflection

Purpose:

- Move the behavior of `slangpy/reflection/reflectiontypes.py` into `src/sgl`, without changing Python-facing behavior yet.
- Make type lookup, tensor type identification, layout lookup, generic parsing, and function lookup native.
- Move the runtime logic in `slangpy/reflection/lookup.py` into native lookup services before Tensor is made fully native.

This phase may be implemented as internal checkpoints, but the reflection types are tightly connected. Treat type/layout/function/field/parameter parity as one cohesive migration if splitting it would create duplicate type systems.

Suggested checkpoints:

1. Type/layout parity: scalar, vector, matrix, array, resource, tensor, tensor view, differential pair, pointer, sampler, raytracing, unknown, void, fields, and parameters.
2. Base module/struct bridge: introduce native `BaseModule` and `BaseStruct` so lookup can understand structs without depending on Python `Module`/`Struct` objects.
3. Tensor-critical lookup: built-in layout, `resolve_program_layout`, `resolve_element_type`, scalar-array mapping, and cross-layout lookup by `full_name`.
4. Function metadata parity: functions, methods, overload lists, modifiers, constructor detection, and specialization metadata. Native function dispatch can still wait until Phase 8.
5. Optional Python provider fallback for custom metadata.

Suggested files:

- `src/sgl/refl/type.h`
- `src/sgl/refl/type.cpp`
- `src/sgl/refl/layout.h`
- `src/sgl/refl/layout.cpp`
- `src/sgl/refl/type_registry.h`
- `src/sgl/refl/type_registry.cpp`
- `src/sgl/func/base_module.h`
- `src/sgl/func/base_module.cpp`
- `src/sgl/func/base_struct.h`
- `src/sgl/func/base_struct.cpp`
- `src/sgl/refl/lookup.h`
- `src/sgl/refl/lookup.cpp`
- `tests/sgl/refl/test_reflection.cpp`
- `tests/sgl/func/test_reflection.cpp`
- `tests/sgl/refl/test_lookup.cpp`
- `src/slangpy_ext/refl/*.cpp`
- `src/slangpy_ext/func/*.cpp`
- `src/sgl/CMakeLists.txt`
- `tests/CMakeLists.txt`
- `src/slangpy_ext/CMakeLists.txt` once bindings begin

Core native type model:

```cpp
namespace sgl::refl {

class TypeLayout : public Object {
public:
    explicit TypeLayout(ref<const TypeLayoutReflection> reflection)
        : m_reflection(std::move(reflection))
    {
    }

    const TypeLayoutReflection* reflection() const { return m_reflection.get(); }
    size_t size() const { return m_reflection->size(); }
    size_t alignment() const { return m_reflection->alignment(); }
    size_t stride() const { return m_reflection->stride(); }

private:
    ref<const TypeLayoutReflection> m_reflection;
};

class Type : public Object {
public:
    Type(ref<Layout> layout, ref<const TypeReflection> reflection, ref<Type> element_type, Shape local_shape);

    Layout* layout() const { return m_layout.get(); }
    const TypeReflection* reflection() const { return m_reflection.get(); }

    std::string name() const;
    std::string full_name() const;
    const Shape& shape() const { return m_shape; }

    virtual bool is_generic() const;
    virtual std::string vector_type_name() const;
    virtual ref<Type> derivative();

    ref<TypeLayout> uniform_layout();
    ref<TypeLayout> buffer_layout();
    ref<Type> element_type() const { return m_element_type; }

protected:
    ref<Layout> m_layout;
    ref<const TypeReflection> m_reflection;
    ref<Type> m_element_type;
    Shape m_shape;
    ref<TypeLayout> m_uniform_layout;
    ref<TypeLayout> m_buffer_layout;
    ref<Type> m_derivative;
};

class ScalarType final : public Type {
public:
    TypeReflection::ScalarType scalar_type() const { return reflection()->scalar_type(); }
};

class TensorType final : public Type {
public:
    enum class Kind { tensor, itensor, diff_tensor, idiff_tensor, primal_tensor, atomic };
    enum class Access { read, write, read_write };

    Kind tensor_kind() const { return m_kind; }
    Access access() const { return m_access; }
    int dims() const { return m_dims; }
    bool readable() const;
    bool writable() const;
    bool has_grad_in() const;
    bool has_grad_out() const;

    static std::string build_name(const Type& element_type, int dims, Access access, Kind kind);

private:
    Kind m_kind;
    Access m_access;
    int m_dims = 0;
};

class TensorViewType final : public Type {
public:
    int dims() const { return m_dims; }
    ref<Type> storage_element_type() const { return m_storage_element_type; }

private:
    int m_dims = 0;
    ref<Type> m_storage_element_type;
};

class DiffTensorViewType final : public Type {
public:
    int dims() const { return m_dims; }
    ref<Type> primal_element_type() const { return m_primal_element_type; }

private:
    int m_dims = 0;
    ref<Type> m_primal_element_type;
};

} // namespace sgl::refl
```

Base module and struct bridge:

Tensor lookup currently accepts a Python `Struct`, and `Struct` carries both a reflected type and a reference to its owning Python `Module`. That is important because resolving the same type name in the wrong composed layout can produce the wrong answer. Before Tensor becomes native, create small native bases that carry the lookup-critical identity without pulling Python behavior into `src/sgl`.

```cpp
namespace sgl::func {

class BaseModule : public Object {
public:
    BaseModule(ref<SlangModule> module, ref<refl::Layout> layout);

    SlangModule* module() const { return m_module.get(); }
    refl::Layout* layout() const { return m_layout.get(); }
    Device* device() const;
    std::string_view name() const;

    void on_hot_reload(ref<SlangModule> module, ref<const sgl::ProgramLayout> low_level_layout);

private:
    ref<SlangModule> m_module;
    ref<refl::Layout> m_layout;
};

class BaseStruct : public Object {
public:
    BaseStruct(ref<BaseModule> module, ref<refl::Type> type);

    BaseModule* module() const { return m_module.get(); }
    refl::Layout* layout() const { return m_module->layout(); }
    refl::Type* type() const { return m_type.get(); }

    std::string_view name() const;
    std::string full_name() const;
    ref<refl::Type> element_type() const;
    ref<refl::Type> derivative() const;
    const Shape& shape() const;
    ref<refl::TypeLayout> uniform_layout() const;
    ref<refl::TypeLayout> buffer_layout() const;

private:
    ref<BaseModule> m_module;
    ref<refl::Type> m_type;
};

} // namespace sgl::func
```

Python should then layer behavior on top:

```python
class Module(BaseModule):
    def __init__(self, device_module: SlangModule, options: dict[str, Any] = {}, ...):
        composed = _compose_python_module(device_module, ...)
        super().__init__(composed, Layout(composed.layout))
        self.options = options
        self.call_data_cache = CallDataCache()


class Struct(BaseStruct):
    def __init__(self, module: Module, slang_struct: Type, options: dict[str, Any] = {}):
        super().__init__(module, slang_struct)
        self.options = options
        self.slangpy_signature = self.full_name
```

`BaseModule` should not own Python options, attribute caches, call-data caches, loaded-module registries, or function dispatch behavior. It only owns enough state for native lookup to answer: "which composed layout should this reflected type be resolved in?" That same object can later become the foothold for moving more module/function behavior native.

Semantic layout:

```cpp
namespace sgl::refl {

class Layout : public Object {
public:
    explicit Layout(ref<const sgl::ProgramLayout> low_level_layout);

    ref<Type> find_type(ref<const TypeReflection> reflection);
    ref<Type> find_type_by_name(std::string_view name);
    ref<Type> require_type_by_name(std::string_view name);

    ref<ScalarType> scalar_type(TypeReflection::ScalarType scalar_type);
    ref<Type> vector_type(TypeReflection::ScalarType scalar_type, int size);
    ref<Type> matrix_type(TypeReflection::ScalarType scalar_type, int rows, int cols);
    ref<TensorType> tensor_type(ref<Type> element_type, int dims, TensorType::Access access, TensorType::Kind kind);

    ref<Function> find_function_by_name(std::string_view name);
    ref<Function> find_function_by_name_in_type(ref<Type> type, std::string_view name);

    using GenericArg = std::variant<int, ref<Type>>;
    std::optional<std::vector<GenericArg>> resolved_generic_args(ref<const TypeReflection> type);

    void on_hot_reload(ref<const sgl::ProgramLayout> low_level_layout);

private:
    ref<Type> get_or_create_type(ref<const TypeReflection> reflection);
    ref<Function> get_or_create_function(ref<const FunctionReflection> reflection, ref<Type> this_type, std::string full_name);

    ref<const sgl::ProgramLayout> m_layout;
    std::unordered_map<const TypeReflection*, ref<Type>> m_types_by_reflection;
    std::unordered_map<std::string, ref<Type>> m_types_by_name;
    std::unordered_map<const FunctionReflection*, ref<Function>> m_functions_by_reflection;
    std::unordered_map<std::string, ref<Function>> m_functions_by_name;
};

} // namespace sgl::refl
```

Lookup services:

`slangpy/reflection/lookup.py` is small but strategically important. Tensor factory paths call `resolve_program_layout()` and `resolve_element_type()`, and those currently fall back to `_get_lookup_module(device)`, which uses a process-global dictionary keyed by Python `Device`.

Move the stable, runtime parts of this into `src/sgl/refl`:

- `innermost_type`
- scalar type name mapping currently used by `slang_to_numpy`
- scalar/vector/array lookup currently used by `numpy_to_slang`
- `resolve_program_layout`
- `resolve_element_type`
- a generic per-device built-in layout equivalent to the current dummy lookup module behavior

The native layer must not know about `np.dtype`, `Marshall`, or Python `Struct`. The Python adapter should translate those into native-friendly inputs, then call native lookup.

Example native API shape:

```cpp
namespace sgl::refl {

struct TypeSpec {
    enum class Kind {
        type,
        type_name,
        type_reflection,
        type_layout_reflection,
        base_struct,
        scalar,
    };

    Kind kind;
    ref<Type> type;
    std::string type_name;
    ref<const TypeReflection> type_reflection;
    ref<const TypeLayoutReflection> type_layout_reflection;
    ref<func::BaseStruct> base_struct;
    TypeReflection::ScalarType scalar_type = TypeReflection::ScalarType::none_;
    Shape array_shape;
};

class LookupContext : public Object {
public:
    explicit LookupContext(ref<Layout> default_layout);

    ref<Layout> default_layout();
    ref<Layout> resolve_layout(ref<Layout> preferred, const TypeSpec& element_type);
    ref<Type> resolve_element_type(ref<Layout> layout, const TypeSpec& element_type);

    ref<Type> scalar_array_type(TypeReflection::ScalarType scalar_type, const Shape& array_shape, ref<Layout> preferred);
    TypeReflection::ScalarType scalar_type_from_external_name(std::string_view name) const;
    std::optional<std::string_view> external_name_from_scalar_type(TypeReflection::ScalarType scalar_type) const;

    ref<Type> innermost_type(ref<Type> type) const;

    void on_hot_reload();

private:
    ref<Layout> m_default_layout;
};

} // namespace sgl::refl
```

Example generic `Device` ownership:

```cpp
class Device : public Object {
public:
    ref<refl::Layout> builtin_layout();
    ref<refl::LookupContext> builtin_lookup();

private:
    ref<refl::Layout> m_builtin_layout;
    ref<refl::LookupContext> m_builtin_lookup;
};
```

The built-in layout should be loaded once per device using a generic SGL support module, not a SlangPy/Python-specific module. It should refresh on shader hot reload and be released with the owning device. This removes `_global_lookup_modules` and makes the cache lifetime match the `Device` lifetime.

Module-level binding functions in `slangpy_ext`, if any, should be stateless shims over `LookupContext`; all lookup state and caches belong to native layout/lookup objects.

Python compatibility adapter:

```python
def resolve_program_layout(
    device: Device, element_type: Any, layout: Layout | None
) -> Layout:
    spec = _element_type_to_native_spec(element_type)
    preferred = layout or _layout_from_python_object(element_type)
    return device._builtin_lookup.resolve_layout(preferred, spec)


def resolve_element_type(layout: Layout, element_type: Any) -> Type:
    spec = _element_type_to_native_spec(element_type)
    return _spy_ext.resolve_element_type(layout, spec)
```

The adapter still owns Python-specific interpretation:

- `np.dtype` becomes a scalar type plus `dtype.shape`.
- `Struct` should already be a `BaseStruct`; native lookup can use its owning `BaseModule`/`Layout` directly.
- Legacy Python struct-like objects can still become a full type name and, when possible, the source `Layout`.
- `Marshall` becomes its native `refl::Type`.
- Python classes still go through the existing type registry/fallback until their marshalling is migrated.

Details to preserve:

- `Shape` composition rules from Python `SlangType.__init__`.
- `full_name` string comparison behavior where exact reflection identity is not enough.
- Buffer layout lookup via `StructuredBuffer<T>` and `get_type_layout(...).element_type_layout()`.
- Generic argument parsing behavior, including nested generics and bool `let` parameters.
- Tensor naming rules for `Tensor`, `WTensor`, `RWTensor`, `DiffTensor`, `PrimalTensor`, `AtomicTensor`, `TensorView`, and `DiffTensorView`.
- Separate semantic models for `TensorType`, `TensorViewType`, and `DiffTensorViewType`.
- `Function`, `Field`, and `Parameter` behavior, including modifiers, overload lists, method lookup, constructor detection, and specialization metadata.
- Pointer, sampler, raytracing, `DifferentialPair`, `Unknown`, and void type handling.
- Hot reload cache refresh semantics.
- `_get_lookup_module` fallback behavior, but with the cache owned by a generic per-device built-in layout rather than by a Python global dictionary.
- `resolve_program_layout` behavior for `Type`, `Marshall`, `BaseStruct`, legacy `Struct`, `TypeReflection`, `TypeLayoutReflection`, strings, Python scalar types, and fallback registry-created types.
- `resolve_element_type` behavior when resolving a type from a different program layout by `full_name`.
- `numpy_to_slang` and `slang_to_numpy` behavior for scalar dtypes, shaped subarray dtypes, and unsupported dtypes.

Hot reload contract:

- Each semantic `Layout` owns a generation id.
- Pointer-keyed maps are valid only within one generation.
- `on_hot_reload()` replaces the low-level layout, increments the generation, clears pointer-keyed caches, and rebuilds name-keyed entries from `full_name` where possible.
- Existing native type/function handles either relink to the new reflection objects by `full_name` or become explicitly invalid if the type/function no longer exists.
- Call-data caches and cached binding offsets store the layout generation they were built against and are invalidated in bulk on reload. Per-dispatch checks should be limited to a cheap generation compare.

Gating tests:

- New C++ test executable covers:
  - scalar lookup for all scalar types except void/none where appropriate
  - vector and matrix shape
  - array element type and generic array dimensions
  - struct fields
  - resource access and element type
  - `BaseModule`/`BaseStruct` preserve the owning composed layout for struct dtype resolution
  - `Tensor.empty(..., dtype=module.SomeStruct)` and `Tensor.from_numpy(..., target_slang_dtype=module.SomeStruct)` resolve against the struct's module layout
  - `Tensor<float,2>`, `RWTensor<float,2>`, `DiffTensor<float,2>`, `TensorView<float4>`, `DiffTensorView<float4>`
  - nested generic parsing: `GenericType<GenericType<float, 1>, 2>`
  - bool generic parsing: `BoolGenericType<true>` and `<false>`
  - function lookup, method lookup, overload list, and specialization
  - generic built-in layout is cached per device, refreshed on hot reload, and released on device close
  - `resolve_element_type` for names, reflection objects, type layout objects, scalar specs, array specs, and cross-layout `full_name` resolution
  - `scalar_array_type` for scalar and shaped subarray equivalents
  - `innermost_type` for vectors, matrices, arrays, resources, and tensors
- Existing Python tests remain in place. Add native C++ coverage rather than replacing Python tests; the Python tests continue validating that the binding layer and compatibility adapters are robust.

```powershell
cmake --build --preset windows-msvc-debug --target sgl_tests
ctest --test-dir build\windows-msvc -C Debug -R sgl_tests --output-on-failure
pytest slangpy/tests/device/test_reflection.py -v
pytest slangpy/tests/slangpy_tests/test_reflection2.py -v
pytest slangpy/tests/slangpy_tests/test_structured_numpy.py -v
pytest slangpy/tests/slangpy_tests/test_textures.py -k numpy -v
```

## Phase 2: Python Reflection Compatibility Over Native Reflection

Purpose:

- Make Python reflection use `sgl::refl` while keeping existing Python APIs stable.
- Keep custom Python type metadata possible through a provider fallback.

Implementation:

- Bind `sgl::refl::Type` and derived classes under `src/slangpy_ext/refl`.
- Bind `sgl::func::BaseModule` and `sgl::func::BaseStruct` under `src/slangpy_ext/func/functional.cpp`.
- Bind transitional Python modules as `slangpy.native_refl` and `slangpy.native_func`; do not add these classes through `slangpy.core.native`.
- Make Python `Module` inherit from or wrap `BaseModule`.
- Make Python `Struct` inherit from or wrap `BaseStruct`.
- Python `SlangType`, `ITensorType`, etc. should become aliases, wrappers, or subclasses only where Python compatibility requires it.
- Do not use trampolines for `SlangType` unless Python must override a virtual method.
- If Python-specific display or convenience helpers are needed, bind lambdas or implement Python wrapper methods.
- Update binding source lists, generated stubs, `py_doc.h`, and generated API docs as the bindings move.

Example binding shape:

```cpp
nb::module_ native_refl = nb::module_::import_("slangpy.native_refl");
nb::module_ native_func = nb::module_::import_("slangpy.native_func");

nb::class_<refl::Type, Object>(native_refl, "Type")
    .def_prop_ro("name", &refl::Type::name)
    .def_prop_ro("full_name", &refl::Type::full_name)
    .def_prop_ro("shape", &refl::Type::shape)
    .def_prop_ro("type_reflection", &refl::Type::reflection)
    .def_prop_ro("element_type", &refl::Type::element_type)
    .def_prop_ro("uniform_layout", &refl::Type::uniform_layout)
    .def_prop_ro("buffer_layout", &refl::Type::buffer_layout)
    .def("__repr__", &refl::Type::to_string);

nb::class_<refl::TensorType, refl::Type>(native_refl, "TensorType")
    .def_prop_ro("dims", &refl::TensorType::dims)
    .def_prop_ro("readable", &refl::TensorType::readable)
    .def_prop_ro("writable", &refl::TensorType::writable)
    .def_prop_ro("has_grad_in", &refl::TensorType::has_grad_in)
    .def_prop_ro("has_grad_out", &refl::TensorType::has_grad_out);

nb::class_<func::BaseModule, Object>(native_func, "BaseModule")
    .def("__repr__", &func::BaseModule::to_string);

nb::class_<func::BaseStruct, Object>(native_func, "BaseStruct")
    .def("__repr__", &func::BaseStruct::to_string);
```

For the transitional bridge, avoid exposing duplicated Python-facing `native_module`, `native_layout`, `native_name`, etc. If Python still needs the old `Module.layout` or `Struct.name`, keep those existing fields until the native field can replace them outright.

Python provider fallback:

```cpp
struct CustomTypeDesc {
    std::string full_name;
    Shape shape;
    ref<refl::Type> element_type;
    ref<const TypeLayoutReflection> uniform_layout;
    ref<const TypeLayoutReflection> buffer_layout;
    std::optional<std::string> derivative_full_name;
};

class PythonTypeProvider final : public refl::TypeProvider {
public:
    explicit PythonTypeProvider(nb::object callback)
        : m_callback(std::move(callback))
    {
    }

    ref<refl::Type> try_create_type(refl::Layout& layout, ref<const TypeReflection> reflection) override
    {
        nb::gil_scoped_acquire gil;
        nb::object result = m_callback(nb::cast(&layout), nb::cast(reflection.get()));
        if (result.is_none())
            return nullptr;
        if (nb::isinstance<refl::Type>(result))
            return nb::cast<ref<refl::Type>>(result);

        CustomTypeDesc desc = custom_type_desc_from_python(result);
        return make_ref<refl::CustomType>(layout, reflection, std::move(desc));
    }

private:
    nb::object m_callback;
};
```

Provider contract:

- Python callbacks may return either an existing native `refl::Type` binding or a descriptor that can be copied into native data.
- `src/sgl` must not store `nb::object` or Python-owned `SlangType` instances.
- Existing Python `TYPE_OVERRIDES` should be adapted by `slangpy_ext` into `CustomTypeDesc` or equivalent native data.
- The provider must never be consulted for native built-in types. Cache warmup only affects custom/provider-created types and negative provider misses.

Gating tests:

- `slangpy/tests/slangpy_tests/test_reflection2.py` passes without requiring Python-owned `SlangType`.
- `slangpy/tests/slangpy_tests/test_type_resolution.py` passes for existing built-in type paths.
- New Python tests verify `Module` is a `BaseModule`, `Struct` is a `BaseStruct`, and Tensor dtype lookup from a `Struct` still uses the struct's owning module layout.
- A new Python test verifies custom Python reflection fallback:
  - register a custom type override from Python
  - first lookup calls provider
  - second lookup is served from native cache
  - built-in `Tensor<float,2>` lookup never calls provider
- `NativeSlangType` still exists only as compatibility, not as the authoritative type model.

Suggested commands:

```powershell
cmake --build --preset windows-msvc-debug --target slangpy_ext sgl_tests
pytest slangpy/tests/slangpy_tests/test_reflection2.py -v
pytest slangpy/tests/slangpy_tests/test_type_resolution.py -v
pytest slangpy/tests/slangpy_tests/test_interfaces.py -v
```

## Phase 3: Remove NDBuffer And StridedBufferView

Purpose:

- Delete deprecated `NDBuffer`.
- Make Tensor independent of `StridedBufferView` early so the old shared base stops shaping Tensor design.
- Remove `StridedBufferView` outright if possible; otherwise quarantine it so Tensor no longer depends on it and it has no public API path.
- Reduce the surface area before moving Tensor runtime into `src/sgl`.
- Preserve direct NumPy support while removing the `NDBuffer` implementation it currently leans on.

Implementation:

- First detach current extension `NativeTensorDesc`/`NativeTensor` from `StridedBufferView`.
  - Move storage, shape, stride, offset, view, broadcast, contiguous checks, clear, cursor/uniform helpers, and NumPy/Torch conversion helpers onto Tensor-specific code.
  - Keep this Tensor-specific code in the extension until Phase 4 moves it to `src/sgl::func`.
- Remove Python exports:
  - `slangpy/types/buffer.py`
  - `slangpy.builtin.ndbuffer`
  - `slangpy/builtin/__init__.py` entries
  - `NDBuffer` from `slangpy/types/__init__.py`
  - `NDBuffer` from root exports and stubs
  - `NDBuffer` registrations in `PYTHON_TYPES`
  - Jupyter `NDBuffer` formatting
  - generated stubs such as `slangpy/slangpy/__init__.pyi`
- Remove native extension classes:
  - `StridedBufferViewDesc`
  - `StridedBufferView`
  - `NativeNDBufferDesc`
  - `NativeNDBuffer`
  - `NativeNDBufferMarshall`
  - source-list entries in `src/slangpy_ext/CMakeLists.txt`
- Preserve `NativeNumpyMarshall`; do not remove NumPy support.
  - If it can be moved cleanly during Tensor detachment, make it a small extension-only adapter.
  - If not, keep a temporary compatibility bridge until Phase 5, where native Tensor marshalling exists and the Tensor-backed NumPy path can be done once.
  - It must not introduce a new native `NDBuffer`-like base in `src/sgl`.
- Replace `cursor_utils` checks for `StridedBufferView` with `Tensor` checks.
- Update docs according to `docs/tensorupdate.rst`.

Example: Tensor should own its view logic directly, not through `StridedBufferView`:

```cpp
class Tensor : public Object {
public:
    const Shape& shape() const { return m_desc.shape; }
    const Shape& strides() const { return m_desc.strides; }
    int offset() const { return m_desc.offset; }

    ref<Tensor> view(Shape shape, Shape strides = Shape(), int offset = 0) const;
    ref<Tensor> broadcast_to(const Shape& shape) const;

private:
    void view_inplace(Shape shape, Shape strides, int offset);
    void broadcast_to_inplace(const Shape& shape);

    TensorDesc m_desc;
    ref<Buffer> m_storage;
};
```

Gating tests:

- `rg -n "NDBuffer|NativeNDBuffer" slangpy src tests samples examples docs` returns only intentional migration notes or generated stale docs being regenerated in the same phase.
- `rg -n "StridedBufferView" src/slangpy_ext/func slangpy/types/tensor.py` returns no matches.
- Tensor tests pass:

```powershell
cmake --build --preset windows-msvc-debug --target slangpy_ext sgl_tests
pytest slangpy/tests/slangpy_tests/test_tensor.py -v
pytest slangpy/tests/slangpy_tests/test_tensor_with_grads.py -v
pytest slangpy/tests/slangpy_tests/test_difftensor*.py -v
pytest slangpy/tests/slangpy_tests/test_tensorview.py -v
pytest slangpy/tests/slangpy_tests/test_structured_numpy.py -v
pytest slangpy/tests/slangpy_tests/test_simple_function_call.py -k numpy -v
```

- Type resolution tests pass without `NDBufferMarshall`.
- Direct NumPy function arguments and NumPy return values continue to work, including structured NumPy error paths.
- Samples that used `NDBuffer` are migrated to `Tensor` or explicitly removed.

## Phase 4: Native Tensor Runtime

Purpose:

- Move Tensor's data model and runtime behavior to pure native `sgl::func`.
- Keep Python factories and marshalling in the extension.
- Depend on Phase 1 native lookup for dtype resolution; Tensor should not need Python-owned `SlangType` behavior.

Suggested files:

- `src/sgl/func/tensor.h`
- `src/sgl/func/tensor.cpp`
- `src/slangpy_ext/func/tensor.cpp`
- `src/sgl/CMakeLists.txt`
- `src/slangpy_ext/CMakeLists.txt`

Native Tensor descriptor:

```cpp
namespace sgl::func {

struct TensorDesc {
    ref<refl::Type> dtype;
    ref<const TypeLayoutReflection> element_layout;
    int offset = 0;
    Shape shape;
    Shape strides;
    BufferUsage usage = BufferUsage::shader_resource | BufferUsage::unordered_access;
    MemoryType memory_type = MemoryType::device_local;
};

class Tensor : public Object {
public:
    Tensor(Device* device, TensorDesc desc, ref<Buffer> storage = nullptr);
    Tensor(TensorDesc desc, ref<Buffer> storage, ref<Tensor> grad_in, ref<Tensor> grad_out);

    Device* device() const;
    const TensorDesc& desc() const { return m_desc; }
    refl::Type* dtype() const { return m_desc.dtype.get(); }
    const Shape& shape() const { return m_desc.shape; }
    const Shape& strides() const { return m_desc.strides; }
    int offset() const { return m_desc.offset; }
    size_t element_count() const { return m_desc.shape.element_count(); }
    size_t element_stride() const { return m_desc.element_layout->stride(); }
    const ref<Buffer>& storage() const { return m_storage; }

    bool is_contiguous() const;
    void clear(CommandEncoder* command_encoder = nullptr);
    ref<BufferCursor> cursor(std::optional<int> start = std::nullopt, std::optional<int> count = std::nullopt) const;

    ref<Tensor> view(Shape shape, Shape strides = Shape(), int offset = 0) const;
    ref<Tensor> broadcast_to(const Shape& shape) const;
    ref<Tensor> detach() const;
    ref<Tensor> with_grads(ref<Tensor> grad_in = nullptr, ref<Tensor> grad_out = nullptr, bool zero = true) const;

    const ref<Tensor>& grad_in() const { return m_grad_in; }
    const ref<Tensor>& grad_out() const { return m_grad_out; }
    ref<Tensor> grad() const;

private:
    TensorDesc m_desc;
    ref<Buffer> m_storage;
    ref<Tensor> m_grad_in;
    ref<Tensor> m_grad_out;
};

} // namespace sgl::func
```

Python-specific factory stays in Python or extension:

```python
class Tensor(NativeTensor):
    @staticmethod
    def empty(device: Device, shape: TShapeOrTuple, dtype: Any, ...) -> Tensor:
        # These names can remain as Python adapters, but the implementation should
        # be the native lookup service from Phase 1.
        layout = resolve_program_layout(device, dtype, program_layout)
        dtype = resolve_element_type(layout, dtype)
        return _native_tensor_empty(device, Shape(shape), dtype, usage, memory_type)
```

The important boundary is that `resolve_program_layout()` and `resolve_element_type()` may still accept Python-shaped inputs for ergonomics, but the lookup cache, lookup module, name resolution, and type construction should already be native.

Python binding should use lambdas for Python-only operations. `NativeTensor` is a transitional internal binding name; public Python should continue exporting `Tensor`, and Phase 7 should either rename the binding or hide it behind the native public name.

```cpp
nb::class_<func::Tensor, Object>(slangpy, "NativeTensor")
    .def_prop_ro("dtype", &func::Tensor::dtype)
    .def_prop_ro("shape", &func::Tensor::shape)
    .def_prop_ro("strides", &func::Tensor::strides)
    .def_prop_ro("storage", &func::Tensor::storage)
    .def("clear", &func::Tensor::clear, "cmd"_a.none() = nullptr)
    .def("view", &func::Tensor::view, "shape"_a, "strides"_a = Shape(), "offset"_a = 0)
    .def("broadcast_to", &func::Tensor::broadcast_to, "shape"_a)
    .def("detach", &func::Tensor::detach)
    .def("with_grads", &func::Tensor::with_grads, "grad_in"_a.none() = nullptr, "grad_out"_a.none() = nullptr, "zero"_a = true)
    .def("to_numpy", [](const func::Tensor& self) { return tensor_to_numpy(self); })
    .def("copy_from_numpy", [](func::Tensor& self, nb::ndarray<nb::numpy> data) { tensor_copy_from_numpy(self, data); })
    .def("__getitem__", [](const ref<func::Tensor>& self, nb::object index) { return tensor_index(self, index); });
```

Gating tests:

- New C++ tests cover:
  - contiguous stride calculation
  - view offset and stride handling
  - broadcast stride zeroing
  - `with_grads` creates derivative dtype storage
  - `detach` drops grads without copying storage
  - `clear` works with and without supplied command encoder
- Python Tensor tests pass:

```powershell
cmake --build --preset windows-msvc-debug --target slangpy_ext sgl_tests
pytest slangpy/tests/slangpy_tests/test_tensor.py -v
pytest slangpy/tests/slangpy_tests/test_tensor_with_grads.py -v
pytest slangpy/tests/slangpy_tests/test_difftensor_load_variants.py -v
pytest slangpy/tests/slangpy_tests/test_tensorview.py -v
```

- `rg -n "NativeSlangType" src/sgl src/slangpy_ext/func` shows Tensor runtime no longer depends on Python-backed type callbacks.

## Phase 5: Keep Marshalling In Extension, Retarget To Native Types

Purpose:

- Leave function marshalling and code generation in `slangpy_ext`/Python.
- Make existing `NativeTensorMarshall` consume `sgl::func::Tensor` and `sgl::refl::Type`.

Implementation:

- Change `NativeMarshall` and related runtime binding objects only as much as needed to accept native reflection types.
- Keep virtual marshalling methods where Python must still override behavior.
- Keep `PyNativeMarshall` only for classes that genuinely allow Python override.
- Remove `PyNativeTensorMarshall` if Tensor marshalling no longer needs Python overrides.
- Keep `NativeNumpyMarshall` as an extension-only adapter. It should reuse Tensor dispatch data where possible so NumPy support survives the `NDBuffer` deletion without creating another native buffer-view abstraction.
- Direct `np.ndarray` arguments are a convenience path, not the high-performance repeated-call path. Persistent `Tensor.from_numpy()` plus `copy_from_numpy()` should remain the recommended route for repeated calls.
- Do not redesign the separate `slangpy_torch` bridge in this phase. Preserve its ABI and current CUDA direct-pointer behavior; update only the extension marshalling surface needed to consume native reflection/Tensor metadata.
- Add an explicit per-call temporary lifetime model before Tensor-backed NumPy dispatch uses temporary native objects.

Example direction:

```cpp
class NativeTensorMarshall : public NativeMarshall {
public:
    NativeTensorMarshall(
        int dims,
        bool writable,
        ref<refl::Type> slang_type,
        ref<refl::Type> slang_element_type,
        ref<const TypeLayoutReflection> element_layout,
        ref<NativeTensorMarshall> d_in,
        ref<NativeTensorMarshall> d_out
    );

    Shape get_shape(nb::object data) const override
    {
        auto tensor = nb::cast<func::Tensor*>(data);
        return tensor->shape();
    }
};
```

Example per-call temporary storage:

```cpp
struct NativeCallTemporaries {
    std::vector<ref<Object>> native_objects;
    nb::list python_objects;
};

class CallContext {
public:
    NativeCallTemporaries& temporaries() { return m_temporaries; }

private:
    NativeCallTemporaries m_temporaries;
};
```

Preferred `NativeNumpyMarshall` direction after native Tensor marshalling exists:

```cpp
class NativeNumpyMarshall : public NativeMarshall {
public:
    NativeNumpyMarshall(
        int dims,
        ref<refl::Type> slang_type,
        ref<refl::Type> slang_element_type,
        ref<const TypeLayoutReflection> element_layout,
        nb::dtype dtype,
        ref<NativeTensorMarshall> tensor_marshall
    );

    Shape get_shape(nb::object data) const override
    {
        nb::ndarray<nb::numpy> ndarray = nb::cast<nb::ndarray<nb::numpy>>(data);
        return numpy_shape_to_call_shape(ndarray, m_dims);
    }

    void write_shader_cursor_pre_dispatch(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        ShaderCursor cursor,
        nb::object value,
        nb::list read_back
    ) const override
    {
        ref<func::Tensor> temporary = tensor_from_numpy(context->device(), value, m_slang_element_type);
        context->temporaries().native_objects.push_back(temporary);
        m_tensor_marshall->write_tensor_dispatch_data(context, binding, cursor, temporary, read_back);
    }

private:
    nb::dtype m_dtype;
    ref<refl::Type> m_slang_element_type;
    ref<NativeTensorMarshall> m_tensor_marshall;
};
```

This keeps NumPy adaptation in `src/slangpy_ext`, where nanobind and NumPy are allowed, while making the runtime dispatch path converge on native Tensor. If direct NumPy output is requested, the marshaller should allocate a temporary native Tensor, dispatch into it, then read back into a NumPy array during `read_calldata()`.

Gating tests:

- Full functional Tensor dispatch tests pass.
- Direct `np.ndarray` arguments still marshal correctly through `NativeNumpyMarshall`, including repeated-call cache hits.
- Torch integration tests still pass on CUDA-capable machines, with no required `slangpy_torch` bridge redesign.
- No new trampoline class is introduced unless its virtual override use is documented in code.
- Repeat-call signature cache still hits for native Tensor values.
- NumPy direct dispatch and NumPy output tests cover temporary lifetime and async dispatch safety.
- Benchmarks distinguish direct NumPy convenience calls from persistent Tensor repeated calls.
- Torch performance checks preserve CUDA direct-pointer behavior and do not introduce an avoidable Tensor copy for the existing fast path.

Suggested commands:

```powershell
cmake --build --preset windows-msvc-debug --target slangpy_ext falcor2_ext falcor2_tests
pytest slangpy/tests/slangpy_tests/test_tensor.py -v
pytest slangpy/tests/slangpy_tests/test_tensor_with_grads.py -v
pytest slangpy/tests/slangpy_tests/test_simple_function_call.py -k numpy -v
pytest slangpy/tests/slangpy_tests/test_structured_numpy.py -v
pytest slangpy/tests/slangpy_tests/test_torchintegration.py -v
pytest slangpy/tests/slangpy_tests/test_tensorview.py -v
```

## Phase 6: Retire NativeSlangType

Purpose:

- Remove the compatibility type once reflection and Tensor no longer need it.

Implementation:

- Replace `ref<NativeSlangType>` in `NativeMarshall`, `NativeBoundVariableRuntime`, resource marshalls, texture marshalls, value marshalls, and extension torch tensor marshalling metadata with `ref<sgl::refl::Type>`.
- Remove `_py_element_type`, `_py_derivative`, `_py_uniform_type_layout`, and `_py_buffer_type_layout`.
- Remove `PyNativeSlangType`.
- Keep Python `SlangType` name as the binding for `sgl::refl::Type`.
- Keep the separate `slangpy_torch` bridge ABI unchanged unless a small compatibility update is required by the extension type changes.

Gating tests:

- `rg -n "NativeSlangType|PyNativeSlangType|_py_element_type|_py_derivative|_py_uniform_type_layout|_py_buffer_type_layout" src slangpy` returns no runtime references.
- Reflection, type resolution, value, texture, tensor, and torch tests pass.
- Stubs regenerate without `NativeSlangType`.

Suggested commands:

```powershell
cmake --build --preset windows-msvc-debug --target slangpy_ext sgl_tests
pytest slangpy/tests/device/test_reflection.py -v
pytest slangpy/tests/slangpy_tests/test_reflection2.py -v
pytest slangpy/tests/slangpy_tests/test_type_resolution.py -v
pytest slangpy/tests/slangpy_tests/test_tensor.py -v
pytest slangpy/tests/slangpy_tests/test_textures.py -v
```

## Phase 7: Native Naming Cleanup In Python

Purpose:

- Remove old compatibility names after the runtime has moved to native reflection types.
- Make Python use the same names as native so future native/Python documentation does not describe two parallel type systems.

Implementation:

- Rename Python-facing reflection exports to match native:
  - `SlangProgramLayout` becomes `Layout`.
  - `SlangType` becomes `Type`.
  - `ScalarType`, `VectorType`, `MatrixType`, `ArrayType`, `TensorType`, `Function`, `Field`, and `Parameter` should align with `sgl::refl` names.
- Remove stale compatibility aliases and imports once all internal Python code has moved.
- Update stubs, docs, examples, and tests to import the native names.
- Keep migration notes in `native.md` or release notes, but do not keep old names as permanent runtime aliases.
- Check that generated signatures and repr strings use the native names consistently.

Example target imports:

```python
from slangpy.reflection import Layout, Type, TensorType

layout: Layout = module.layout
dtype: Type = layout.find_type_by_name("float")
```

Gating tests:

- `rg -n "SlangType|SlangProgramLayout|NativeSlangType" slangpy src tests docs examples samples` returns no runtime references, except migration notes if intentionally kept.
- Existing reflection and Tensor tests pass with native names.
- Public stubs expose `Layout` and `Type` without the old compatibility names.

Suggested commands:

```powershell
cmake --build --preset windows-msvc-debug --target slangpy_ext sgl_tests
pytest slangpy/tests/device/test_reflection.py -v
pytest slangpy/tests/slangpy_tests/test_reflection2.py -v
pytest slangpy/tests/slangpy_tests/test_tensor.py -v
pytest slangpy/tests/slangpy_tests/test_type_resolution.py -v
```

## Phase 8: Native Module And Function Follow-Up

Purpose:

- After reflection and Tensor are native, identify which parts of `Module`, `Function`, `CallData`, and type resolution can move to native.

Candidate migration order:

1. Expand the existing `BaseModule`/`BaseStruct` bridge into native module composition and owned `sgl::refl::Layout`.
2. Native function metadata and overload resolution.
3. Native `CallArg`/`NativeValue` descriptors for Tensor/scalar/resource calls.
4. Native binding tree representation that does not require Python objects.
5. Binary signature keys for native values.
6. C++ readback vectors and per-call temporary state with no nanobind objects on the native-only path.
7. Native call-data cache for native-only callers.
8. Optional native code generation and dispatch wrapper for C++ users.

Gating tests:

- Existing Python API behavior remains unchanged.
- Native-only C++ sample can:
  - create a device
  - load a module importing `slangpy`
  - find a function
  - create a native Tensor
  - dispatch a simple elementwise function
  - read back results

Example native-only target:

```cpp
ref<Device> device = Device::create(DeviceDesc{});
ref<SlangModule> module = device->load_module("myshader.slang");
ref<func::Module> func_module = make_ref<func::Module>(module);

ref<refl::Type> float_type = func_module->layout()->require_type_by_name("float");
ref<func::Tensor> a = func::Tensor::empty(device.get(), {1024}, float_type);
ref<func::Tensor> b = func::Tensor::empty(device.get(), {1024}, float_type);

func_module->require_function("copy")->call({a, b});
```

Performance contract:

- Native-only cached dispatch should not require the GIL, `nb::object`, or Python readback lists.
- Call-data caches should be explicitly owned by native module/function objects.
- Command encoder batching and async copy paths should be possible without going through Python.
- Hot reload should invalidate cached native call data by layout generation, not by per-dispatch string checks.

## Final Acceptance Criteria

The migration is complete when:

- `Tensor` runtime behavior is implemented in pure `src/sgl`.
- `NDBuffer` is removed, and Tensor no longer depends on `StridedBufferView`. If any `StridedBufferView` compatibility code remains temporarily, it has no public API path and no Tensor dependency.
- High-level reflection is native and Python-compatible.
- Python `Module`/`Struct` are layered over native `BaseModule`/`BaseStruct`, and struct dtype lookup preserves the owning composed layout.
- Built-in type lookup is native and cached.
- `resolve_program_layout`, `resolve_element_type`, `numpy_to_slang`, and `slang_to_numpy` are backed by native lookup services, with Python only adapting Python-specific inputs.
- Python custom reflection is still possible but only via fallback/provider paths.
- `NativeSlangType` is gone.
- Old Python compatibility names have been removed after the native naming pass.
- New `src/sgl/refl` and `src/sgl/func` remain Python-free, and no new core SGL Python hooks are added for this migration.
- Existing SlangPy Tensor, reflection, type-resolution, and torch integration tests pass.

## Performance Gates

Add lightweight benchmarks or timing tests before the migration is considered complete:

- Warm and cached built-in type lookup latency.
- Cached signature lookup for existing Tensor/scalar/resource calls.
- Cached dispatch with existing native Tensor arguments.
- Direct NumPy convenience dispatch versus persistent Tensor upload/copy dispatch.
- NumPy output readback overhead and temporary lifetime behavior.
- Torch CUDA direct-pointer path and D3D/Vulkan interop paths, ensuring the bridge ABI and current fast path are preserved.
- Hot reload cache rebuild cost and stale-cache invalidation behavior.
- Native-only dispatch overhead with precompiled call data.

## Standard Verification

Each phase should end with:

```powershell
cmake --build --preset windows-msvc-debug
pytest slangpy/tests/device/test_reflection.py -v
pytest slangpy/tests/slangpy_tests/test_reflection2.py -v
pytest slangpy/tests/slangpy_tests/test_type_resolution.py -v
pytest slangpy/tests/slangpy_tests/test_tensor.py -v
pytest slangpy/tests/slangpy_tests/test_structured_numpy.py -v
pre-commit run --all-files
```

Phase-specific tests listed above should be added to this base set.

During subphase work, use targeted build/tests and `pre-commit run --files` for touched files. The full build and `pre-commit run --all-files` remain the gate for completing a phase or merging the branch.
