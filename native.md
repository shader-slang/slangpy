# Native SlangPy Migration Plan

## Goals

Move SlangPy toward a native-first architecture:

- Make `Tensor` a native runtime type, with as much of its data model and runtime behavior as possible in pure `sgl`.
- Keep Python as an adapter layer for Python-only conveniences, not as the owner of core runtime semantics.
- Preserve a path to native-only SlangPy-like functionality for C++ users.
- Keep the work structured, testable, and shippable in small phases.

## Boundaries

The most important boundary is simple:

- Code under `src/sgl` must not include `nanobind.h`, use `nb::object`, use `PyObject`, or call Python.
- Code under `src/slangpy_ext` owns nanobind bindings, Python object adaptation, NumPy/Torch conversion, and any Python registry fallback.
- Python files under `slangpy/` should become compatibility and ergonomic layers over native implementations.

The current low-level Slang reflection wrappers already live in `src/sgl/device/reflection.h`. The missing native piece is the higher-level semantic reflection model currently implemented in `slangpy/reflection/reflectiontypes.py`.

## Naming

Use native names that do not bake in Python:

```cpp
namespace sgl::refl {
class ProgramLayout;
class Type;
class Function;
class Field;
class Parameter;
}

namespace sgl::func {
class Tensor;
struct TensorDesc;
}
```

Python can keep exporting names such as `SlangType` and `SlangProgramLayout` for compatibility while the migration is in progress. Internally, those should become bindings or thin wrappers around `sgl::refl::Type` and `sgl::refl::ProgramLayout`.

`NativeSlangType` should eventually disappear. It exists today because the semantic `SlangType` is Python-only and native marshalling needs to call back into Python for `shape`, `derivative`, and layouts. Once `sgl::refl::Type` owns those concepts natively, `NativeSlangType` should first become a compatibility alias/shim and then be removed.

After that compatibility window, Python should use the same names as native (`ProgramLayout`, `Type`, `Tensor`, etc.) and the old compatibility names should be removed rather than kept indefinitely.

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
    virtual ref<refl::Type> try_create_type(refl::ProgramLayout* layout, const TypeReflection* refl) = 0;
};

struct PyTypeProvider : TypeProvider {
    NB_TRAMPOLINE(TypeProvider, 1);
    ref<refl::Type> try_create_type(refl::ProgramLayout* layout, const TypeReflection* refl) override
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

The provider fallback should be called only on cache miss and only when native built-ins cannot create the type.

Example native registry:

```cpp
namespace sgl::refl {

using TypeFactory = std::function<ref<Type>(ProgramLayout&, ref<const TypeReflection>)>;

class TypeRegistry {
public:
    void register_factory(std::string name, TypeFactory factory);
    ref<Type> create_builtin(ProgramLayout& layout, ref<const TypeReflection> reflection) const;

private:
    std::unordered_map<std::string, TypeFactory> m_by_short_name;
    std::unordered_map<std::string, TypeFactory> m_by_full_name;
};

class TypeProvider : public Object {
public:
    virtual ref<Type> try_create_type(ProgramLayout& layout, ref<const TypeReflection> reflection) = 0;
};

} // namespace sgl::refl
```

Example lookup order:

```cpp
ref<refl::Type> ProgramLayout::get_or_create_type(ref<const TypeReflection> reflection)
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

- Add a short comment block near new native headers stating that `src/sgl` must stay Python-free.
- Keep `Shape` in native code. It already lives in `src/sgl/utils/slangpy.h`; consider moving it later to a less SlangPy-specific native utility header.
- Open a tracking checklist for all public `NDBuffer` exports and generated docs updates.

Gating tests:

- `rg -n "nanobind|nb::|PyObject|python" src/sgl` shows no new Python dependencies, except comments explicitly documenting the boundary.
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

Suggested files:

- `src/sgl/refl/type.h`
- `src/sgl/refl/type.cpp`
- `src/sgl/refl/program_layout.h`
- `src/sgl/refl/program_layout.cpp`
- `src/sgl/refl/type_registry.h`
- `src/sgl/refl/type_registry.cpp`
- `src/sgl/refl/lookup.h`
- `src/sgl/refl/lookup.cpp`
- `tests/sgl/refl/test_reflection.cpp`
- `tests/sgl/refl/test_lookup.cpp`

Core native type model:

```cpp
namespace sgl::refl {

class Layout : public Object {
public:
    explicit Layout(ref<const TypeLayoutReflection> reflection)
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
    Type(ref<ProgramLayout> program, ref<const TypeReflection> reflection, ref<Type> element_type, Shape local_shape);

    ProgramLayout* program() const { return m_program.get(); }
    const TypeReflection* reflection() const { return m_reflection.get(); }

    std::string name() const;
    std::string full_name() const;
    const Shape& shape() const { return m_shape; }

    virtual bool is_generic() const;
    virtual std::string vector_type_name() const;
    virtual ref<Type> derivative();

    ref<Layout> uniform_layout();
    ref<Layout> buffer_layout();
    ref<Type> element_type() const { return m_element_type; }

protected:
    ref<ProgramLayout> m_program;
    ref<const TypeReflection> m_reflection;
    ref<Type> m_element_type;
    Shape m_shape;
    ref<Layout> m_uniform_layout;
    ref<Layout> m_buffer_layout;
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

} // namespace sgl::refl
```

Program layout:

```cpp
namespace sgl::refl {

class ProgramLayout : public Object {
public:
    explicit ProgramLayout(ref<const sgl::ProgramLayout> layout);

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

    void on_hot_reload(ref<const sgl::ProgramLayout> layout);

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
- device-owned lookup layout equivalent to the current `import "slangpy";` dummy module

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
        scalar,
    };

    Kind kind;
    ref<Type> type;
    std::string type_name;
    ref<const TypeReflection> type_reflection;
    ref<const TypeLayoutReflection> type_layout_reflection;
    TypeReflection::ScalarType scalar_type = TypeReflection::ScalarType::none;
    Shape array_shape;
};

class LookupContext : public Object {
public:
    explicit LookupContext(ref<Device> device);

    ref<ProgramLayout> lookup_layout();
    ref<ProgramLayout> resolve_program_layout(ref<ProgramLayout> preferred, const TypeSpec& element_type);
    ref<Type> resolve_element_type(ref<ProgramLayout> program_layout, const TypeSpec& element_type);

    ref<Type> numpy_type(TypeReflection::ScalarType scalar_type, const Shape& array_shape, ref<ProgramLayout> preferred);
    TypeReflection::ScalarType numpy_scalar_type_from_name(std::string_view name) const;
    std::optional<std::string_view> numpy_name_from_scalar_type(TypeReflection::ScalarType scalar_type) const;

    ref<Type> innermost_type(ref<Type> type) const;

    void on_device_close();
    void on_hot_reload();

private:
    ref<Device> m_device;
    ref<ProgramLayout> m_lookup_layout;
};

} // namespace sgl::refl
```

Example `Device` ownership:

```cpp
class Device : public Object {
public:
    ref<refl::LookupContext> slangpy_lookup();
    ref<refl::ProgramLayout> slangpy_lookup_layout();

private:
    ref<refl::LookupContext> m_slangpy_lookup;
};
```

The lookup layout should be loaded once per device using the native module-loading path with source equivalent to:

```slang
import "slangpy";
```

It should refresh on shader hot reload and be released with the owning device. This removes `_global_lookup_modules` and makes the cache lifetime match the `Device` lifetime.

Python compatibility adapter:

```python
def resolve_program_layout(
    device: Device, element_type: Any, program_layout: ProgramLayout | None
) -> ProgramLayout:
    spec = _element_type_to_native_spec(element_type)
    preferred = program_layout or _program_layout_from_python_object(element_type)
    return device._native_lookup.resolve_program_layout(preferred, spec)


def resolve_element_type(program_layout: ProgramLayout, element_type: Any) -> Type:
    spec = _element_type_to_native_spec(element_type)
    return _spy_ext.resolve_element_type(program_layout, spec)
```

The adapter still owns Python-specific interpretation:

- `np.dtype` becomes a scalar type plus `dtype.shape`.
- `Struct` becomes a full type name and, when possible, the source `ProgramLayout`.
- `Marshall` becomes its native `refl::Type`.
- Python classes still go through the existing type registry/fallback until their marshalling is migrated.

Details to preserve:

- `Shape` composition rules from Python `SlangType.__init__`.
- `full_name` string comparison behavior where exact reflection identity is not enough.
- Buffer layout lookup via `StructuredBuffer<T>` and `get_type_layout(...).element_type_layout()`.
- Generic argument parsing behavior, including nested generics and bool `let` parameters.
- Tensor naming rules for `Tensor`, `WTensor`, `RWTensor`, `DiffTensor`, `PrimalTensor`, `AtomicTensor`, `TensorView`, and `DiffTensorView`.
- Hot reload cache refresh semantics.
- `_get_lookup_module` behavior, but with the cache owned by `Device` rather than by a Python global dictionary.
- `resolve_program_layout` behavior for `Type`, `Marshall`, `Struct`, `TypeReflection`, `TypeLayoutReflection`, strings, Python scalar types, and fallback registry-created types.
- `resolve_element_type` behavior when resolving a type from a different program layout by `full_name`.
- `numpy_to_slang` and `slang_to_numpy` behavior for scalar dtypes, shaped subarray dtypes, and unsupported dtypes.

Gating tests:

- New C++ test executable covers:
  - scalar lookup for all scalar types except void/none where appropriate
  - vector and matrix shape
  - array element type and generic array dimensions
  - struct fields
  - resource access and element type
  - `Tensor<float,2>`, `RWTensor<float,2>`, `DiffTensor<float,2>`, `TensorView<float4>`, `DiffTensorView<float4>`
  - nested generic parsing: `GenericType<GenericType<float, 1>, 2>`
  - bool generic parsing: `BoolGenericType<true>` and `<false>`
  - function lookup, method lookup, overload list, and specialization
  - device-owned lookup layout is cached per device, refreshed on hot reload, and released on device close
  - `resolve_element_type` for names, reflection objects, type layout objects, scalar specs, array specs, and cross-layout `full_name` resolution
  - `numpy_type` for scalar and shaped subarray equivalents
  - `innermost_type` for vectors, matrices, arrays, resources, and tensors
- Existing Python tests remain in place. Add native C++ coverage rather than replacing Python tests; the Python tests continue validating that the binding layer and compatibility adapters are robust.

```powershell
cmake --build --preset windows-msvc-debug --target sgl_tests
ctest --test-dir build\windows-msvc -C Debug -R reflection --output-on-failure
ctest --test-dir build\windows-msvc -C Debug -R lookup --output-on-failure
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

- Bind `sgl::refl::Type` and derived classes into `slangpy_ext`.
- Python `SlangType`, `ITensorType`, etc. should become aliases, wrappers, or subclasses only where Python compatibility requires it.
- Do not use trampolines for `SlangType` unless Python must override a virtual method.
- If Python-specific display or convenience helpers are needed, bind lambdas or implement Python wrapper methods.

Example binding shape:

```cpp
nb::class_<refl::Type, Object>(m, "SlangType")
    .def_prop_ro("name", &refl::Type::name)
    .def_prop_ro("full_name", &refl::Type::full_name)
    .def_prop_ro("shape", &refl::Type::shape)
    .def_prop_ro("type_reflection", &refl::Type::reflection)
    .def_prop_ro("element_type", &refl::Type::element_type)
    .def_prop_ro("uniform_layout", &refl::Type::uniform_layout)
    .def_prop_ro("buffer_layout", &refl::Type::buffer_layout)
    .def("__repr__", &refl::Type::to_string);

nb::class_<refl::TensorType, refl::Type>(m, "ITensorType")
    .def_prop_ro("dims", &refl::TensorType::dims)
    .def_prop_ro("readable", &refl::TensorType::readable)
    .def_prop_ro("writable", &refl::TensorType::writable)
    .def_prop_ro("has_grad_in", &refl::TensorType::has_grad_in)
    .def_prop_ro("has_grad_out", &refl::TensorType::has_grad_out);
```

Python provider fallback:

```cpp
class PythonTypeProvider : public refl::TypeProvider {
public:
    explicit PythonTypeProvider(nb::object callback)
        : m_callback(std::move(callback))
    {
    }

    ref<refl::Type> try_create_type(refl::ProgramLayout& layout, ref<const TypeReflection> reflection) override
    {
        nb::gil_scoped_acquire gil;
        nb::object result = m_callback(nb::cast(&layout), nb::cast(reflection));
        if (result.is_none())
            return nullptr;
        return nb::cast<ref<refl::Type>>(result);
    }

private:
    nb::object m_callback;
};
```

The Python fallback must be optional and must not run for native built-in types after cache warmup.

Gating tests:

- `slangpy/tests/slangpy_tests/test_reflection2.py` passes without requiring Python-owned `SlangType`.
- `slangpy/tests/slangpy_tests/test_type_resolution.py` passes for existing built-in type paths.
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
- Remove `StridedBufferView` instead of preserving it as a shared base.
- Reduce the surface area before moving Tensor runtime into `src/sgl`.
- Preserve direct NumPy support while removing the `NDBuffer` implementation it currently leans on.

Implementation:

- Remove Python exports:
  - `slangpy/types/buffer.py`
  - `slangpy.builtin.ndbuffer`
  - `NDBuffer` from `slangpy/types/__init__.py`
  - `NDBuffer` from root exports and stubs
  - Jupyter `NDBuffer` formatting
- Remove native extension classes:
  - `StridedBufferViewDesc`
  - `StridedBufferView`
  - `NativeNDBufferDesc`
  - `NativeNDBuffer`
  - `NativeNDBufferMarshall`
- Rework `NativeNumpyMarshall`; do not remove NumPy support.
  - Today it inherits from `NativeNDBufferMarshall`.
  - After this phase it should either delegate to Tensor marshalling or be a small extension-only marshaller that uses the same native tensor dispatch representation.
  - It must not introduce a new native `NDBuffer`-like base in `src/sgl`.
- Move required Tensor-only logic into Tensor or Tensor Python helpers:
  - shape/stride/offset access
  - contiguous check
  - view/broadcast
  - clear
  - cursor/uniform helpers if still needed
  - NumPy/Torch conversion helpers
- Replace `cursor_utils` checks for `StridedBufferView` with `Tensor` checks.
- Update docs according to `docs/tensorupdate.rst`.

Preferred `NativeNumpyMarshall` direction:

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
        context->keep_alive(temporary);
        m_tensor_marshall->write_tensor_dispatch_data(context, binding, cursor, temporary, read_back);
    }

private:
    nb::dtype m_dtype;
    ref<refl::Type> m_slang_element_type;
    ref<NativeTensorMarshall> m_tensor_marshall;
};
```

This keeps NumPy adaptation in `src/slangpy_ext`, where nanobind and NumPy are allowed, while making the runtime dispatch path converge on native Tensor. If direct NumPy output is requested, the marshaller should allocate a temporary native Tensor, dispatch into it, then read back into a NumPy array during `read_calldata()`.

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

- `rg -n "NDBuffer|NativeNDBuffer|StridedBufferView" slangpy src tests samples examples docs` returns only intentional migration notes or generated stale docs being regenerated in the same phase.
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
- `src/slangpy_ext/utils/tensor_py.cpp`

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

Python binding should use lambdas for Python-only operations:

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

- `rg -n "NativeSlangType" src/sgl src/slangpy_ext/utils/slangpytensor.*` shows Tensor runtime no longer depends on Python-backed type callbacks.

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

Gating tests:

- Full functional Tensor dispatch tests pass.
- Direct `np.ndarray` arguments still marshal correctly through `NativeNumpyMarshall`, including repeated-call cache hits.
- Torch integration tests still pass on CUDA-capable machines.
- No new trampoline class is introduced unless its virtual override use is documented in code.
- Repeat-call signature cache still hits for native Tensor values.

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

- Replace `ref<NativeSlangType>` in `NativeMarshall`, `NativeBoundVariableRuntime`, resource marshalls, texture marshalls, value marshalls, and torch tensor marshalling with `ref<sgl::refl::Type>`.
- Remove `_py_element_type`, `_py_derivative`, `_py_uniform_type_layout`, and `_py_buffer_type_layout`.
- Remove `PyNativeSlangType`.
- Keep Python `SlangType` name as the binding for `sgl::refl::Type`.

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
  - `SlangProgramLayout` becomes `ProgramLayout`.
  - `SlangType` becomes `Type`.
  - `ScalarType`, `VectorType`, `MatrixType`, `ArrayType`, `TensorType`, `Function`, `Field`, and `Parameter` should align with `sgl::refl` names.
- Remove stale compatibility aliases and imports once all internal Python code has moved.
- Update stubs, docs, examples, and tests to import the native names.
- Keep migration notes in `native.md` or release notes, but do not keep old names as permanent runtime aliases.
- Check that generated signatures and repr strings use the native names consistently.

Example target imports:

```python
from slangpy.reflection import ProgramLayout, Type, TensorType

layout: ProgramLayout = module.layout
dtype: Type = layout.find_type_by_name("float")
```

Gating tests:

- `rg -n "SlangType|SlangProgramLayout|NativeSlangType" slangpy src tests docs examples samples` returns no runtime references, except migration notes if intentionally kept.
- Existing reflection and Tensor tests pass with native names.
- Public stubs expose `ProgramLayout` and `Type` without the old compatibility names.

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

1. Native module composition and owned `sgl::refl::ProgramLayout`.
2. Native function metadata and overload resolution.
3. Native binding tree representation that does not require Python objects.
4. Native call signature generation for native values.
5. Native call-data cache for native-only callers.
6. Optional native code generation and dispatch wrapper for C++ users.

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

## Final Acceptance Criteria

The migration is complete when:

- `Tensor` runtime behavior is implemented in pure `src/sgl`.
- `NDBuffer` and `StridedBufferView` are removed.
- High-level reflection is native and Python-compatible.
- Built-in type lookup is native and cached.
- `resolve_program_layout`, `resolve_element_type`, `numpy_to_slang`, and `slang_to_numpy` are backed by native lookup services, with Python only adapting Python-specific inputs.
- Python custom reflection is still possible but only via fallback/provider paths.
- `NativeSlangType` is gone.
- Old Python compatibility names have been removed after the native naming pass.
- `src/sgl` remains Python-free.
- Existing SlangPy Tensor, reflection, type-resolution, and torch integration tests pass.

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
