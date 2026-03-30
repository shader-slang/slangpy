// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <vector>
#include <map>
#include <typeindex>
#include <unordered_map>

#include "nanobind.h"

#include "sgl/core/macros.h"
#include "sgl/core/fwd.h"
#include "sgl/core/object.h"
#include "sgl/core/short_vector.h"
#include "sgl/device/fwd.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/shader_object.h"
#include "sgl/utils/slangpy.h"
#include "utils/torch_bridge.h"

namespace sgl::slangpy {

/// Transparent hash functor for std::string / std::string_view heterogeneous lookup.
struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view sv) const { return std::hash<std::string_view>{}(sv); }
    size_t operator()(const std::string& s) const { return std::hash<std::string_view>{}(s); }
};

/// Transparent equality functor for std::string / std::string_view heterogeneous lookup.
struct StringEqual {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const { return a == b; }
};

/// Stack-allocated signature buffer (non-Object, non-ref-counted).
/// Used in the hot dispatch path to avoid heap-allocating a SignatureBuilder.
/// Backed by short_vector<uint8_t, 1024> for inline SBO with heap fallback.
class SignatureBuffer {
public:
    SignatureBuffer() = default;

    // Non-copyable, non-movable (stack-only usage)
    SignatureBuffer(const SignatureBuffer&) = delete;
    SignatureBuffer& operator=(const SignatureBuffer&) = delete;

    void add(const std::string& value) { add_bytes(reinterpret_cast<const uint8_t*>(value.data()), value.length()); }
    void add(const char* value) { add_bytes(reinterpret_cast<const uint8_t*>(value), strlen(value)); }

    void add(uint32_t value)
    {
        static constexpr char hex[] = "0123456789abcdef";
        uint8_t buf[8];
        for (int i = 0; i < 8; ++i)
            buf[7 - i] = static_cast<uint8_t>(hex[(value >> (i * 4)) & 0xF]);
        add_bytes(buf, 8);
    }

    void add(uint64_t value)
    {
        static constexpr char hex[] = "0123456789abcdef";
        uint8_t buf[16];
        for (int i = 0; i < 16; ++i)
            buf[15 - i] = static_cast<uint8_t>(hex[(value >> (i * 4)) & 0xF]);
        add_bytes(buf, 16);
    }

    template<typename T>
    SignatureBuffer& operator<<(const T& v)
    {
        add(v);
        return *this;
    }

    std::string_view view() const { return {reinterpret_cast<const char*>(m_buf.data()), m_buf.size()}; }

private:
    short_vector<uint8_t, 1024> m_buf;

    void add_bytes(const uint8_t* data, size_t sz)
    {
        size_t old_size = m_buf.size();
        m_buf.resize(old_size + sz);
        memcpy(m_buf.data() + old_size, data, sz);
    }
};

/// Helper function to convert Shape to nb::list efficiently (avoids std::vector allocation)
inline nb::list shape_to_list(const Shape& shape)
{
    nb::list result;
    for (size_t i = 0; i < shape.size(); ++i) {
        result.append(shape[i]);
    }
    return result;
}

class NativeBoundVariableRuntime;
class NativeCallData;
class NativeFunctionNode;

/// General exception that includes a message and the bound variable from which the error
/// originated.
class NativeBoundVariableException : public std::exception {
public:
    NativeBoundVariableException(
        std::string_view message,
        ref<NativeBoundVariableRuntime> source = nullptr,
        ref<NativeCallData> context = nullptr
    )
        : m_message(message)
        , m_source(std::move(source))
        , m_context(std::move(context))
    {
    }

    virtual const char* what() const noexcept override { return m_message.c_str(); }

    std::string_view message() const { return m_message; }
    ref<NativeBoundVariableRuntime> source() const { return m_source; }
    ref<NativeCallData> context() const { return m_context; }

private:
    std::string m_message;
    ref<NativeBoundVariableRuntime> m_source;
    ref<NativeCallData> m_context;
};

/// Used during calculation of slangpy signature.
/// Thin Object wrapper around SignatureBuffer for Python binding compatibility.
class SignatureBuilder : public Object {
    SGL_OBJECT(SignatureBuilder)
public:
    SignatureBuilder() = default;

    void add(const std::string& value) { m_buf.add(value); }
    void add(const char* value) { m_buf.add(value); }
    void add(const uint32_t value) { m_buf.add(value); }
    void add(const uint64_t value) { m_buf.add(value); }

    template<typename T>
    SignatureBuilder& operator<<(const T& value)
    {
        m_buf.add(value);
        return *this;
    }

    nb::bytes bytes() const;

    std::string str() const;

    std::string_view view() const { return m_buf.view(); }

    std::string dbg_as_string() const { return std::string(m_buf.view()); }

    /// Access the underlying buffer (for callers that accept either type).
    SignatureBuffer& buffer() { return m_buf; }

private:
    SignatureBuffer m_buf;
};

/// Base class for types that can be passed to a slang function. Use of
/// this is optional, but it is the fastest way to supply signatures
/// to slangpy without entering python code. A user can set a fixed
/// signature on a NativeObject on construction, or override the
/// read_signature function to generate a signature dynamically.
class NativeObject : public Object {
    SGL_OBJECT(NativeObject)
public:
    NativeObject() = default;

    std::string_view slangpy_signature() const { return m_signature; }
    void set_slangpy_signature(std::string_view signature) { m_signature = signature; }

    virtual void read_signature(SignatureBuilder* builder) const { builder->add(m_signature); }

    /// Write signature into a stack-allocated SignatureBuffer (hot path).
    /// Non-virtual: delegates to the virtual read_signature(SignatureBuilder*) via a temporary wrapper.
    /// For NativeObject base class, we can write directly without the virtual call.
    void read_signature_fast(SignatureBuffer& builder) const { builder.add(m_signature); }

private:
    std::string m_signature;
};

/// Nanobind trampoline class for NativeObject
class PyNativeObject : public NativeObject {
    SGL_OBJECT(PyNativeObject)
public:
    NB_TRAMPOLINE(NativeObject, 1);

    virtual void read_signature(SignatureBuilder* builder) const override { NB_OVERRIDE(read_signature, builder); }
};

/// Base class for a slang reflection type
class NativeSlangType : public Object {
    SGL_OBJECT(NativeSlangType)
public:
    NativeSlangType() = default;

    virtual ~NativeSlangType() {

    };

    /// Get the reflection type.
    ref<TypeReflection> type_reflection() const { return m_type_reflection; }

    /// Set the reflection type.
    void set_type_reflection(const ref<TypeReflection>& reflection) { m_type_reflection = reflection; }

    /// Get the shape of the type.
    const Shape& shape() const { return m_shape; }

    /// Set the shape of the type.
    void set_shape(const Shape& shape) { m_shape = shape; }

    /// Get the element type of the type (call into Python).
    ref<NativeSlangType> element_type() const { return _py_element_type(); }

    /// Check if the type has a derivative (call into Python).
    bool has_derivative() const { return _py_has_derivative(); }

    /// Get the derivative type of the type (call into Python).
    ref<NativeSlangType> derivative() const { return _py_derivative(); }

    /// Get the uniform type layout of the type (call into Python).
    ref<TypeLayoutReflection> uniform_type_layout() const { return _py_uniform_type_layout(); }

    /// Get the buffer type layout of the type (call into Python).
    ref<TypeLayoutReflection> buffer_type_layout() const { return _py_buffer_type_layout(); }

    /// Get string representation of the type.
    std::string to_string() const override;

    /// Virtual accessors to give native system access to python defined reflection properties
    virtual ref<NativeSlangType> _py_element_type() const { return nullptr; }
    virtual bool _py_has_derivative() const { return false; }
    virtual ref<NativeSlangType> _py_derivative() const { return nullptr; }
    virtual ref<TypeLayoutReflection> _py_uniform_type_layout() const { return nullptr; }
    virtual ref<TypeLayoutReflection> _py_buffer_type_layout() const { return nullptr; }

private:
    ref<TypeReflection> m_type_reflection;
    Shape m_shape;
};

/// Nanobind trampoline class for NativeSlangType
struct PyNativeSlangType : public NativeSlangType {
    NB_TRAMPOLINE(NativeSlangType, 5);
    ref<NativeSlangType> _py_element_type() const override { NB_OVERRIDE(_py_element_type); }
    bool _py_has_derivative() const override { NB_OVERRIDE(_py_has_derivative); }
    ref<NativeSlangType> _py_derivative() const override { NB_OVERRIDE(_py_derivative); }
    ref<TypeLayoutReflection> _py_uniform_type_layout() const override { NB_OVERRIDE(_py_uniform_type_layout); }
    ref<TypeLayoutReflection> _py_buffer_type_layout() const override { NB_OVERRIDE(_py_buffer_type_layout); }
};

/// Base class for a marshal to a slangpy supported type.
class NativeMarshall : public Object {
    SGL_OBJECT(NativeMarshall)
public:
    NativeMarshall() = default;

    NativeMarshall(ref<NativeSlangType> slang_type)
        : m_slang_type(std::move(slang_type))
    {
    }

    virtual ~NativeMarshall() = default;

    /// Get the concrete shape of the type. For none-concrete types such as buffers,
    /// this will return an invalid shape.
    Shape concrete_shape() const { return m_concrete_shape; }

    /// Set the concrete shape of the type.
    void set_concrete_shape(const Shape& concrete_shape) { m_concrete_shape = concrete_shape; }

    /// Get whether this type always defines its shape by matching that of the
    /// activate call, rather than by inspecting the value. This is preferred
    /// for some generator types.
    bool match_call_shape() const { return m_match_call_shape; }

    /// Set whether this type matches call shape.
    void set_match_call_shape(bool match_call_shape) { m_match_call_shape = match_call_shape; }

    /// Get the shape of the type (only used if not concrete).
    virtual Shape get_shape(nb::object data) const
    {
        SGL_UNUSED(data);
        return Shape();
    }

    /// Get the slang type.
    ref<NativeSlangType> slang_type() const { return m_slang_type; }

    /// Set the slang type.
    void set_slang_type(const ref<NativeSlangType>& slang_type) { m_slang_type = slang_type; }

    /// Writes call data to a shader cursor before dispatch, optionally writing data for
    /// read back after the kernel has executed. By default, this calls through to
    /// create_calldata, which is typically overridden python side to generate a dictionary.
    virtual void write_shader_cursor_pre_dispatch(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        ShaderCursor cursor,
        nb::object value,
        nb::list read_back
    ) const;

    /// Create call data (uniform values) to be passed to a compute kernel.
    virtual nb::object create_calldata(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data) const
    {
        SGL_UNUSED(context);
        SGL_UNUSED(binding);
        SGL_UNUSED(data);
        return nb::none();
    }

    /// Create dispatch (uniform values) to be passed to a compute kernel in raw dispatch
    virtual nb::object create_dispatchdata(nb::object data) const
    {
        SGL_UNUSED(data);
        return nb::none();
    }

    /// Optionally reads back changes from call data after a kernel has been executed.
    virtual void
    read_calldata(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data, nb::object result) const
    {
        SGL_UNUSED(context);
        SGL_UNUSED(binding);
        SGL_UNUSED(data);
        SGL_UNUSED(result);
    }

    /// For types that can be used to return data from a kernel, allocate
    /// an instance of the correct size to match the call context.
    virtual nb::object create_output(CallContext* context, NativeBoundVariableRuntime* binding) const
    {
        SGL_UNUSED(context);
        SGL_UNUSED(binding);
        return nb::none();
    };

    /// For types that can create_output, read the data allocated in create_output and
    /// return the python object to be provided to the user.
    virtual nb::object read_output(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data) const
    {
        SGL_UNUSED(context);
        SGL_UNUSED(binding);
        SGL_UNUSED(data);
        return nb::none();
    };

    /// Check if the value has a derivative.
    bool has_derivative() const { return false; }

    /// Check if the value is writable.
    bool is_writable() const { return false; }

    /// Code gen only, takes Python types:
    ///   cgb: CodeGenBlock
    ///   context: BindContext
    ///   binding: BoundVariable
    /// Override to generate the code for the uniforms that will represent this value in the kernel.
    virtual void gen_calldata(nb::object cgb, nb::object context, nb::object binding) const
    {
        SGL_UNUSED(cgb);
        SGL_UNUSED(context);
        SGL_UNUSED(binding);
        SGL_THROW("Not implemented");
    }

    /// Code gen only, takes Python types:
    ///   context: BindContext
    /// Override to get the slang type for this variable when a given number of dimensions are removed.
    virtual ref<NativeSlangType> reduce_type(nb::object context, int dimensions) const
    {
        SGL_UNUSED(context);
        SGL_UNUSED(dimensions);
        SGL_THROW("Not implemented");
    }

    /// Code gen only, takes Python types:
    ///   context: BindContext
    /// Return the slang type for this variable when passed to a parameter of the given type.
    /// Default behaviour is to always cast directly to the bound type.
    virtual ref<NativeSlangType> resolve_type(nb::object context, const ref<NativeSlangType> bound_type) const
    {
        SGL_UNUSED(context);
        SGL_UNUSED(bound_type);
        return m_slang_type;
    }

    /// Code gen only, takes Python types:
    ///   context: BindContext
    /// Return a list of slang types for a variable when passed to a parameter of the given type.
    /// Default behaviour is to return a list containing the result of resolve_type.
    virtual std::vector<ref<NativeSlangType>>
    resolve_types(nb::object context, const ref<NativeSlangType> bound_type) const
    {
        SGL_UNUSED(context);
        SGL_UNUSED(bound_type);
        return {resolve_type(context, bound_type)};
    }

    /// Code gen only, takes Python types CodeGenBlock, BindContext, BoundVariable.
    /// Calculate the call dimensionality when this value is passed as a given type.
    virtual int
    resolve_dimensionality(nb::object context, nb::object binding, ref<NativeSlangType> vector_target_type) const
    {
        SGL_UNUSED(context);
        SGL_UNUSED(binding);
        if (!m_slang_type) {
            SGL_THROW("Cannot resolve dimensionality without slang type");
        }
        return static_cast<int>(m_slang_type->shape().size()) - static_cast<int>(vector_target_type->shape().size());
    }

    /// Builds finalized shader object for a value when requested.
    virtual ref<ShaderObject> build_shader_object(nb::object context, nb::object data) const
    {
        SGL_UNUSED(context);
        SGL_UNUSED(data);
        return nullptr;
    }


protected:
    void
    store_readback(NativeBoundVariableRuntime* binding, nb::list& read_back, nb::object value, nb::object data) const;

private:
    Shape m_concrete_shape;
    ref<NativeSlangType> m_slang_type;
    bool m_match_call_shape;
};

/// Nanobind trampoline class for NativeMarshall
struct PyNativeMarshall : public NativeMarshall {
    NB_TRAMPOLINE(NativeMarshall, 15);

    Shape get_shape(nb::object data) const override { NB_OVERRIDE(get_shape, data); }

    void write_shader_cursor_pre_dispatch(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        ShaderCursor cursor,
        nb::object value,
        nb::list read_back
    ) const override
    {
        NB_OVERRIDE(write_shader_cursor_pre_dispatch, context, binding, cursor, value, read_back);
    }

    nb::object
    create_calldata(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data) const override
    {
        NB_OVERRIDE(create_calldata, context, binding, data);
    }

    nb::object create_dispatchdata(nb::object data) const override { NB_OVERRIDE(create_dispatchdata, data); }

    void read_calldata(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        nb::object data,
        nb::object result
    ) const override
    {
        NB_OVERRIDE(read_calldata, context, binding, data, result);
    }

    nb::object create_output(CallContext* context, NativeBoundVariableRuntime* binding) const override
    {
        NB_OVERRIDE(create_output, context, binding);
    }
    nb::object read_output(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data) const override
    {
        NB_OVERRIDE(read_output, context, binding, data);
    }

    void gen_calldata(nb::object cgb, nb::object context, nb::object binding) const override
    {
        NB_OVERRIDE(gen_calldata, cgb, context, binding);
    }

    ref<NativeSlangType> reduce_type(nb::object context, int dimensions) const override
    {
        NB_OVERRIDE(reduce_type, context, dimensions);
    }

    ref<NativeSlangType> resolve_type(nb::object context, const ref<NativeSlangType> bound_type) const override
    {
        NB_OVERRIDE(resolve_type, context, bound_type);
    }

    std::vector<ref<NativeSlangType>>
    resolve_types(nb::object context, const ref<NativeSlangType> bound_type) const override
    {
        NB_OVERRIDE(resolve_types, context, bound_type);
    }

    int resolve_dimensionality(
        nb::object context,
        nb::object binding,
        ref<NativeSlangType> vector_target_type
    ) const override
    {
        NB_OVERRIDE(resolve_dimensionality, context, binding, vector_target_type);
    }

    ref<ShaderObject> build_shader_object(nb::object context, nb::object data) const override
    {
        NB_OVERRIDE(build_shader_object, context, data);
    }
};

/// Binding information that links a python argument to a slang parameter. In
/// the case of structs, this can be nested, mapping python dictionary fields to
/// slang struct fields.
class NativeBoundVariableRuntime : public Object {
    SGL_OBJECT(NativeBoundVariableRuntime)
public:
    NativeBoundVariableRuntime() = default;

    /// Get required access for primal and derivative.
    std::pair<AccessType, AccessType> access() const { return m_access; }

    /// Set required access for primal and derivative.
    void set_access(const std::pair<AccessType, AccessType>& access) { m_access = access; }

    /// Get the call transform.
    const Shape& transform() const { return m_transform; }

    /// Set the call transform.
    void set_transform(const Shape& transform) { m_transform = transform; }

    /// Get the python type marshal.
    ref<NativeMarshall> python_type() const { return m_python_type; }

    /// Set the python type marshal.
    void set_python_type(const ref<NativeMarshall>& python_type) { m_python_type = python_type; }

    /// Get the vector slang type.
    ref<NativeSlangType> vector_type() const { return m_vector_type; }

    /// Set the vector slang type.
    void set_vector_type(ref<NativeSlangType> vector_type) { m_vector_type = vector_type; }

    /// Get the shape being used for the current call.
    Shape get_shape() const { return m_shape; }

    /// Set the shape being used for the current call.
    void set_shape(const Shape& shape) { m_shape = shape; }

    /// Check if this is a parameter block.
    bool is_param_block() const { return m_is_param_block; }

    /// Set if this is a parameter block.
    void set_is_param_block(bool is_param_block) { m_is_param_block = is_param_block; }

    /// Get the uniform variable name.
    std::string_view variable_name() const { return m_variable_name; }

    /// Set the uniform variable name.
    void set_variable_name(std::string_view variable_name) { m_variable_name = variable_name; }

    /// Get children (for structs).
    std::optional<std::map<std::string, ref<NativeBoundVariableRuntime>>> children() const { return m_children; }

    /// Set children (for structs).
    void set_children(const std::optional<std::map<std::string, ref<NativeBoundVariableRuntime>>>& children)
    {
        m_children = children;
    }

    /// Get the call dimensionality.
    int call_dimensionality() const { return m_call_dimensionality; }

    /// Set the call dimensionality.
    void set_call_dimensionality(int call_dimensionality) { m_call_dimensionality = call_dimensionality; }

    /// Whether this variable uses direct binding (raw Slang type, no wrapper).
    bool direct_bind() const { return m_direct_bind; }

    /// Set the direct_bind flag.
    void set_direct_bind(bool direct_bind) { m_direct_bind = direct_bind; }

    /// Recursively populate the overall kernel call shape.
    void populate_call_shape(Shape& call_shape, nb::object value, NativeCallData* error_context);

    /// Write call data to shader cursor before dispatch, optionally writing data for read back after the kernel has
    /// run.
    void
    write_shader_cursor_pre_dispatch(CallContext* context, ShaderCursor cursor, nb::object value, nb::list read_back);

    /// Read back changes from call data after a kernel has been executed by calling read_calldata on the marshal.
    void read_call_data_post_dispatch(CallContext* context, nb::dict call_data, nb::object value);

    /// Read output data from a compute kernel by calling read_output on the marshal.
    nb::object read_output(CallContext* context, nb::object data);

    /// Write uniforms for raw dispatch.
    void write_raw_dispatch_data(nb::dict call_data, nb::object value);

private:
    std::pair<AccessType, AccessType> m_access{AccessType::none, AccessType::none};
    Shape m_transform;
    ref<NativeMarshall> m_python_type;
    Shape m_shape;
    std::string m_variable_name;
    std::optional<std::map<std::string, ref<NativeBoundVariableRuntime>>> m_children;
    int m_call_dimensionality{0};
    ref<NativeSlangType> m_vector_type;
    bool m_is_param_block{false};
    bool m_direct_bind{false};
};

/// Binding information for a call to a compute kernel. Includes a set of positional
/// and keyword arguments as bound variables.
class NativeBoundCallRuntime : Object {
    SGL_OBJECT(NativeBoundCallRuntime)
public:
    NativeBoundCallRuntime() = default;

    /// Get positional arguments.
    const std::vector<ref<NativeBoundVariableRuntime>>& args() const { return m_args; }

    /// Set positional arguments.
    void set_args(const std::vector<ref<NativeBoundVariableRuntime>>& args) { m_args = args; }

    /// Get keyword arguments.
    const std::map<std::string, ref<NativeBoundVariableRuntime>>& kwargs() const { return m_kwargs; }

    /// Set keyword arguments.
    void set_kwargs(const std::map<std::string, ref<NativeBoundVariableRuntime>>& kwargs) { m_kwargs = kwargs; }

    /// Find a keyword argument by name.
    ref<NativeBoundVariableRuntime> find_kwarg(const char* name) const
    {
        auto it = m_kwargs.find(name);
        if (it == m_kwargs.end()) {
            return nullptr;
        }
        return it->second;
    }

    /// Calculate the overall call shape by combining the shapes of all arguments.
    /// args can be nb::list or nb::tuple (nb::args); both support [idx] and nb::len().
    Shape
    calculate_call_shape(int call_dimensionality, nb::object args, nb::dict kwargs, NativeCallData* error_context);

    void write_shader_cursor_pre_dispatch(
        CallContext* context,
        ShaderCursor root_cursor,
        ShaderCursor call_data_cursor,
        nb::object args,
        nb::dict kwargs,
        nb::list read_back
    );

    /// Read back changes from call data after a kernel has been executed by calling read_calldata on the argument
    /// marshals.
    void read_call_data_post_dispatch(CallContext* context, nb::dict call_data, nb::list args, nb::dict kwargs);

    /// Write uniforms for raw dispatch.
    void write_raw_dispatch_data(nb::dict call_data, nb::dict kwargs);

private:
    std::vector<ref<NativeBoundVariableRuntime>> m_args;
    std::map<std::string, ref<NativeBoundVariableRuntime>> m_kwargs;
};

/// Lightweight, stack-allocated runtime options. Single source of truth for all fields.
/// Used directly on the hot dispatch path (no heap allocation).
/// NativeCallRuntimeOptions wraps this for Python binding / ref-counted usage.
struct RuntimeOptions {
    nb::object this_obj; // Default null handle — check with is_valid(), not is_none()
    NativeHandle cuda_stream;
    bool is_ray_tracing{false};
    int thread_count{0};

    // Inline storage for uniforms (most calls have 0-2; heap fallback if more)
    short_vector<nb::object, 4> uniforms;

    bool has_thread_count() const { return thread_count > 0; }
};

/// Ref-counted wrapper around RuntimeOptions for Python binding and torch autograd paths.
class NativeCallRuntimeOptions : Object {
    SGL_OBJECT(NativeCallRuntimeOptions)
public:
    /// Get the uniforms as a Python list (constructs on demand for Python compat).
    nb::list uniforms() const { return m_uniforms; }

    /// Set the uniforms from a Python list.
    void set_uniforms(const nb::list& uniforms) { m_uniforms = uniforms; }

    /// Get this
    nb::object get_this() const { return m_opts.this_obj.is_valid() ? m_opts.this_obj : nb::none(); }

    /// Set this
    void set_this(const nb::object& this_) { m_opts.this_obj = this_; }

    /// Get the CUDA stream.
    NativeHandle cuda_stream() const { return m_opts.cuda_stream; }

    /// Set the CUDA stream.
    void set_cuda_stream(NativeHandle cuda_stream) { m_opts.cuda_stream = cuda_stream; }

    /// Get ray tracing pipeline flag.
    bool is_ray_tracing() const { return m_opts.is_ray_tracing; }

    /// Set ray tracing pipeline flag.
    void set_is_ray_tracing(bool is_ray_tracing) { m_opts.is_ray_tracing = is_ray_tracing; }

    /// Get the thread count override.
    int thread_count() const { return m_opts.thread_count; }

    /// Set the thread count override.
    void set_thread_count(int thread_count) { m_opts.thread_count = thread_count; }

    /// Check if thread count override is set.
    bool has_thread_count() const { return m_opts.has_thread_count(); }

    /// Access the underlying RuntimeOptions.
    RuntimeOptions& opts() { return m_opts; }
    const RuntimeOptions& opts() const { return m_opts; }

    /// Convert from RuntimeOptions (for torch autograd path).
    static ref<NativeCallRuntimeOptions> from_runtime_options(const RuntimeOptions& rt)
    {
        auto opts = make_ref<NativeCallRuntimeOptions>();
        opts->m_opts = rt;
        // Copy uniforms to the Python list for Python-side access
        for (const auto& u : rt.uniforms)
            opts->m_uniforms.append(u);
        return opts;
    }

    /// Clear internal data for garbage collection
    void garbage_collect()
    {
        m_uniforms.clear();
        m_opts.this_obj = nb::object();
    }

private:
    RuntimeOptions m_opts;
    nb::list m_uniforms; // Python-facing list for uniforms (kept in sync via gather_runtime_options)
};

/// Defines the common logging functions for a given log level.
/// The functions are:
/// - name(msg)
/// - name(fmt, ...)
#define SGL_LOG_FUNC_FAMILY(name, level)                                                                               \
    inline void name(const std::string_view msg)                                                                       \
    {                                                                                                                  \
        log(level, msg, LogFrequency::always);                                                                         \
    }                                                                                                                  \
    template<typename... Args>                                                                                         \
    inline void name(fmt::format_string<Args...> fmt, Args&&... args)                                                  \
    {                                                                                                                  \
        log(level, fmt::format(fmt, std::forward<Args>(args)...), LogFrequency::always);                               \
    }

struct CallShapeInfo {
    Shape call_shape = Shape(0, 0);
    Shape strides = Shape(0, 0);
    Shape call_group_shape = Shape(0, 0);
    Shape call_group_strides = Shape(0, 0);
    Shape call_grid_shape = Shape(0, 0);
    Shape call_grid_strides = Shape(0, 0);
    Shape aligned_call_shape = Shape(0, 0);
    bool is_call_shape_unaligned{false};
    int total_threads{0};
};

/// Contains the compute pipeline for a call, the corresponding bindings and any additional
/// options provided by the user.
class NativeCallData : Object {
    SGL_OBJECT(NativeCallData)
public:
    NativeCallData() = default;

    /// Get the device.
    ref<Device> device() const { return m_device; }

    /// Set the device.
    void set_device(const ref<Device>& device) { m_device = device; }

    /// Get the pipeline.
    ref<Pipeline> pipeline() const { return m_pipeline; }

    /// Set the pipeline.
    void set_pipeline(const ref<Pipeline>& pipeline) { m_pipeline = pipeline; }

    /// Get the ray tracing shader table.
    ref<ShaderTable> shader_table() const { return m_shader_table; }

    /// Set the ray tracing shader table.
    void set_shader_table(const ref<ShaderTable>& shader_table) { m_shader_table = shader_table; }

    /// Get the call dimensionality.
    int call_dimensionality() const { return m_call_dimensionality; }

    /// Set the call dimensionality.
    void set_call_dimensionality(int call_dimensionality) { m_call_dimensionality = call_dimensionality; }

    /// Get the runtime bindings.
    ref<NativeBoundCallRuntime> runtime() const { return m_runtime; }

    /// Set the runtime bindings.
    void set_runtime(const ref<NativeBoundCallRuntime>& runtime) { m_runtime = runtime; }

    /// Get the call mode (primitive/forward/backward).
    CallMode call_mode() const { return m_call_mode; }

    /// Set the call mode (primitive/forward/backward).
    void set_call_mode(CallMode call_mode) { m_call_mode = call_mode; }

    /// Get the shape of the last call (useful for debugging).
    const Shape& last_call_shape() const { return m_last_call_shape; }

    /// Get the debug name
    std::string debug_name() const { return m_debug_name; }

    /// Set the debug name
    void set_debug_name(std::string debug_name) { m_debug_name = debug_name; }

    /// Get the logger
    ref<Logger> logger() const { return m_logger; }

    /// Set the logger
    void set_logger(ref<Logger> logger) { m_logger = logger; }

    /// Get torch integration status.
    bool is_torch_integration() const { return m_torch_integration; }

    /// Set torch integration status.
    void set_torch_integration(bool torch_integration) { m_torch_integration = torch_integration; }

    /// Get torch autograd status.
    bool is_torch_autograd() const { return m_torch_autograd; }

    /// Set torch autograd status.
    void set_torch_autograd(bool torch_autograd) { m_torch_autograd = torch_autograd; }

    /// Get whether args need unpacking (have get_this/update_this).
    bool needs_unpack() const { return m_needs_unpack; }

    /// Set whether args need unpacking.
    void set_needs_unpack(bool needs_unpack) { m_needs_unpack = needs_unpack; }

    /// Get whether this call data expects a _thread_count kwarg.
    bool has_thread_count() const { return m_has_thread_count; }

    /// Set whether this call data expects a _thread_count kwarg.
    void set_has_thread_count(bool has_thread_count) { m_has_thread_count = has_thread_count; }

    /// Get whether this call uses direct entry-point parameters (fast path).
    bool use_entrypoint_args() const { return m_use_entrypoint_args; }

    /// Set whether this call uses direct entry-point parameters (fast path).
    void set_use_entrypoint_args(bool use_entrypoint_args) { m_use_entrypoint_args = use_entrypoint_args; }

    /// Get the autograd access list.
    /// This is a flat list of AutogradAccess values precomputed at build time.
    /// At dispatch time, find_torch_tensors steps through this list as it encounters tensors.
    const std::vector<AutogradAccess>& autograd_access_list() const { return m_autograd_access_list; }

    /// Set the autograd access list.
    void set_autograd_access_list(const std::vector<AutogradAccess>& list) { m_autograd_access_list = list; }

    /// Get the cached backward-pass call data (generated on first backward call).
    ref<NativeCallData> bwds_call_data() const { return m_bwds_call_data; }

    /// Set the cached backward-pass call data.
    void set_bwds_call_data(const ref<NativeCallData>& bwds_call_data) { m_bwds_call_data = bwds_call_data; }

    /// Find all torch tensors in args/kwargs, wrap them in NativeTorchTensorDiffPair,
    /// and replace the tensors in args/kwargs with the pairs.
    /// Uses the precomputed autograd_access_list to determine input/output roles.
    /// @param args Mutable list of positional arguments (modified in place).
    /// @param kwargs Mutable dict of keyword arguments (modified in place).
    /// @return List of NativeTorchTensorDiffPair for all found tensors.
    nb::list find_torch_tensors(nb::list args, nb::dict kwargs);

    /// Run the forward kernel and prepare data for autograd.
    /// Returns a tuple of (input_tensors, output_tensors, result, pairs).
    /// - input_tensors: list of primal tensors from input pairs (for save_for_backward)
    /// - output_tensors: list of primal tensors from output pairs (returned to autograd)
    /// - result: the kernel result (or None)
    /// - pairs: updated pairs list (with _result pair appended if needed)
    /// @param opts Runtime options for the call.
    /// @param args Positional arguments (containing NativeTorchTensorDiffPair objects).
    /// @param kwargs Keyword arguments (containing NativeTorchTensorDiffPair objects).
    /// @param pairs List of NativeTorchTensorDiffPair from find_torch_tensors.
    /// @return Tuple of (input_tensors, output_tensors, result, pairs).
    nb::tuple autograd_forward(ref<NativeCallRuntimeOptions> opts, nb::list args, nb::dict kwargs, nb::list pairs);

    /// Run the backward pass: restore tensors, create gradients, call bwds kernel.
    /// Returns tuple of input gradients (matching order of input tensors from forward).
    /// @param function_node The FunctionNode (for calling .bwds).
    /// @param pairs The NativeTorchTensorDiffPair list from forward.
    /// @param args Saved args (with DiffPair placeholders).
    /// @param kwargs Saved kwargs (with DiffPair placeholders).
    /// @param saved_tensors Input tensors from save_for_backward.
    /// @param grad_outputs Upstream gradients for output tensors.
    /// @return Tuple of input gradients.
    nb::tuple autograd_backward(
        nb::handle function_node,
        nb::list pairs,
        nb::list args,
        nb::dict kwargs,
        nb::list saved_tensors,
        nb::tuple grad_outputs
    );

    /// Set the shape of call groups when a dispatch is made.
    void set_call_group_shape(std::optional<Shape> call_group_shape)
    {
        if (call_group_shape.has_value()) {
            m_call_group_shape = call_group_shape.value();
        } else {
            m_call_group_shape = Shape(std::nullopt);
        }
    }

    /// Get the shape of call groups when a dispatch is made.
    const Shape& call_group_shape() const { return m_call_group_shape; }

    /// Call the compute kernel with the provided arguments and keyword arguments.
    nb::object call(RuntimeOptions& opts, nb::args args, nb::kwargs kwargs);

    /// Python-facing overload (unwraps NativeCallRuntimeOptions).
    nb::object call(ref<NativeCallRuntimeOptions> opts, nb::args args, nb::kwargs kwargs);

    /// Append the compute kernel to a command encoder with the provided arguments and keyword arguments.
    nb::object append_to(RuntimeOptions& opts, CommandEncoder* command_encoder, nb::args args, nb::kwargs kwargs);

    /// Python-facing overload (unwraps NativeCallRuntimeOptions).
    nb::object
    append_to(ref<NativeCallRuntimeOptions> opts, CommandEncoder* command_encoder, nb::args args, nb::kwargs kwargs);

    /// Log a message, using either the provided logger or the default logger.
    void log(LogLevel level, const std::string_view msg, LogFrequency frequency = LogFrequency::always)
    {
        if (m_logger)
            m_logger->log(level, msg, frequency);
        else
            sgl::Logger::get().log(level, msg, frequency);
    }

    /// Check if log level is enabled. Used to avoid generating strings native side that would be thrown away.
    bool is_log_enabled(LogLevel level) const
    {
        LogLevel curr_level;
        if (m_logger)
            curr_level = m_logger->level();
        else
            curr_level = sgl::Logger::get().level();
        return level != LogLevel::none && level >= curr_level;
    }

    // Define logging functions.
    SGL_LOG_FUNC_FAMILY(log_debug, LogLevel::debug)
    SGL_LOG_FUNC_FAMILY(log_info, LogLevel::info)
    SGL_LOG_FUNC_FAMILY(log_warn, LogLevel::warn)
    SGL_LOG_FUNC_FAMILY(log_error, LogLevel::error)
    SGL_LOG_FUNC_FAMILY(log_fatal, LogLevel::fatal)

private:
    /// Cached shader offsets for call data fields
    struct CallDataOffsets {
        ShaderOffset call_dim;
        ShaderOffset grid_stride;
        ShaderOffset grid_dim;
        ShaderOffset thread_count;
        ShaderOffset field_offset; // Base offset of the call_data structure (or entry-point)
        uint32_t field_size = 0;   // Total size of the call_data in uniform data
        int array_stride = 0;      // Stride for array elements
        // Cached information for navigating to call_data field (fallback path)
        int32_t call_data_field_index = -1;  // Field index for "call_data" lookup
        bool call_data_is_reference = false; // Whether call_data needs dereference
        bool is_valid = false;               // Whether offsets have been initialized
    };

    ref<Device> m_device;
    ref<Pipeline> m_pipeline;
    ref<ShaderTable> m_shader_table;
    int m_call_dimensionality{0};
    ref<NativeBoundCallRuntime> m_runtime;
    CallMode m_call_mode{CallMode::prim};
    Shape m_last_call_shape;
    std::string m_debug_name;
    ref<Logger> m_logger;
    Shape m_call_group_shape;
    bool m_torch_integration{false};
    bool m_torch_autograd{false};
    bool m_needs_unpack{true};
    bool m_has_thread_count{false};
    bool m_use_entrypoint_args{false};
    std::vector<AutogradAccess> m_autograd_access_list;
    ref<NativeCallData> m_bwds_call_data;
    mutable CallDataOffsets m_cached_call_data_offsets;
    mutable ref<CallContext> m_cached_context;

    /// Recursive helper for find_torch_tensors.
    nb::object find_torch_tensors_recurse(nb::object arg, nb::list& pairs, size_t& access_idx);

    CallShapeInfo
    compute_call_shape_info(RuntimeOptions& opts, const nb::object& unpacked_args, const nb::dict& unpacked_kwargs);

    nb::object exec(RuntimeOptions& opts, CommandEncoder* command_encoder, nb::args args, nb::kwargs kwargs);
};
#undef SGL_LOG_FUNC_FAMILY

typedef std::function<bool(SignatureBuffer& builder, nb::handle)> BuildSignatureFunc;

/// Native side of system for caching call data info for given function signatures.
class NativeCallDataCache : Object {
    SGL_OBJECT(NativeCallDataCache)

public:
    NativeCallDataCache();

    /// Python-facing wrappers (delegate to the SignatureBuffer implementations via SignatureBuilder::buffer()).
    void get_value_signature(const ref<SignatureBuilder> builder, nb::handle o)
    {
        get_value_signature(builder->buffer(), o);
    }

    void get_args_signature(const ref<SignatureBuilder> builder, nb::args args, nb::kwargs kwargs)
    {
        get_args_signature(builder->buffer(), args, kwargs);
    }

    /// Core implementations operating on SignatureBuffer (used by both Python and C++ hot paths).
    void get_value_signature(SignatureBuffer& builder, nb::handle o);
    void get_args_signature(SignatureBuffer& builder, nb::args args, nb::kwargs kwargs);

    /// Transparent string_view lookup (no std::string allocation on cache hit).
    ref<NativeCallData> find_call_data(std::string_view signature)
    {
        auto it = m_cache.find(signature);
        if (it != m_cache.end()) {
            return it->second;
        }
        return nullptr;
    }

    /// Store call data (cache miss only — takes ownership of std::string key).
    void add_call_data(std::string signature, const ref<NativeCallData>& call_data)
    {
        m_cache[std::move(signature)] = call_data;
    }

    virtual std::optional<std::string> lookup_value_signature(nb::handle o)
    {
        SGL_UNUSED(o);
        return std::nullopt;
    }

private:
    std::unordered_map<std::string, ref<NativeCallData>, StringHash, StringEqual> m_cache;
    std::unordered_map<std::type_index, BuildSignatureFunc> m_type_signature_table;
};

class PyNativeCallDataCache : public NativeCallDataCache {
public:
    NB_TRAMPOLINE(NativeCallDataCache, 1);
    std::optional<std::string> lookup_value_signature(nb::handle o) override { NB_OVERRIDE(lookup_value_signature, o); }
};

nb::list unpack_args(nb::args args, bool& out_had_unpack);
nb::dict unpack_kwargs(nb::kwargs kwargs, bool& out_had_unpack);
nb::object unpack_arg(nanobind::object arg, bool& out_had_unpack);
void pack_arg(nb::object arg, nb::object unpacked_arg);

void hash_signature(
    const std::function<std::string(nb::handle)>& value_to_id,
    nb::args args,
    nb::kwargs kwargs,
    const ref<SignatureBuilder>& builder
);

} // namespace sgl::slangpy
