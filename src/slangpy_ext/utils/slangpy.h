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
#include "sgl/device/fwd.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/shader_object.h"
#include "sgl/utils/slangpy.h"

namespace sgl::slangpy {

class NativeBoundVariableRuntime;
class NativeCallData;

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

/// Used during calculation of slangpy signature
class SignatureBuilder : public Object {
public:
    SignatureBuilder()
    {
        m_buffer = m_initial_buffer;
        m_size = 0;
        m_capacity = sizeof(m_initial_buffer);
    }
    ~SignatureBuilder()
    {
        if (m_buffer != m_initial_buffer)
            delete[] m_buffer;
    }

    void add(const std::string& value);
    void add(const char* value);

    template<typename T>
    SignatureBuilder& operator<<(const T& value)
    {
        add(value);
        return *this;
    }

    nb::bytes bytes() const;

    std::string str() const;

    std::string dbg_as_string() const { return std::string((const char*)m_buffer, m_size); }

private:
    uint8_t m_initial_buffer[1024];
    uint8_t* m_buffer;
    size_t m_size;
    size_t m_capacity;

    void add_bytes(const uint8_t* data, size_t size)
    {
        if (m_size + size > m_capacity) {
            m_capacity = std::max(m_capacity * 2, m_size + size);
            uint8_t* new_buffer = new uint8_t[m_capacity];
            memcpy(new_buffer, m_buffer, m_size);
            if (m_buffer != m_initial_buffer)
                delete[] m_buffer;
            m_buffer = new_buffer;
        }
        memcpy(m_buffer + m_size, data, size);
        m_size += size;
    };
};

/// Base class for types that can be passed to a slang function. Use of
/// this is optional, but it is the fastest way to supply signatures
/// to slangpy without entering python code. A user can set a fixed
/// signature on a NativeObject on construction, or override the
/// read_signature function to generate a signature dynamically.
class NativeObject : public Object {
public:
    NativeObject() = default;

    std::string_view slangpy_signature() const { return m_signature; }
    void set_slangpy_signature(std::string_view signature) { m_signature = signature; }

    virtual void read_signature(SignatureBuilder* builder) const { builder->add(m_signature); }


private:
    std::string m_signature;
};

/// Nanobind trampoline class for NativeObject
class PyNativeObject : public NativeObject {
public:
    NB_TRAMPOLINE(NativeObject, 1);

    virtual void read_signature(SignatureBuilder* builder) const { NB_OVERRIDE(read_signature, builder); }
};

/// Base class for a slang reflection type
class NativeSlangType : public Object {
public:
    NativeSlangType() = default;

    virtual ~NativeSlangType(){

    };

    /// Get the reflection type.
    ref<TypeReflection> get_type_reflection() const { return m_type_reflection; }

    /// Set the reflection type.
    void set_type_reflection(const ref<TypeReflection>& reflection) { m_type_reflection = reflection; }

    /// Get the shape of the type.
    Shape get_shape() const { return m_shape; }

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
public:
    NativeMarshall() = default;

    NativeMarshall(ref<NativeSlangType> slang_type)
        : m_slang_type(std::move(slang_type))
    {
    }

    virtual ~NativeMarshall() = default;

    /// Get the concrete shape of the type. For none-concrete types such as buffers,
    /// this will return an invalid shape.
    Shape get_concrete_shape() const { return m_concrete_shape; }

    /// Set the concrete shape of the type.
    void set_concrete_shape(const Shape& concrete_shape) { m_concrete_shape = concrete_shape; }

    /// Get whether this type always defines its shape by matching that of the
    /// activate call, rather than by inspecting the value. This is preferred
    /// for some generator types.
    bool get_match_call_shape() const { return m_match_call_shape; }

    /// Set whether this type matches call shape.
    void set_match_call_shape(bool match_call_shape) { m_match_call_shape = match_call_shape; }

    /// Get the shape of the type (only used if not concrete).
    virtual Shape get_shape(nb::object data) const
    {
        SGL_UNUSED(data);
        return Shape();
    }

    /// Get the slang type.
    ref<NativeSlangType> get_slang_type() const { return m_slang_type; }

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
        return static_cast<int>(m_slang_type->get_shape().size())
            - static_cast<int>(vector_target_type->get_shape().size());
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
    NB_TRAMPOLINE(NativeMarshall, 14);

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

    void read_calldata(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data, nb::object result)
        const override
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

    int resolve_dimensionality(nb::object context, nb::object binding, ref<NativeSlangType> vector_target_type)
        const override
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
public:
    NativeBoundVariableRuntime() = default;

    /// Get required access for primal and derivative.
    std::pair<AccessType, AccessType> get_access() const { return m_access; }

    /// Set required access for primal and derivative.
    void set_access(const std::pair<AccessType, AccessType>& access) { m_access = access; }

    /// Get the call transform.
    const Shape& get_transform() const { return m_transform; }

    /// Set the call transform.
    void set_transform(const Shape& transform) { m_transform = transform; }

    /// Get the python type marshal.
    ref<NativeMarshall> get_python_type() const { return m_python_type; }

    /// Set the python type marshal.
    void set_python_type(const ref<NativeMarshall>& python_type) { m_python_type = python_type; }

    /// Get the vector slang type.
    ref<NativeSlangType> get_vector_type() const { return m_vector_type; }

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
    std::string_view get_variable_name() const { return m_variable_name; }

    /// Set the uniform variable name.
    void set_variable_name(std::string_view variable_name) { m_variable_name = variable_name; }

    /// Get children (for structs).
    std::optional<std::map<std::string, ref<NativeBoundVariableRuntime>>> get_children() const { return m_children; }

    /// Set children (for structs).
    void set_children(const std::optional<std::map<std::string, ref<NativeBoundVariableRuntime>>>& children)
    {
        m_children = children;
    }

    /// Get the call dimensionality.
    int get_call_dimensionality() const { return m_call_dimensionality; }

    /// Set the call dimensionality.
    void set_call_dimensionality(int call_dimensionality) { m_call_dimensionality = call_dimensionality; }

    /// Recursively populate the overall kernel call shape.
    void populate_call_shape(std::vector<int>& call_shape, nb::object value, NativeCallData* error_context);

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
};

/// Binding information for a call to a compute kernel. Includes a set of positional
/// and keyword arguments as bound variables.
class NativeBoundCallRuntime : Object {
public:
    NativeBoundCallRuntime() = default;

    /// Get positional arguments.
    const std::vector<ref<NativeBoundVariableRuntime>>& get_args() const { return m_args; }

    /// Set positional arguments.
    void set_args(const std::vector<ref<NativeBoundVariableRuntime>>& args) { m_args = args; }

    /// Get keyword arguments.
    const std::map<std::string, ref<NativeBoundVariableRuntime>>& get_kwargs() const { return m_kwargs; }

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
    Shape calculate_call_shape(int call_dimensionality, nb::list args, nb::dict kwargs, NativeCallData* error_context);

    void write_shader_cursor_pre_dispatch(
        CallContext* context,
        ShaderCursor root_cursor,
        ShaderCursor call_data_cursor,
        nb::list args,
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

class NativeCallRuntimeOptions : Object {
public:
    /// Get the uniforms.
    nb::list get_uniforms() const { return m_uniforms; }

    /// Set the uniforms.
    void set_uniforms(const nb::list& uniforms) { m_uniforms = uniforms; }

    /// Get this
    nb::object get_this() const { return m_this; }

    /// Set this
    void set_this(const nb::object& this_) { m_this = this_; }

private:
    nb::list m_uniforms;
    nb::object m_this{nb::none()};
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

/// Contains the compute kernel for a call, the corresponding bindings and any additional
/// options provided by the user.
class NativeCallData : Object {
public:
    NativeCallData() = default;

    /// Get the device.
    ref<Device> get_device() const { return m_device; }

    /// Set the device.
    void set_device(const ref<Device>& device) { m_device = device; }

    /// Get the compute kernel.
    ref<ComputeKernel> get_kernel() const { return m_kernel; }

    /// Set the compute kernel.
    void set_kernel(const ref<ComputeKernel>& kernel) { m_kernel = kernel; }

    /// Get the call dimensionality.
    int get_call_dimensionality() const { return m_call_dimensionality; }

    /// Set the call dimensionality.
    void set_call_dimensionality(int call_dimensionality) { m_call_dimensionality = call_dimensionality; }

    /// Get the runtime bindings.
    ref<NativeBoundCallRuntime> get_runtime() const { return m_runtime; }

    /// Set the runtime bindings.
    void set_runtime(const ref<NativeBoundCallRuntime>& runtime) { m_runtime = runtime; }

    /// Get the call mode (primitive/forward/backward).
    CallMode get_call_mode() const { return m_call_mode; }

    /// Set the call mode (primitive/forward/backward).
    void set_call_mode(CallMode call_mode) { m_call_mode = call_mode; }

    /// Get the shape of the last call (useful for debugging).
    const Shape& get_last_call_shape() const { return m_last_call_shape; }

    /// Get the debug name
    std::string get_debug_name() const { return m_debug_name; }

    /// Set the debug name
    void set_debug_name(std::string debug_name) { m_debug_name = debug_name; }

    /// Get the logger
    ref<Logger> get_logger() const { return m_logger; }

    /// Set the logger
    void set_logger(ref<Logger> logger) { m_logger = logger; }

    /// Set the shape of call groups when a dispatch is made.
    void set_call_group_shape(Shape call_group_shape) { m_call_group_shape = call_group_shape; }

    /// Get the shape of call groups when a dispatch is made.
    const Shape& get_call_group_shape() const { return m_call_group_shape; }

    /// Call the compute kernel with the provided arguments and keyword arguments.
    nb::object call(ref<NativeCallRuntimeOptions> opts, nb::args args, nb::kwargs kwargs);

    /// Append the compute kernel to a command encoder with the provided arguments and keyword arguments.
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
    ref<Device> m_device;
    ref<ComputeKernel> m_kernel;
    int m_call_dimensionality{0};
    ref<NativeBoundCallRuntime> m_runtime;
    CallMode m_call_mode{CallMode::prim};
    Shape m_last_call_shape;
    std::string m_debug_name;
    ref<Logger> m_logger;
    Shape m_call_group_shape;

    nb::object
    exec(ref<NativeCallRuntimeOptions> opts, CommandEncoder* command_encoder, nb::args args, nb::kwargs kwargs);
};
#undef SGL_LOG_FUNC_FAMILY

typedef std::function<bool(const ref<SignatureBuilder>& builder, nb::handle)> BuildSignatureFunc;

/// Native side of system for caching call data info for given function signatures.
class NativeCallDataCache : Object {
public:
    NativeCallDataCache();

    void get_value_signature(const ref<SignatureBuilder> builder, nb::handle o);

    void get_args_signature(const ref<SignatureBuilder> builder, nb::args args, nb::kwargs kwargs);

    ref<NativeCallData> find_call_data(const std::string& signature)
    {
        auto it = m_cache.find(signature);
        if (it != m_cache.end()) {
            return it->second;
        }
        return nullptr;
    }

    void add_call_data(const std::string& signature, const ref<NativeCallData>& call_data)
    {
        m_cache[signature] = call_data;
    }

    virtual std::optional<std::string> lookup_value_signature(nb::handle o)
    {
        SGL_UNUSED(o);
        return std::nullopt;
    }

private:
    std::unordered_map<std::string, ref<NativeCallData>> m_cache;
    std::unordered_map<std::type_index, BuildSignatureFunc> m_type_signature_table;
};

class PyNativeCallDataCache : public NativeCallDataCache {
public:
    NB_TRAMPOLINE(NativeCallDataCache, 1);
    std::optional<std::string> lookup_value_signature(nb::handle o) override { NB_OVERRIDE(lookup_value_signature, o); }
};

nb::list unpack_args(nb::args args);
nb::dict unpack_kwargs(nb::kwargs kwargs);
nb::object unpack_arg(nanobind::object arg);
void pack_arg(nb::object arg, nb::object unpacked_arg);

void hash_signature(
    const std::function<std::string(nb::handle)>& value_to_id,
    nb::args args,
    nb::kwargs kwargs,
    const ref<SignatureBuilder>& builder
);

} // namespace sgl::slangpy
