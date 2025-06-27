// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sstream>
#include <cmath>

#include "nanobind.h"

#include "sgl/core/macros.h"
#include "sgl/core/logger.h"
#include "sgl/utils/slangpy.h"
#include "sgl/device/device.h"
#include "sgl/device/kernel.h"
#include "sgl/device/command.h"

#include "utils/slangpy.h"
#include "utils/slangpyvalue.h"
#include "utils/slangpybuffer.h"
#include "utils/slangpypackedarg.h"

namespace sgl {
extern void write_shader_cursor(ShaderCursor& cursor, nb::object value);
extern nb::ndarray<nb::numpy> buffer_to_numpy(Buffer* self);
extern void buffer_copy_from_numpy(Buffer* self, nb::ndarray<nb::numpy> data);
extern nb::ndarray<nb::pytorch, nb::device::cuda>
buffer_to_torch(Buffer* self, DataType type, std::vector<size_t> shape, std::vector<int64_t> strides, size_t offset);

} // namespace sgl

namespace sgl::slangpy {

void SignatureBuilder::add(const std::string& value)
{
    add_bytes((const uint8_t*)value.data(), (int)value.length());
}
void SignatureBuilder::add(const char* value)
{
    add_bytes((const uint8_t*)value, (int)strlen(value));
}

nb::bytes SignatureBuilder::bytes() const
{
    return nb::bytes(m_buffer, m_size);
}

std::string SignatureBuilder::str() const
{
    return std::string(reinterpret_cast<const char*>(m_buffer), m_size);
}

void NativeMarshall::write_shader_cursor_pre_dispatch(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderCursor cursor,
    nb::object value,
    nb::list read_back
) const
{
    // We are a leaf node, so generate and store call data for this node.
    nb::object cd_val = create_calldata(context, binding, value);
    if (!cd_val.is_none()) {
        ShaderCursor child_field = cursor[binding->get_variable_name()];
        write_shader_cursor(child_field, cd_val);
        store_readback(binding, read_back, value, cd_val);
    }
}
void NativeMarshall::store_readback(
    NativeBoundVariableRuntime* binding,
    nb::list& read_back,
    nb::object value,
    nb::object data
) const
{
    read_back.append(nb::make_tuple(binding, value, data));
}

void NativeBoundVariableRuntime::populate_call_shape(
    std::vector<int>& call_shape,
    nb::object value,
    NativeCallData* error_context
)
{
    if (m_children) {
        // We have children, so load each child value and recurse down the tree.
        for (const auto& [name, child_ref] : *m_children) {
            if (child_ref) {
                nb::object child_value = value[name.c_str()];
                child_ref->populate_call_shape(call_shape, child_value, error_context);
            }
        }
    } else if (!value.is_none()) {
        // We are a leaf node, so we can populate the call shape.
        if (!m_transform.valid()) {
            throw NativeBoundVariableException(
                fmt::format("Transform shape is not set for {}. This is an internal error.", m_variable_name),
                ref(this),
                ref(error_context)
            );
        }

        // Read the transform and call shape size.
        const std::vector<int>& tf = m_transform.as_vector();
        size_t csl = call_shape.size();

        // Get the shape of the value. In the case of none-concrete types,
        // only the container shape is needed, as we never map elements.
        // Types that match the call shape simply take their transform
        // and set every corresponding dimension to 1 so it is broadcast.
        if (m_python_type->get_concrete_shape().valid())
            m_shape = m_python_type->get_concrete_shape();
        else if (m_python_type->get_match_call_shape())
            m_shape = Shape(std::vector<int>(tf.size(), 1));
        else {
            NativePackedArg* packed_arg = nullptr;
            auto src_value = value;
            if (nb::try_cast<NativePackedArg*>(value, packed_arg))
                src_value = packed_arg->python_object();
            m_shape = m_python_type->get_shape(src_value);
        }

        // Apply this shape to the overall call shape.
        const std::vector<int>& shape = m_shape.as_vector();
        for (size_t i = 0; i < tf.size(); ++i) {
            int shape_dim = shape[i];
            int call_idx = tf[i];

            // If the call index loaded from the transform is
            // out of bounds, this dimension is a sub-element index,
            // so ignore it.
            if (call_idx >= static_cast<int>(csl)) {
                continue;
            }

            // Apply the new dimension to the call shape.
            //- if it's the same, we're fine
            //- if current call shape == 1, shape_dim != 1, call is expanded
            //- if current call shape != 1, shape_dim == 1, shape is broadcast
            //- if current call shape != 1, shape_dim != 1, it's a mismatch
            int& cs = call_shape[call_idx];
            if (cs != shape_dim) {
                if (cs != 1 && shape_dim != 1) {
                    throw NativeBoundVariableException(
                        fmt::format(
                            "Shape mismatch for {} between value ({}) and call ({})\nThis is typically caused when "
                            "attempting to combine containers with the same dimensionality but different sizes.",
                            m_variable_name,
                            shape_dim,
                            cs
                        ),
                        ref(this),
                        ref(error_context)
                    );
                }
                if (shape_dim != 1) {
                    cs = shape_dim;
                }
            }
        }
    }
}

void NativeBoundVariableRuntime::write_shader_cursor_pre_dispatch(
    CallContext* context,
    ShaderCursor cursor,
    nb::object value,
    nb::list read_back
)
{
    if (is_param_block()) {
        // This variable is represented as a fixed parameter block so just
        // write it straight in.
        auto pb_cursor = cursor[m_variable_name.c_str()];
        write_shader_cursor(pb_cursor, value);
    } else if (m_children) {
        // We have children, so generate call data for each child and
        // store in a dictionary, then store the dictionary as the call data.
        ShaderCursor child_field = cursor[m_variable_name.c_str()];
        for (const auto& [name, child_ref] : *m_children) {
            if (child_ref) {
                nb::object child_value = value[name.c_str()];
                child_ref->write_shader_cursor_pre_dispatch(context, child_field, child_value, read_back);
            }
        }
    } else {
        // We are a leaf node, so generate and store call data for this node.
        m_python_type->write_shader_cursor_pre_dispatch(context, this, cursor, value, read_back);
    }
}

void NativeBoundVariableRuntime::read_call_data_post_dispatch(
    CallContext* context,
    nb::dict call_data,
    nb::object value
)
{
    // Bail if the call data does not contain the variable name.
    if (!call_data.contains(m_variable_name.c_str())) {
        return;
    }

    // Get the call data value.
    auto cd_val = call_data[m_variable_name.c_str()];
    if (m_children) {
        // We have children, so the call data value should be a dictionary
        // containing the call data for each child.
        auto dict = nb::cast<nb::dict>(cd_val);
        for (const auto& [name, child_ref] : *m_children) {
            if (child_ref) {
                nb::object child_value = value[name.c_str()];
                child_ref->read_call_data_post_dispatch(context, dict, child_value);
            }
        }
    } else {
        // We are a leaf node, so the read call data.
        m_python_type->read_calldata(context, this, value, cd_val);
    }
}

void NativeBoundVariableRuntime::write_raw_dispatch_data(nb::dict call_data, nb::object value)
{
    if (m_children) {
        // We have children, so generate call data for each child and
        // store in a dictionary, then store the dictionary as the call data.
        nb::dict cd_val;
        for (const auto& [name, child_ref] : *m_children) {
            if (child_ref) {
                nb::object child_value = value[name.c_str()];
                child_ref->write_raw_dispatch_data(cd_val, child_value);
            }
        }
        if (cd_val.size() > 0) {
            call_data[m_variable_name.c_str()] = cd_val;
        }
    } else {
        // We are a leaf node, so generate and store call data for this node.
        nb::object cd_val = m_python_type->create_dispatchdata(value);
        if (!cd_val.is_none()) {
            call_data[m_variable_name.c_str()] = cd_val;
        }
    }
}

nb::object NativeBoundVariableRuntime::read_output(CallContext* context, nb::object data)
{
    if (m_children) {
        // We have children, so read the output for each child and store in a dictionary.
        nb::dict res;
        for (const auto& [name, child_ref] : *m_children) {
            if (res.contains(name.c_str())) {
                if (child_ref) {
                    nb::object child_data = data[child_ref->m_variable_name.c_str()];
                    res[name.c_str()] = child_ref->read_output(context, child_data);
                }
            }
        }
        return res;
    } else {
        // We are a leaf node, so read the output if the variable was writable.
        if (m_access.first == AccessType::write || m_access.first == AccessType::readwrite) {
            return m_python_type->read_output(context, this, data);
        }
        return nb::none();
    }
}

Shape NativeBoundCallRuntime::calculate_call_shape(
    int call_dimensionality,
    nb::list args,
    nb::dict kwargs,
    NativeCallData* error_context
)
{
    // Setup initial call shape of correct dimensionality, with all dimensions set to 1.
    std::vector<int> call_shape(call_dimensionality, 1);

    // Populate call shape for each positional argument.
    for (size_t idx = 0; idx < args.size(); ++idx) {
        m_args[idx]->populate_call_shape(call_shape, args[idx], error_context);
    }

    // Populate call shape for each keyword argument.
    for (auto [key, value] : kwargs) {
        auto it = m_kwargs.find(nb::str(key).c_str());
        if (it != m_kwargs.end()) {
            it->second->populate_call_shape(call_shape, nb::cast<nb::object>(value), error_context);
        }
    }

    // Return finalized shape.
    return Shape(call_shape);
}

void NativeBoundCallRuntime::write_shader_cursor_pre_dispatch(
    CallContext* context,
    ShaderCursor root_cursor,
    ShaderCursor call_data_cursor,
    nb::list args,
    nb::dict kwargs,
    nb::list read_back

)
{
    // Write call data for each positional argument.
    for (size_t idx = 0; idx < args.size(); ++idx) {
        auto cursor = m_args[idx]->is_param_block() ? root_cursor : call_data_cursor;
        m_args[idx]->write_shader_cursor_pre_dispatch(context, cursor, args[idx], read_back);
    }

    // Write call data for each keyword argument.
    for (auto [key, value] : kwargs) {
        auto it = m_kwargs.find(nb::str(key).c_str());
        if (it != m_kwargs.end()) {
            auto cursor = it->second->is_param_block() ? root_cursor : call_data_cursor;
            it->second->write_shader_cursor_pre_dispatch(context, cursor, nb::cast<nb::object>(value), read_back);
        }
    }
}


void NativeBoundCallRuntime::read_call_data_post_dispatch(
    CallContext* context,
    nb::dict call_data,
    nb::list args,
    nb::dict kwargs
)
{
    // Read call data for each positional argument.
    for (size_t idx = 0; idx < args.size(); ++idx) {
        m_args[idx]->read_call_data_post_dispatch(context, call_data, args[idx]);
    }

    // Read call data for each keyword argument.
    for (auto [key, value] : kwargs) {
        auto it = m_kwargs.find(nb::str(key).c_str());
        if (it != m_kwargs.end()) {
            it->second->read_call_data_post_dispatch(context, call_data, nb::cast<nb::object>(value));
        }
    }
}

void NativeBoundCallRuntime::write_raw_dispatch_data(nb::dict call_data, nb::dict kwargs)
{
    // Write call data for each keyword argument.
    for (auto [key, value] : kwargs) {
        auto it = m_kwargs.find(nb::str(key).c_str());
        if (it != m_kwargs.end()) {
            it->second->write_raw_dispatch_data(call_data, nb::cast<nb::object>(value));
        }
    }
}

nb::object NativeCallData::call(ref<NativeCallRuntimeOptions> opts, nb::args args, nb::kwargs kwargs)
{
    return exec(opts, nullptr, args, kwargs);
}

nb::object NativeCallData::append_to(
    ref<NativeCallRuntimeOptions> opts,
    CommandEncoder* command_encoder,
    nb::args args,
    nb::kwargs kwargs
)
{
    return exec(opts, command_encoder, args, kwargs);
}

nb::object NativeCallData::exec(
    ref<NativeCallRuntimeOptions> opts,
    CommandEncoder* command_encoder,
    nb::args args,
    nb::kwargs kwargs
)
{
    // Unpack args and kwargs.
    nb::list unpacked_args = unpack_args(args);
    nb::dict unpacked_kwargs = unpack_kwargs(kwargs);

    // Calculate call shape.
    Shape call_shape = m_runtime->calculate_call_shape(m_call_dimensionality, unpacked_args, unpacked_kwargs, this);
    m_last_call_shape = call_shape;

    // Setup context.
    auto context = make_ref<CallContext>(m_device, call_shape, m_call_mode);

    // Allocate return value if needed.
    if (!command_encoder && m_call_mode == CallMode::prim) {
        ref<NativeBoundVariableRuntime> rv_node = m_runtime->find_kwarg("_result");
        if (rv_node && (!kwargs.contains("_result") || kwargs["_result"].is_none())) {
            nb::object output = rv_node->get_python_type()->create_output(context, rv_node.get());
            kwargs["_result"] = output;
            unpacked_kwargs["_result"] = output;
            rv_node->populate_call_shape(call_shape.as_vector(), output, this);
        }
    }

    const std::vector<int>& cs = call_shape.as_vector();
    std::vector<int> strides;
    int current_stride = 1;
    for (auto it = cs.rbegin(); it != cs.rend(); ++it) {
        strides.push_back(current_stride);
        current_stride *= *it;
    }
    std::reverse(strides.begin(), strides.end());

    // Get call group shape from build info
    std::vector<int> call_group_shape;

    if (m_call_group_shape.valid() && m_call_group_shape.size() > 0) {
        // Similar to cs, this will be recieved with the first dimension
        // as the right most element, ex: [..., z, y, x].
        call_group_shape = m_call_group_shape.as_vector();

        // Verify that call_group_shape has valid dimensions and values.
        // Our check above should have already validated that
        // call_group_shape.size() > 0.
        if (call_group_shape.size() > cs.size()) {
            throw std::runtime_error(fmt::format(
                "call_group_shape dimensionality ({}) must be <= call_shape dimensionality ({}). "
                "call_group_shape cannot have more dimensions than call_shape.",
                call_group_shape.size(),
                cs.size()
            ));
        } else if (call_group_shape.size() < cs.size()) {
            // Call group shape size is less than the call shape size so we need to
            // pad the call group shape with 1's to account for the missing dimensions.
            // However, inserting at the front of the vector will be inefficient, so
            // log a debug message, giving users a chance to correct their calls.
            if (is_log_enabled(LogLevel::debug)) {
                log_debug(
                    "call_group_shape dimensionality ({}) < call_shape dimensionality ({}). "
                    "Padding call_group_shape with {} leading 1's. "
                    "Consider specifying full dimensions for better performance.",
                    call_group_shape.size(),
                    cs.size(),
                    cs.size() - call_group_shape.size()
                );
            }

            for (size_t i = 0; i < (cs.size() - call_group_shape.size()); ++i) {
                // Insert 1's for the dimensions we were not given
                call_group_shape.insert(call_group_shape.begin(), 1);
            }
        }

        // Verify that all elements of call_group_shape are >= 1
        for (size_t i = 0; i < call_group_shape.size(); ++i) {
            if (call_group_shape[i] < 1) {
                throw std::runtime_error(fmt::format(
                    "call_group_shape[{}] = {} is invalid. All call_group_shape elements must be >= 1.",
                    i,
                    call_group_shape[i]
                ));
            }
        }
    } else {

        // We already know the size/dimensionality of all the shape and
        // stride vectors at this point. Use reserve() to preallocate
        // to the exact size required for efficiency.
        call_group_shape.reserve(cs.size());

        // Default to making the call group shape all 1's. This will force the
        // grid shape to be identical to the call shape giving us linear
        // dispatches by default when a call group shape is not specified.
        // In this case, conceptually all thread's have their own "group" even
        // though they will still be executed in groups of 32.
        for (int i = 0; i < cs.size(); i++) {
            call_group_shape.push_back(1);
        }
    }

    // Calculate the group strides
    short_vector<int, 32> call_group_strides;
    current_stride = 1;
    for (auto it = call_group_shape.rbegin(); it != call_group_shape.rend(); ++it) {
        call_group_strides.push_back(current_stride);
        current_stride *= *it;
    }
    std::reverse(call_group_strides.begin(), call_group_strides.end());

    // Calculate the grid shape and total threads.
    //
    // Note: The call shape may not be call group shape aligned, in which case we
    //       will align up the call shape. This will result in
    //       aligned_call_shape.size - call_shape.size wasted threads. It might be
    //       possible to create some logic to avoid waste, but a call group would
    //       likely end up torn and representing different regions of the call shape,
    //       which would likely defeat the purpose of using call groups for better
    //       memory coherency and uses of shared memory.
    int total_threads = 1;
    short_vector<int, 32> call_grid_shape;
    short_vector<int, 32> aligned_call_shape;
    bool is_call_shape_unaligned = false;
    for (int i = 0; i < cs.size(); i++) {
        // When the call shape is not call group shape aligned, we will add some
        // padding to align up.
        call_grid_shape.push_back((int)std::ceil((float)cs[i] / (float)call_group_shape[i]));
        aligned_call_shape.push_back(call_grid_shape.back() * call_group_shape[i]);
        if (aligned_call_shape[i] != cs[i])
            is_call_shape_unaligned = true;
        total_threads *= aligned_call_shape.back();
    }

    // Calculate the grid strides
    short_vector<int, 32> call_grid_strides;
    current_stride = 1;
    for (auto it = call_grid_shape.end() - 1; it >= call_grid_shape.begin(); --it) {
        call_grid_strides.push_back(current_stride);
        current_stride *= *it;
    }
    std::reverse(call_grid_strides.begin(), call_grid_strides.end());

    nb::list read_back;

    // Dispatch the kernel.
    auto bind_vars = [&](ShaderCursor cursor)
    {
        auto call_data_cursor = cursor.find_field("call_data");

        // Dereference the cursor if it is a reference.
        // We do this here to avoid doing it automatically for every
        // child. Shouldn't need to do recursively as its only
        // relevant for parameter blocks and constant buffers.
        if (call_data_cursor.is_reference())
            call_data_cursor = call_data_cursor.dereference();

        if (!cs.empty()) {
            call_data_cursor["_call_dim"]._set_array_unsafe(&cs[0], cs.size() * 4, cs.size());
            call_data_cursor["_grid_stride"]
                ._set_array_unsafe(&call_grid_strides[0], call_grid_strides.size() * 4, call_grid_strides.size());
            call_data_cursor["_grid_dim"]
                ._set_array_unsafe(&call_grid_shape[0], call_grid_shape.size() * 4, call_grid_shape.size());
        }

        call_data_cursor["_thread_count"] = uint3(total_threads, 1, 1);

        m_runtime->write_shader_cursor_pre_dispatch(
            context,
            cursor,
            call_data_cursor,
            unpacked_args,
            unpacked_kwargs,
            read_back
        );

        nb::list uniforms = opts->get_uniforms();
        if (uniforms) {
            for (auto u : uniforms) {
                if (nb::isinstance<nb::dict>(u)) {
                    write_shader_cursor(cursor, nb::cast<nb::dict>(u));
                } else {
                    write_shader_cursor(cursor, nb::cast<nb::dict>(u(this)));
                }
            }
        }
    };

    if (is_log_enabled(LogLevel::debug)) {
        log_debug("Dispatching {}", m_debug_name);
        log_debug("  Call type: {}", command_encoder ? "append" : "call");
        log_debug("  Call shape: {}", call_shape.to_string());
        log_debug("  Call mode: {}", m_call_mode);
        log_debug("  Strides: [{}]", fmt::join(strides, ", "));
        log_debug("  Call grid shape: [{}]", fmt::join(call_grid_shape, ", "));
        log_debug("  Call grid strides: [{}]", fmt::join(call_grid_strides, ", "));
        log_debug("  Call group shape: [{}]", fmt::join(call_group_shape, ", "));
        log_debug("  Call group strides: [{}]", fmt::join(call_group_strides, ", "));
        if (is_call_shape_unaligned) {
            log_debug("  Call shape was not aligned to the given call group shape");
            log_debug("  and has been padded up as a reult. Note that this will");
            log_debug("  result in wasted threads.");
            log_debug("  Aligned call shape: [{}]", fmt::join(aligned_call_shape, ", "));
        }
        log_debug("  Threads: {}", total_threads);
    }

    m_kernel->dispatch(uint3(total_threads, 1, 1), bind_vars, command_encoder);

    // If command_buffer is not null, return early.
    if (command_encoder != nullptr) {
        return nanobind::none();
    }

    // Read call data post dispatch.
    // m_runtime->read_call_data_post_dispatch(context, call_data, unpacked_args, unpacked_kwargs);
    for (auto val : read_back) {
        auto t = nb::cast<nb::tuple>(val);
        auto bvr = nb::cast<ref<NativeBoundVariableRuntime>>(t[0]);
        auto rb_val = t[1];
        auto rb_data = t[2];
        bvr->get_python_type()->read_calldata(context, bvr.get(), rb_val, rb_data);
    }

    // Pack updated 'this' values back.
    for (size_t i = 0; i < args.size(); ++i) {
        pack_arg(args[i], unpacked_args[i]);
    }
    for (auto [k, v] : kwargs) {
        pack_arg(nb::cast<nb::object>(v), unpacked_kwargs[k]);
    }

    // Handle return value based on call mode.
    if (m_call_mode == CallMode::prim) {
        auto rv_node_it = m_runtime->find_kwarg("_result");
        if (rv_node_it && !unpacked_kwargs["_result"].is_none()) {
            return rv_node_it->read_output(context, unpacked_kwargs["_result"]);
        }
    }
    return nb::none();
}

NativeCallDataCache::NativeCallDataCache()
{
    m_cache.reserve(1024);

    m_type_signature_table[typeid(Texture)] = [](const ref<SignatureBuilder>& builder, nb::handle o)
    {
        auto tex = nb::cast<Texture*>(o);

        // Note: Using snprintf here as fmt library is quite
        // a bit slower for this use case. (over 4x).
        char temp[256];
        std::snprintf(
            temp,
            sizeof(temp),
            "[%d,%d,%d,%d]",
            (int)tex->desc().type,
            (int)tex->desc().usage,
            (int)tex->desc().format,
            (int)tex->desc().array_length
        );
        builder->add(temp);

        return true;
    };

    m_type_signature_table[typeid(Buffer)] = [](const ref<SignatureBuilder>& builder, nb::handle o)
    {
        auto buffer = nb::cast<Buffer*>(o);

        // Note: Using snprintf here as fmt library is quite
        // a bit slower for this use case. (over 4x).
        char temp[256];
        std::snprintf(temp, sizeof(temp), "[%d]", (int)buffer->desc().usage);
        builder->add(temp);

        return true;
    };
}

void NativeCallDataCache::get_value_signature(const ref<SignatureBuilder> builder, nb::handle o)
{
    // Get python type.
    auto type = o.type();

    // Check if this is a bound native type, in which case we can hopefully do fast things!
    bool is_bound_type = nb::type_check(type);
    if (is_bound_type) {

        // Get C++ type info, and attempt to cast to a slangpy native object
        const auto& type_info = nb::type_info(type);

        // If we have a native object, can directly request the signature.
        const NativeObject* native_object;
        if (nb::try_cast<const NativeObject*>(o, native_object)) {
            *builder << type_info.name() << "\n";
            native_object->read_signature(builder);
            return;
        }

        // Attempt to use type signature table to lookup type
        auto it = m_type_signature_table.find(type_info);
        if (it != m_type_signature_table.end()) {
            if (it->second(builder, o)) {
                return;
            }
        }
    }

    // Fast path for basic Python types (int/float) here.
    if (nb::isinstance<int>(o)) {
        *builder << "int\n";
        return;
    }
    if (nb::isinstance<float>(o)) {
        *builder << "float\n";
        return;
    }
    if (nb::isinstance<bool>(o)) {
        *builder << "bool\n";
        return;
    }
    if (nb::isinstance<nb::str>(o)) {
        *builder << "string\n";
        return;
    }

    // Python tuple/list
    nb::tuple tuple;
    if (nb::try_cast<nb::tuple>(o, tuple)) {
        *builder << "tuple\n";
        for (const auto& i : tuple) {
            get_value_signature(builder, i);
        }
        return;
    }
    nb::list list;
    if (nb::try_cast<nb::list>(o, list)) {
        *builder << "list\n";
        for (const auto& i : list) {
            get_value_signature(builder, i);
        }
        return;
    }

    // Add type name.
    auto type_name = nb::str(nb::getattr(o.type(), "__name__"));
    *builder << type_name.c_str() << "\n";

    // Handle objects with get_this method.
    auto get_this = nb::getattr(o, "get_this", nb::none());
    if (!get_this.is_none()) {
        auto this_ = get_this();
        get_value_signature(builder, this_);
        return;
    }

    // If x has signature attribute, use it.
    if (nb::hasattr(o, "slangpy_signature")) {

        auto slangpy_sig = nb::getattr(o, "slangpy_signature");
        *builder << nb::str(slangpy_sig).c_str() << "\n";
        return;
    }

    // If x is a dictionary get signature of its children.
    nb::dict dict;
    if (nb::try_cast(o, dict)) {
        *builder << "\n";
        for (const auto& [k, v] : dict) {
            nb::str key(k);
            *builder << key.c_str() << ":";

            nb::str _type;
            if (strcmp(key.c_str(), "_type") == 0 && nb::try_cast<nb::str>(v, _type)) {
                // If the dictionary contains a _type key with string value,
                // we have to encode the value directly, as it affects type resolution
                *builder << _type.c_str() << "\n";
            } else {
                get_value_signature(builder, v);
            }
        }
        return;
    }

    // Use value_to_id function.
    std::optional<std::string> s = lookup_value_signature(o);
    if (s.has_value()) {
        *builder << *s;
    }
    *builder << "\n";
}

void NativeCallDataCache::get_args_signature(const ref<SignatureBuilder> builder, nb::args args, nb::kwargs kwargs)
{
    builder->add("args\n");
    for (const auto& arg : args) {
        builder->add("N:");
        get_value_signature(builder, arg);
    }

    builder->add("kwargs\n");
    for (const auto& [k, v] : kwargs) {
        builder->add(nb::str(k).c_str());
        builder->add(":");
        get_value_signature(builder, v);
    }
}

nb::list unpack_args(nb::args args)
{
    nb::list unpacked;
    for (auto arg : args) {
        unpacked.append(unpack_arg(nb::cast<nb::object>(arg)));
    }
    return unpacked;
}

nb::dict unpack_kwargs(nb::kwargs kwargs)
{
    nb::dict unpacked;
    for (const auto& [k, v] : kwargs) {
        unpacked[k] = unpack_arg(nb::cast<nb::object>(v));
    }
    return unpacked;
}

nb::object unpack_arg(nb::object arg)
{
    auto obj = arg;

    // If object has 'get_this', read it.
    if (nb::hasattr(obj, "get_this")) {
        obj = nb::getattr(obj, "get_this")();
    }

    // Recursively unpack dictionaries.
    nb::dict d;
    if (nb::try_cast(obj, d)) {
        nb::dict res;
        for (auto [k, v] : d) {
            res[k] = unpack_arg(nb::cast<nb::object>(v));
        }
        obj = res;
    }

    // Recursively unpack lists.
    nb::list l;
    if (nb::try_cast(obj, l)) {
        nb::list res;
        for (auto v : l) {
            res.append(unpack_arg(nb::cast<nb::object>(v)));
        }
        obj = res;
    }

    // Return unpacked object.
    return obj;
}

void pack_arg(nanobind::object arg, nanobind::object unpacked_arg)
{
    // If object has 'update_this', update it.
    if (nb::hasattr(arg, "update_this")) {
        nb::getattr(arg, "update_this")(unpacked_arg);
    }

    // Recursively pack dictionaries.
    nb::dict d;
    if (nb::try_cast(arg, d)) {
        for (auto [k, v] : d) {
            pack_arg(nb::cast<nb::object>(v), unpacked_arg[k]);
        }
    }

    // Recursively pack lists.
    nb::list l;
    if (nb::try_cast(arg, l)) {
        for (size_t i = 0; i < l.size(); ++i) {
            pack_arg(l[i], unpacked_arg[i]);
        }
    }
}

// Helper to get signature of a single value.
std::string get_value_signature(nb::handle o)
{
    static NativeCallDataCache cache;
    auto builder = make_ref<SignatureBuilder>();
    cache.get_value_signature(builder, o);
    return builder->str();
}

} // namespace sgl::slangpy

SGL_PY_EXPORT(utils_slangpy)
{
    using namespace sgl;
    using namespace sgl::slangpy;

    nb::module_ slangpy = nb::module_::import_("slangpy.slangpy");

    nb::sgl_enum<AccessType>(slangpy, "AccessType");
    nb::sgl_enum<CallMode>(slangpy, "CallMode");

    slangpy.def(
        "unpack_args",
        [](nb::args args) { return unpack_args(args); },
        "args"_a,
        D_NA(slangpy, unpack_args)
    );
    slangpy.def(
        "unpack_kwargs",
        [](nb::kwargs kwargs) { return unpack_kwargs(kwargs); },
        "kwargs"_a,
        D_NA(slangpy, unpack_kwargs)
    );
    slangpy.def(
        "unpack_arg",
        [](nb::object arg) { return unpack_arg(arg); },
        "arg"_a,
        D_NA(slangpy, unpack_arg)
    );
    slangpy.def(
        "pack_arg",
        [](nb::object arg, nb::object unpacked_arg) { pack_arg(arg, unpacked_arg); },
        "arg"_a,
        "unpacked_arg"_a,
        D_NA(slangpy, pack_arg)
    );
    slangpy.def("get_value_signature", &get_value_signature, "o"_a, D_NA(slangpy, get_value_signature));

    nb::register_exception_translator(
        [](const std::exception_ptr& p, void* /* unused */)
        {
            try {
                std::rethrow_exception(p);
            } catch (const NativeBoundVariableException& e) {
                nb::dict data;
                data["message"] = e.message();
                data["source"] = e.source();
                data["context"] = e.context();
                PyErr_SetObject(PyExc_ValueError, data.ptr());
            }
        }
    );

    nb::class_<SignatureBuilder, Object>(slangpy, "SignatureBuilder") //
        .def(nb::init<>(), D_NA(SignatureBuilder, SignatureBuilder))
        .def("add", nb::overload_cast<const std::string&>(&SignatureBuilder::add), "value"_a, D_NA(NativeObject, add))
        .def_prop_ro("str", &SignatureBuilder::str, D_NA(SignatureBuilder, str))
        .def_prop_ro(
            "bytes",
            &SignatureBuilder::bytes,
            nb::rv_policy::reference_internal,
            D_NA(SignatureBuilder, bytes)
        );

    nb::class_<NativeObject, PyNativeObject, Object>(slangpy, "NativeObject") //
        .def(
            "__init__",
            [](NativeObject& self) { new (&self) PyNativeObject(); },
            D_NA(NativeObject, NativeObject)
        )
        .def_prop_rw("slangpy_signature", &NativeObject::slangpy_signature, &NativeObject::set_slangpy_signature)
        .def("read_signature", &NativeObject::read_signature, "builder"_a, D_NA(NativeObject, read_signature));

    nb::class_<NativeSlangType, PyNativeSlangType, Object>(slangpy, "NativeSlangType") //
        .def(
            "__init__",
            [](NativeSlangType& self) { new (&self) PyNativeSlangType(); },
            D_NA(NativeSlangType, NativeSlangType)
        )
        .def_prop_rw(
            "type_reflection",
            &NativeSlangType::get_type_reflection,
            &NativeSlangType::set_type_reflection,
            D_NA(NativeSlangType, type_reflection)
        )
        .def_prop_rw("shape", &NativeSlangType::get_shape, &NativeSlangType::set_shape, D_NA(NativeSlangType, shape))
        .def("_py_element_type", &NativeSlangType::_py_element_type)
        .def("_py_has_derivative", &NativeSlangType::_py_has_derivative)
        .def("_py_derivative", &NativeSlangType::_py_derivative)
        .def("_py_uniform_type_layout", &NativeSlangType::_py_uniform_type_layout)
        .def("_py_buffer_type_layout", &NativeSlangType::_py_buffer_type_layout);

    nb::class_<NativeMarshall, PyNativeMarshall, Object>(slangpy, "NativeMarshall") //
        .def(
            "__init__",
            [](NativeMarshall& self) { new (&self) PyNativeMarshall(); },
            D_NA(NativeMarshall, NativeMarshall)
        )

        .def_prop_rw(
            "concrete_shape",
            &NativeMarshall::get_concrete_shape,
            &NativeMarshall::set_concrete_shape,
            D_NA(NativeMarshall, concrete_shape)
        )
        .def_prop_rw(
            "match_call_shape",
            &NativeMarshall::get_match_call_shape,
            &NativeMarshall::set_match_call_shape,
            D_NA(NativeMarshall, match_call_shape)
        )
        .def("get_shape", &NativeMarshall::get_shape, "value"_a, D_NA(NativeMarshall, get_shape))
        .def_prop_rw(
            "slang_type",
            &NativeMarshall::get_slang_type,
            &NativeMarshall::set_slang_type,
            D_NA(NativeMarshall, slang_type)
        )
        .def(
            "write_shader_cursor_pre_dispatch",
            &NativeMarshall::write_shader_cursor_pre_dispatch,
            "context"_a,
            "binding"_a,
            "cursor"_a,
            "value"_a,
            "read_back"_a,
            D_NA(NativeMarshall, write_shader_cursor_pre_dispatch)
        )
        .def("create_calldata", &NativeMarshall::create_calldata, D_NA(NativeMarshall, create_calldata))
        .def("read_calldata", &NativeMarshall::read_calldata, D_NA(NativeMarshall, read_calldata))
        .def("create_output", &NativeMarshall::create_output, D_NA(NativeMarshall, create_output))
        .def("read_output", &NativeMarshall::read_output, D_NA(NativeMarshall, read_output))
        .def_prop_ro("has_derivative", &NativeMarshall::has_derivative, D_NA(NativeMarshall, has_derivative))
        .def_prop_ro("is_writable", &NativeMarshall::is_writable, D_NA(NativeMarshall, is_writable))
        .def(
            "gen_calldata",
            &NativeMarshall::gen_calldata,
            "cgb"_a,
            "context"_a,
            "binding"_a,
            D_NA(NativeMarshall, gen_calldata)
        )
        .def(
            "reduce_type",
            &NativeMarshall::reduce_type,
            "context"_a,
            "dimensions"_a,
            D_NA(NativeMarshall, reduce_type)
        )
        .def(
            "resolve_type",
            &NativeMarshall::resolve_type,
            "context"_a,
            "bound_type"_a,
            D_NA(NativeMarshall, resolve_type)
        )
        .def(
            "resolve_dimensionality",
            &NativeMarshall::resolve_dimensionality,
            "context"_a,
            "binding"_a,
            "vector_target_type"_a,
            D_NA(NativeMarshall, resolve_dimensionality)
        )
        .def(
            "build_shader_object",
            &NativeMarshall::build_shader_object,
            "context"_a,
            "data"_a,
            D_NA(NativeMarshall, build_shader_object)
        );

    nb::class_<NativeBoundVariableRuntime, Object>(slangpy, "NativeBoundVariableRuntime") //
        .def(nb::init<>(), D_NA(NativeBoundVariableRuntime, NativeBoundVariableRuntime))
        .def_prop_rw(
            "access",
            &NativeBoundVariableRuntime::get_access,
            &NativeBoundVariableRuntime::set_access,
            D_NA(NativeBoundVariableRuntime, access)
        )
        .def_prop_rw(
            "transform",
            &NativeBoundVariableRuntime::get_transform,
            &NativeBoundVariableRuntime::set_transform,
            D_NA(NativeBoundVariableRuntime, transform)
        )
        .def_prop_rw(
            "python_type",
            &NativeBoundVariableRuntime::get_python_type,
            &NativeBoundVariableRuntime::set_python_type,
            D_NA(NativeBoundVariableRuntime, python_type)
        )
        .def_prop_rw(
            "vector_type",
            &NativeBoundVariableRuntime::get_vector_type,
            &NativeBoundVariableRuntime::set_vector_type,
            D_NA(NativeBoundVariableRuntime, vector_type)
        )
        .def_prop_rw(
            "shape",
            &NativeBoundVariableRuntime::get_shape,
            &NativeBoundVariableRuntime::set_shape,
            D_NA(NativeBoundVariableRuntime, shape)
        )
        .def_prop_rw(
            "is_param_block",
            &NativeBoundVariableRuntime::is_param_block,
            &NativeBoundVariableRuntime::set_is_param_block,
            D_NA(NativeBoundVariableRuntime, is_param_block)
        )
        .def_prop_rw(
            "variable_name",
            &NativeBoundVariableRuntime::get_variable_name,
            &NativeBoundVariableRuntime::set_variable_name,
            D_NA(NativeBoundVariableRuntime, variable_name)
        )
        .def_prop_rw(
            "children",
            &NativeBoundVariableRuntime::get_children,
            &NativeBoundVariableRuntime::set_children,
            D_NA(NativeBoundVariableRuntime, children)
        )
        .def(
            "populate_call_shape",
            &NativeBoundVariableRuntime::populate_call_shape,
            D_NA(NativeBoundVariableRuntime, populate_call_shape)
        )
        .def(
            "read_call_data_post_dispatch",
            &NativeBoundVariableRuntime::read_call_data_post_dispatch,
            D_NA(NativeBoundVariableRuntime, read_call_data_post_dispatch)
        )
        .def(
            "write_raw_dispatch_data",
            &NativeBoundVariableRuntime::write_raw_dispatch_data,
            D_NA(NativeBoundVariableRuntime, write_raw_dispatch_data)
        )
        .def("read_output", &NativeBoundVariableRuntime::read_output, D_NA(NativeBoundVariableRuntime, read_output));

    nb::class_<NativeBoundCallRuntime, Object>(slangpy, "NativeBoundCallRuntime") //
        .def(nb::init<>(), D_NA(NativeBoundCallRuntime, NativeBoundCallRuntime))
        .def_prop_rw(
            "args",
            &NativeBoundCallRuntime::get_args,
            &NativeBoundCallRuntime::set_args,
            D_NA(NativeBoundCallRuntime, args)
        )
        .def_prop_rw(
            "kwargs",
            &NativeBoundCallRuntime::get_kwargs,
            &NativeBoundCallRuntime::set_kwargs,
            D_NA(NativeBoundCallRuntime, kwargs)
        )
        .def("find_kwarg", &NativeBoundCallRuntime::find_kwarg, D_NA(NativeBoundCallRuntime, find_kwarg))
        .def(
            "calculate_call_shape",
            &NativeBoundCallRuntime::calculate_call_shape,
            D_NA(NativeBoundCallRuntime, calculate_call_shape)
        )
        .def(
            "read_call_data_post_dispatch",
            &NativeBoundCallRuntime::read_call_data_post_dispatch,
            D_NA(NativeBoundCallRuntime, read_call_data_post_dispatch)
        )
        .def(
            "write_raw_dispatch_data",
            &NativeBoundCallRuntime::write_raw_dispatch_data,
            D_NA(NativeBoundCallRuntime, write_raw_dispatch_data)
        );

    nb::class_<NativeCallRuntimeOptions, Object>(slangpy, "NativeCallRuntimeOptions") //
        .def(nb::init<>(), D_NA(NativeCallRuntimeOptions, NativeCallRuntimeOptions))
        .def_prop_rw(
            "uniforms",
            &NativeCallRuntimeOptions::get_uniforms,
            &NativeCallRuntimeOptions::set_uniforms,
            D_NA(NativeCallRuntimeOptions, uniforms)
        );

    // clang-format off
#define DEF_LOG_METHOD(name) def(#name, [](NativeCallData& self, const std::string_view msg) { self.name(msg); }, "msg"_a)
    // clang-format on

    nb::class_<NativeCallData, Object>(slangpy, "NativeCallData") //
        .def(nb::init<>(), D_NA(NativeCallData, NativeCallData))
        .def_prop_rw("device", &NativeCallData::get_device, &NativeCallData::set_device, D_NA(NativeCallData, device))
        .def_prop_rw("kernel", &NativeCallData::get_kernel, &NativeCallData::set_kernel, D_NA(NativeCallData, kernel))
        .def_prop_rw(
            "call_dimensionality",
            &NativeCallData::get_call_dimensionality,
            &NativeCallData::set_call_dimensionality,
            D_NA(NativeCallData, call_dimensionality)
        )
        .def_prop_rw(
            "runtime",
            &NativeCallData::get_runtime,
            &NativeCallData::set_runtime,
            D_NA(NativeCallData, runtime)
        )
        .def_prop_rw(
            "call_mode",
            &NativeCallData::get_call_mode,
            &NativeCallData::set_call_mode,
            D_NA(NativeCallData, call_mode)
        )
        .def_prop_ro("last_call_shape", &NativeCallData::get_last_call_shape, D_NA(NativeCallData, last_call_shape))
        .def_prop_rw(
            "debug_name",
            &NativeCallData::get_debug_name,
            &NativeCallData::set_debug_name,
            D_NA(NativeCallData, debug_name)
        )
        .def_prop_rw(
            "logger",
            &NativeCallData::get_logger,
            &NativeCallData::set_logger,
            nb::arg().none(),
            D_NA(NativeCallData, logger)
        )
        .def(
            "call",
            &NativeCallData::call,
            nb::arg("opts"),
            nb::arg("args"),
            nb::arg("kwargs"),
            D_NA(NativeCallData, call)
        )
        .def(
            "append_to",
            &NativeCallData::append_to,
            nb::arg("opts"),
            nb::arg("command_buffer"),
            nb::arg("args"),
            nb::arg("kwargs"),
            D_NA(NativeCallData, append_to)
        )
        .def_prop_rw(
            "call_group_shape",
            &NativeCallData::get_call_group_shape,
            &NativeCallData::set_call_group_shape,
            nb::arg().none(),
            D_NA(NativeCallData, call_group_shape)
        )
        .def("log", &NativeCallData::log, "level"_a, "msg"_a, "frequency"_a = LogFrequency::always, D(Logger, log))
        .DEF_LOG_METHOD(log_debug)
        .DEF_LOG_METHOD(log_info)
        .DEF_LOG_METHOD(log_warn)
        .DEF_LOG_METHOD(log_error)
        .DEF_LOG_METHOD(log_fatal);

#undef DEF_LOG_METHOD

    nb::class_<NativeCallDataCache, PyNativeCallDataCache, Object>(slangpy, "NativeCallDataCache")
        .def(
            "__init__",
            [](NativeCallDataCache& self) { new (&self) PyNativeCallDataCache(); },
            D_NA(NativeCallDataCache, NativeCallDataCache)
        )
        .def(
            "get_value_signature",
            &NativeCallDataCache::get_value_signature,
            "builder"_a,
            "o"_a,
            D_NA(NativeCallDataCache, get_value_signature)
        )
        .def(
            "get_args_signature",
            &NativeCallDataCache::get_args_signature,
            "builder"_a,
            "args"_a,
            "kwargs"_a,
            D_NA(NativeCallDataCache, get_args_signature)
        )
        .def(
            "find_call_data",
            &NativeCallDataCache::find_call_data,
            "signature"_a,
            D_NA(NativeCallDataCache, find_call_data)
        )
        .def(
            "add_call_data",
            &NativeCallDataCache::add_call_data,
            "signature"_a,
            "call_data"_a,
            D_NA(NativeCallDataCache, add_call_data)
        )
        .def(
            "lookup_value_signature",
            &NativeCallDataCache::lookup_value_signature,
            "o"_a,
            D_NA(NativeCallDataCache, lookup_value_signature)
        );


    nb::class_<Shape>(slangpy, "Shape") //
        .def(
            "__init__",
            [](Shape& self, nb::args args)
            {
                if (args.size() == 0) {
                    new (&self) Shape(std::vector<int>());
                } else if (args.size() == 1) {
                    if (args[0].is_none()) {
                        new (&self) Shape(std::nullopt);
                    } else if (nb::isinstance<nb::tuple>(args[0])) {
                        new (&self) Shape(nb::cast<std::vector<int>>(args[0]));
                    } else if (nb::isinstance<Shape>(args[0])) {
                        new (&self) Shape(nb::cast<Shape>(args[0]));
                    } else {
                        new (&self) Shape(nb::cast<std::vector<int>>(args));
                    }
                } else {
                    new (&self) Shape(nb::cast<std::vector<int>>(args));
                }
            },
            "args"_a,
            D_NA(Shape, Shape)
        )
        .def(
            "__add__",
            [](const Shape& self, const Shape& other) { return self + other; },
            nb::is_operator(),
            D_NA(Shape, operator+)
        )
        .def(
            "__getitem__",
            [](const Shape& self, size_t i) -> int
            {
                if (i >= self.size())
                    throw nb::index_error(); // throwing index_error allows this to be used as a python iterator
                return self[i];
            },
            nb::arg("index"),
            D_NA(Shape, operator[])
        )
        .def("__len__", &Shape::size, D_NA(Shape, size))
        .def_prop_ro("valid", &Shape::valid, D_NA(Shape, valid))
        .def_prop_ro("concrete", &Shape::concrete, D_NA(Shape, concrete))
        .def(
            "as_tuple",
            [](Shape& self)
            {
                std::vector<int>& v = self.as_vector();
                nb::list py_list;
                for (const int& item : v) {
                    py_list.append(item);
                }
                return nb::tuple(py_list);
            },
            D_NA(Shape, as_tuple)
        )
        .def(
            "as_list",
            [](Shape& self) { return self.as_vector(); },
            nb::rv_policy::reference_internal,
            D_NA(Shape, as_list)
        )
        .def("calc_contiguous_strides", &Shape::calc_contiguous_strides, D_NA(Shape, calc_contiguous_strides))
        .def("__repr__", &Shape::to_string, D_NA(Shape, to_string))
        .def("__str__", &Shape::to_string, D_NA(Shape, to_string))
        .def(
            "__eq__",
            [](const Shape& self, nb::object other)
            {
                if (nb::isinstance<Shape>(other)) {
                    return self.as_vector() == nb::cast<Shape>(other).as_vector();
                }

                std::vector<int> v;
                if (nb::try_cast(other, v)) {
                    return self.as_vector() == v;
                }

                return false;
            },
            D_NA(Shape, operator==)
        );

    nb::class_<CallContext, Object>(slangpy, "CallContext") //
        .def(
            nb::init<ref<Device>, const Shape&, CallMode>(),
            nb::arg("device"),
            nb::arg("call_shape"),
            nb::arg("call_mode"),
            D_NA(CallContext, CallContext)
        )
        .def_prop_ro(
            "device",
            [](const CallContext& self) -> Device* { return self.device(); },
            D_NA(CallContext, device)
        )
        .def_prop_ro(
            "call_shape",
            &CallContext::call_shape,
            nb::rv_policy::reference_internal,
            D_NA(CallContext, call_shape)
        )
        .def_prop_ro("call_mode", &CallContext::call_mode, D_NA(CallContext, call_mode));
}
