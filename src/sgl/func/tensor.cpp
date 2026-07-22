// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/func/tensor.h"

#include "sgl/core/signature_buffer.h"
#include "sgl/device/buffer_cursor.h"
#include "sgl/device/command.h"
#include "sgl/device/device.h"
#include "sgl/device/shader_cursor.h"

#include <fmt/format.h>

#include <string_view>

namespace sgl::func {

namespace {

    template<typename TCursor>
    std::string_view cursor_type_name(const TCursor& cursor)
    {
        slang::TypeLayoutReflection* layout = cursor.slang_type_layout();
        if (!layout)
            return {};

        const char* name = layout->getName();
        return name ? std::string_view(name) : std::string_view{};
    }

    const Tensor* select_difftensor_view_gradient(const Tensor& tensor)
    {
        const ref<Tensor>& grad_in = tensor.grad_in();
        const ref<Tensor>& grad_out = tensor.grad_out();

        if (grad_in && grad_out && grad_in != grad_out) {
            SGL_THROW("DiffTensorView has a single diff field, but Tensor has distinct grad_in and grad_out tensors.");
        }

        if (grad_out)
            return grad_out.get();
        if (grad_in)
            return grad_in.get();

        SGL_THROW("Missing required tensor gradient for DiffTensorView target.");
    }

    template<typename TCursor>
    bool is_tensor_view_target(const TCursor& cursor)
    {
        std::string_view type_name = cursor_type_name(cursor);
        return type_name.find("TensorView") != std::string_view::npos
            && type_name.find("DiffTensorView") == std::string_view::npos;
    }

    template<typename TCursor>
    bool is_pointer_target(const TCursor& cursor)
    {
        slang::TypeLayoutReflection* layout = cursor.slang_type_layout();
        return layout && TypeReflection::Kind(layout->getKind()) == TypeReflection::Kind::pointer;
    }

    template<typename TCursor>
    void write_tensor_pointer_to_cursor_impl(const TCursor& cursor, const Tensor& tensor)
    {
        const uint64_t byte_offset = static_cast<uint64_t>(tensor.offset()) * tensor.element_stride();
        cursor.set_pointer(tensor.storage()->device_address() + byte_offset);
    }

    template<typename TCursor>
    void write_tensor_view_to_cursor_impl(
        const TCursor& cursor,
        const ref<const Buffer>& storage,
        const slangpy::Shape& shape,
        const slangpy::Shape& strides,
        int offset,
        size_t element_stride
    )
    {
        TensorViewData data = Tensor::make_tensor_view_data(storage, shape, strides, offset, element_stride);
        cursor.set_data(&data, sizeof(data));
    }

    template<typename TCursor>
    void write_tensor_view_to_cursor_impl(const TCursor& cursor, const Tensor& tensor)
    {
        write_tensor_view_to_cursor_impl(
            cursor,
            tensor.storage(),
            tensor.shape(),
            tensor.strides(),
            tensor.offset(),
            tensor.element_stride()
        );
    }

    template<typename TCursor>
    bool write_tensor_fields_to_cursor_impl(
        const TCursor& cursor,
        const ref<const Buffer>& storage,
        const slangpy::Shape& shape,
        const slangpy::Shape& strides,
        int offset,
        size_t element_stride
    )
    {
        auto tensor_data_cursor = cursor.find_field("_data");
        if (!tensor_data_cursor.is_valid())
            return false;

        auto shape_cursor = cursor.find_field("_shape");
        auto strides_cursor = cursor.find_field("_strides");
        auto offset_cursor = cursor.find_field("_offset");

        slang::TypeLayoutReflection* shape_layout = shape_cursor.slang_type_layout();
        const size_t expected_shape_dims = static_cast<size_t>(shape_layout->getElementCount());
        SGL_CHECK(
            shape.size() == expected_shape_dims,
            "\"{}\" expects tensor rank {} but got {}.",
            cursor_type_name(cursor),
            expected_shape_dims,
            shape.size()
        );

        if constexpr (requires { tensor_data_cursor.set_buffer(storage); }) {
            tensor_data_cursor.set_buffer(storage);
        } else {
            slang::TypeLayoutReflection* data_layout = tensor_data_cursor.slang_type_layout();
            slang::TypeReflection* data_type = data_layout ? data_layout->getType() : nullptr;
            if (data_type && data_type->getKind() == slang::TypeReflection::Kind::Pointer) {
                tensor_data_cursor.set_pointer(storage->device_address());
            } else {
                SGL_THROW(
                    "\"{}\" requires a shader resource binding and cannot be written through a BufferElementCursor.",
                    cursor_type_name(tensor_data_cursor)
                );
            }
        }

        shape_cursor
            ._set_array(shape.data(), shape.size() * sizeof(int), TypeReflection::ScalarType::int32, shape.size());
        strides_cursor._set_array(
            strides.data(),
            strides.size() * sizeof(int),
            TypeReflection::ScalarType::int32,
            strides.size()
        );
        offset_cursor.set(offset);

        auto element_byte_stride_cursor = cursor.find_field("_element_byte_stride");
        if (element_byte_stride_cursor.is_valid()) {
            const uint32_t element_byte_stride = static_cast<uint32_t>(element_stride);
            element_byte_stride_cursor.set(element_byte_stride);
        }

        return true;
    }

    template<typename TCursor>
    bool write_tensor_fields_to_cursor_impl(const TCursor& cursor, const Tensor& tensor)
    {
        return write_tensor_fields_to_cursor_impl(
            cursor,
            tensor.storage(),
            tensor.shape(),
            tensor.strides(),
            tensor.offset(),
            tensor.element_stride()
        );
    }

    template<typename TCursor>
    void write_tensor_to_cursor_impl(const Tensor& tensor, const TCursor& cursor)
    {
        if (is_pointer_target(cursor)) {
            write_tensor_pointer_to_cursor_impl(cursor, tensor);
            return;
        }

        if (write_tensor_fields_to_cursor_impl(cursor, tensor))
            return;

        // SlangPy differentiable tensor wrappers use separate primal/gradient fields.
        auto primal_cursor = cursor.find_field("_primal");
        if (primal_cursor.is_valid()) {
            bool wrote_primal = write_tensor_fields_to_cursor_impl(primal_cursor, tensor);
            SGL_CHECK(
                wrote_primal,
                "\"{}\" has a _primal field that is not a supported Tensor field target.",
                cursor_type_name(cursor)
            );

            auto grad_in_cursor = cursor.find_field("_grad_in");
            if (grad_in_cursor.is_valid()) {
                SGL_CHECK(tensor.grad_in(), "Missing required input gradients.");
                bool wrote_grad_in = write_tensor_fields_to_cursor_impl(grad_in_cursor, *tensor.grad_in());
                SGL_CHECK(
                    wrote_grad_in,
                    "\"{}\" has a _grad_in field that is not a supported Tensor field target.",
                    cursor_type_name(cursor)
                );
            }

            auto grad_out_cursor = cursor.find_field("_grad_out");
            if (grad_out_cursor.is_valid()) {
                SGL_CHECK(tensor.grad_out(), "Missing required output gradients.");
                bool wrote_grad_out = write_tensor_fields_to_cursor_impl(grad_out_cursor, *tensor.grad_out());
                SGL_CHECK(
                    wrote_grad_out,
                    "\"{}\" has a _grad_out field that is not a supported Tensor field target.",
                    cursor_type_name(cursor)
                );
            }
            return;
        }

        // Slang's CUDA TensorView/DiffTensorView path is represented as packed uniform data.
        if (is_tensor_view_target(cursor)) {
            write_tensor_view_to_cursor_impl(cursor, tensor);
            return;
        }

        auto view_primal_cursor = cursor.find_field("primal");
        auto view_diff_cursor = cursor.find_field("diff");
        if (view_primal_cursor.is_valid() && view_diff_cursor.is_valid()) {
            if (!write_tensor_fields_to_cursor_impl(view_primal_cursor, tensor)) {
                SGL_CHECK(
                    is_tensor_view_target(view_primal_cursor),
                    "\"{}\" primal field is not a supported TensorView target.",
                    cursor_type_name(cursor)
                );
                write_tensor_view_to_cursor_impl(view_primal_cursor, tensor);
            }

            const Tensor* diff_tensor = select_difftensor_view_gradient(tensor);
            if (!write_tensor_fields_to_cursor_impl(view_diff_cursor, *diff_tensor)) {
                SGL_CHECK(
                    is_tensor_view_target(view_diff_cursor),
                    "\"{}\" diff field is not a supported TensorView target.",
                    cursor_type_name(cursor)
                );
                write_tensor_view_to_cursor_impl(view_diff_cursor, *diff_tensor);
            }

            return;
        }

        SGL_THROW("\"{}\" is not a supported Tensor cursor target.", cursor_type_name(cursor));
    }

} // namespace

Tensor::Tensor(TensorDesc desc, ref<Buffer> storage, ref<Tensor> grad_in, ref<Tensor> grad_out)
    : m_desc(std::move(desc))
    , m_storage(std::move(storage))
    , m_grad_in(std::move(grad_in))
    , m_grad_out(std::move(grad_out))
{
    SGL_CHECK(m_storage, "Tensor requires storage");
    SGL_CHECK(m_desc.dtype, "Tensor requires a dtype");
    SGL_CHECK(m_desc.element_layout, "Tensor requires an element layout");
    update_signature();
}

Device* Tensor::device() const
{
    SGL_CHECK(m_storage, "Tensor has no storage");
    return m_storage->device();
}

TensorViewData Tensor::make_tensor_view_data(
    const ref<const Buffer>& storage,
    const slangpy::Shape& shape,
    const slangpy::Shape& strides,
    int offset,
    size_t element_stride
)
{
    SGL_CHECK(
        shape.size() <= kSlangPyTensorViewMaxDim,
        "TensorView supports at most {} dimensions, got {}.",
        kSlangPyTensorViewMaxDim,
        shape.size()
    );

    TensorViewData data = {};
    data.data = storage->device_address() + static_cast<uint64_t>(offset) * element_stride;

    for (size_t i = 0; i < shape.size(); ++i) {
        data.strides[i] = static_cast<uint32_t>(static_cast<size_t>(strides[i]) * element_stride);
        data.sizes[i] = static_cast<uint32_t>(shape[i]);
    }
    data.dimensionCount = static_cast<uint32_t>(shape.size());

    return data;
}

void Tensor::write_to_cursor(const ShaderCursor& cursor, const Tensor* value)
{
    if (!value) {
        SGL_CHECK(is_pointer_target(cursor), "Cannot write a null tensor pointer to a non-pointer shader cursor.");
        cursor.set_pointer(0);
        return;
    }
    write_tensor_to_cursor_impl(*value, cursor);
}

void Tensor::write_to_cursor(const BufferElementCursor& cursor, const Tensor* value)
{
    if (!value) {
        SGL_CHECK(is_pointer_target(cursor), "Cannot write a null tensor pointer to a non-pointer buffer cursor.");
        cursor.set_pointer(0);
        return;
    }
    write_tensor_to_cursor_impl(*value, cursor);
}

void Tensor::update_signature()
{
    m_signature = fmt::format("[{},{},{}]", m_desc.dtype->full_name(), m_desc.shape.size(), m_desc.usage);
}

void Tensor::write_slangpy_signature(SignatureBuffer& signature, const Tensor* value)
{
    SGL_CHECK(value, "Cannot write a SlangPy signature for a null tensor pointer.");
    signature.add("Tensor\n");
    signature.add(value->m_signature);
}

bool Tensor::is_contiguous() const
{
    const slangpy::Shape& shape_ref = shape();
    const slangpy::Shape& stride_ref = strides();

    int prod = 1;
    for (int i = dims() - 1; i >= 0; --i) {
        if (shape_ref[i] == 1)
            continue;

        if (stride_ref[i] != prod)
            return false;
        prod *= shape_ref[i];
    }

    return true;
}

ref<BufferCursor> Tensor::cursor(std::optional<int> start, std::optional<int> count) const
{
    size_t el_stride = m_desc.element_layout->stride();
    size_t size = count.value_or(element_count()) * el_stride;
    size_t byte_offset = (m_desc.offset + start.value_or(0)) * el_stride;
    return make_ref<BufferCursor>(m_desc.element_layout, m_storage, size, byte_offset);
}

void Tensor::view_inplace(slangpy::Shape shape, slangpy::Shape strides, int offset)
{
    SGL_CHECK(shape.valid(), "New shape must be valid");

    if (!strides.valid() || strides.size() == 0)
        strides = shape.calc_contiguous_strides();

    for (size_t i = 0; i < strides.size(); ++i)
        SGL_CHECK(strides[i] >= 0, "Strides must be positive");

    SGL_CHECK(
        shape.size() == strides.size(),
        "Shape dimensions ({}) must match stride dimensions ({})",
        shape.size(),
        strides.size()
    );

    m_desc.shape = std::move(shape);
    m_desc.strides = std::move(strides);
    m_desc.offset += offset;

    SGL_CHECK(m_desc.offset >= 0, "Tensor view offset is negative");
    update_signature();
}

void Tensor::broadcast_to_inplace(const slangpy::Shape& new_shape)
{
    const slangpy::Shape& curr_shape = shape();
    const slangpy::Shape& curr_strides = strides();

    int dim_delta = static_cast<int>(new_shape.size()) - static_cast<int>(curr_shape.size());
    if (dim_delta < 0)
        SGL_THROW("Broadcast shape must be larger than tensor shape");

    for (size_t i = 0; i < curr_shape.size(); ++i) {
        if (curr_shape[i] != new_shape[dim_delta + i] && curr_shape[i] != 1) {
            SGL_THROW(
                "Current dimension {} at index {} must be equal to new dimension {} or 1",
                curr_shape[i],
                i,
                new_shape[dim_delta + i]
            );
        }
    }

    slangpy::Shape new_strides(new_shape.size());
    int* new_strides_data = new_strides.data();
    for (size_t i = 0; i < new_shape.size(); ++i)
        new_strides_data[i] = 0;
    for (size_t i = 0; i < curr_shape.size(); ++i) {
        if (curr_shape[i] > 1)
            new_strides_data[dim_delta + i] = curr_strides[i];
    }

    view_inplace(new_shape, new_strides, 0);
}

void Tensor::clear(CommandEncoder* cmd)
{
    if (cmd) {
        cmd->clear_buffer(m_storage);
    } else {
        ref<CommandEncoder> temp_cmd = device()->create_command_encoder();
        temp_cmd->clear_buffer(m_storage);
        device()->submit_command_buffer(temp_cmd->finish());
    }
}

void Tensor::point_to(ref<Tensor> target)
{
    SGL_CHECK(target, "Tensor point_to target must not be null");
    SGL_CHECK(shape() == target->shape(), "Shape of existing and new tensor view must match");
    SGL_CHECK(usage() == target->usage(), "Usage flags of existing and new tensor storage must match");
    SGL_CHECK(memory_type() == target->memory_type(), "Memory type of existing and new tensor storage must match");
    SGL_CHECK(element_stride() == target->element_stride(), "Element size of new and existing data type must match");

    m_desc.offset = target->offset();
    m_desc.strides = target->strides();
    m_storage = target->m_storage;
    update_signature();
}

ref<Tensor> Tensor::view(slangpy::Shape shape, slangpy::Shape strides, int offset) const
{
    ref<Tensor> result = make_ref<Tensor>(m_desc, storage(), m_grad_in, m_grad_out);
    result->view_inplace(std::move(shape), std::move(strides), offset);
    return result;
}

ref<Tensor> Tensor::broadcast_to(const slangpy::Shape& shape) const
{
    ref<Tensor> result = make_ref<Tensor>(m_desc, storage(), m_grad_in, m_grad_out);
    result->broadcast_to_inplace(shape);
    return result;
}

ref<Tensor> Tensor::grad() const
{
    SGL_CHECK(m_grad_out, "Tensor has no grad.");
    return m_grad_out;
}

ref<Tensor> Tensor::with_grads(ref<Tensor> grad_in, ref<Tensor> grad_out, bool zero) const
{
    ref<Tensor> new_grad_in = std::move(grad_in);
    ref<Tensor> new_grad_out = std::move(grad_out);

    if (!new_grad_in && !new_grad_out) {
        ref<refl::Type> dtype = m_desc.dtype->derivative();
        if (!dtype)
            SGL_THROW("No derivative type found for {}", m_desc.dtype->name());
        ref<TypeLayoutReflection> layout = dtype->buffer_type_layout();

        BufferDesc buffer_desc;
        buffer_desc.usage = BufferUsage::shader_resource | BufferUsage::unordered_access | BufferUsage::shared;
        buffer_desc.struct_size = layout->stride();
        buffer_desc.element_count = element_count();
        ref<Buffer> buffer = device()->create_buffer(buffer_desc);

        TensorDesc desc;
        desc.dtype = dtype;
        desc.element_layout = layout;
        desc.shape = m_desc.shape;
        desc.strides = m_desc.shape.calc_contiguous_strides();
        desc.offset = m_desc.offset;
        desc.usage = buffer_desc.usage;
        desc.memory_type = buffer_desc.memory_type;

        new_grad_in = make_ref<Tensor>(desc, buffer, nullptr, nullptr);
        new_grad_out = new_grad_in;
    }

    ref<Tensor> result = make_ref<Tensor>(m_desc, storage(), new_grad_in, new_grad_out);

    if (zero) {
        if (new_grad_in)
            new_grad_in->clear();
        if (new_grad_out && new_grad_out != new_grad_in)
            new_grad_out->clear();
    }

    return result;
}

ref<Tensor> Tensor::detach() const
{
    return make_ref<Tensor>(m_desc, storage(), nullptr, nullptr);
}

std::string Tensor::to_string() const
{
    return fmt::format(
        "Tensor(dtype={}, shape={}, has_grad_in={}, has_grad_out={})",
        m_desc.dtype->to_string(),
        m_desc.shape.to_string(),
        m_grad_in ? "true" : "false",
        m_grad_out ? "true" : "false"
    );
}

} // namespace sgl::func
