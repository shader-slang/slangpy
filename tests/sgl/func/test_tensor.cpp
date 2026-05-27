// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"

#include "sgl/device/buffer_cursor.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/func/tensor.h"
#include "sgl/refl/layout.h"
#include "sgl/refl/type.h"

using namespace sgl;

TEST_SUITE_BEGIN("func");

namespace {

struct TensorTestType {
    ref<refl::Type> dtype;
    ref<TypeLayoutReflection> element_layout;
};

TensorTestType make_tensor_test_type(Device* device)
{
    ref<SlangModule> module = device->load_module_from_source(
        "func_tensor_tests",
        R"(
struct Foo {
    float value;
};
)"
    );

    ref<refl::Layout> layout = make_ref<refl::Layout>(module->layout());
    ref<refl::Type> dtype = layout->require_type_by_name("float");
    ref<TypeLayoutReflection> element_layout = dtype->buffer_type_layout();
    REQUIRE(dtype);
    REQUIRE(element_layout);
    return {std::move(dtype), std::move(element_layout)};
}

ref<Buffer> make_tensor_storage(
    Device* device,
    const TensorTestType& type,
    size_t element_count,
    BufferUsage usage = BufferUsage::shader_resource | BufferUsage::unordered_access,
    MemoryType memory_type = MemoryType::device_local
)
{
    BufferDesc buffer_desc;
    buffer_desc.usage = usage;
    buffer_desc.memory_type = memory_type;
    buffer_desc.struct_size = type.element_layout->stride();
    buffer_desc.element_count = element_count;
    return device->create_buffer(buffer_desc);
}

ref<func::Tensor> make_tensor(
    Device* device,
    const TensorTestType& type,
    slangpy::Shape shape,
    BufferUsage usage = BufferUsage::shader_resource | BufferUsage::unordered_access,
    MemoryType memory_type = MemoryType::device_local,
    size_t storage_element_count = 0
)
{
    if (storage_element_count == 0)
        storage_element_count = shape.element_count();

    func::TensorDesc desc;
    desc.dtype = type.dtype;
    desc.element_layout = type.element_layout;
    desc.shape = shape;
    desc.strides = shape.calc_contiguous_strides();
    desc.offset = 0;
    desc.usage = usage;
    desc.memory_type = memory_type;

    ref<Buffer> storage = make_tensor_storage(device, type, storage_element_count, usage, memory_type);
    return make_ref<func::Tensor>(std::move(desc), std::move(storage));
}

} // namespace

TEST_CASE_GPU("tensor metadata and views")
{
    TensorTestType type = make_tensor_test_type(ctx.device);
    ref<func::Tensor> tensor = make_tensor(ctx.device, type, slangpy::Shape({2, 3}));

    CHECK(tensor->device() == ctx.device);
    CHECK(tensor->dtype() == type.dtype);
    CHECK(tensor->shape() == slangpy::Shape({2, 3}));
    CHECK(tensor->strides() == slangpy::Shape({3, 1}));
    CHECK(tensor->element_count() == 6);
    CHECK(tensor->element_stride() == type.element_layout->stride());
    CHECK(tensor->is_contiguous());
    CHECK(tensor->signature().find("float") != std::string::npos);
    CHECK(tensor->to_string().find("Tensor(") != std::string::npos);

    ref<BufferCursor> cursor = tensor->cursor(1, 2);
    CHECK(cursor->element_count() == 2);
    CHECK(cursor->element_stride() == type.element_layout->stride());

    ref<func::Tensor> view = tensor->view(slangpy::Shape({3, 2}), slangpy::Shape({1, 3}), 1);
    CHECK(view->shape() == slangpy::Shape({3, 2}));
    CHECK(view->strides() == slangpy::Shape({1, 3}));
    CHECK(view->offset() == 1);
    CHECK(!view->is_contiguous());

    ref<func::Tensor> broadcast = tensor->broadcast_to(slangpy::Shape({4, 2, 3}));
    CHECK(broadcast->shape() == slangpy::Shape({4, 2, 3}));
    CHECK(broadcast->strides() == slangpy::Shape({0, 3, 1}));
    CHECK(!broadcast->is_contiguous());
}

TEST_CASE_GPU("tensor point_to and gradients")
{
    TensorTestType type = make_tensor_test_type(ctx.device);
    slangpy::Shape shape({2, 2});

    ref<func::Tensor> tensor = make_tensor(ctx.device, type, shape);
    ref<func::Tensor> target = make_tensor(ctx.device, type, shape, tensor->usage(), tensor->memory_type(), 5);
    ref<func::Tensor> target_view = target->view(shape, slangpy::Shape({2, 1}), 1);

    ref<func::Tensor> alias = make_tensor(ctx.device, type, shape);
    alias->point_to(target_view);
    CHECK(alias->storage() == target->storage());
    CHECK(alias->offset() == 1);
    CHECK(alias->shape() == target_view->shape());
    CHECK(alias->strides() == target_view->strides());

    ref<func::Tensor> grad_in = make_tensor(ctx.device, type, shape);
    ref<func::Tensor> grad_out = make_tensor(ctx.device, type, shape);
    ref<func::Tensor> with_grads = tensor->with_grads(grad_in, grad_out, false);

    CHECK(with_grads->grad_in() == grad_in);
    CHECK(with_grads->grad_out() == grad_out);
    CHECK(with_grads->grad() == grad_out);

    ref<func::Tensor> detached = with_grads->detach();
    CHECK(detached->grad_in() == nullptr);
    CHECK(detached->grad_out() == nullptr);
    CHECK(detached->storage() == tensor->storage());
}

TEST_SUITE_END();
