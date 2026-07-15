// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"

#include "sgl/device/buffer_cursor.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/shader_object.h"
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

ref<ShaderProgram> make_tensor_cursor_program(Device* device)
{
    ref<SlangModule> module = device->load_module_from_source(
        "func_tensor_cursor_tests",
        R"(
import slangpy;

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main(
    uniform Tensor<float, 2> plain,
    uniform DiffTensor<float, 2> diff,
    uniform WDiffTensor<float, 2> wdiff,
    uniform RWDiffTensor<float, 2> rwdiff)
{
}
)"
    );
    return device->link_program({module}, {module->entry_point("compute_main")});
}

ref<const TypeLayoutReflection> make_tensor_holder_layout(Device* device)
{
    ref<SlangModule> module = device->load_module_from_source(
        "func_tensor_buffer_cursor_tests",
        R"(
import slangpy;

struct TensorHolder
{
    Tensor<float, 2> tensor;
};
)"
    );

    auto layout = module->layout();
    auto holder_type = layout->find_type_by_name("TensorHolder");
    return layout->get_type_layout(holder_type);
}

bool is_pointer_cursor(const BufferElementCursor& cursor)
{
    slang::TypeLayoutReflection* layout = cursor.slang_type_layout();
    slang::TypeReflection* type = layout ? layout->getType() : nullptr;
    return type && type->getKind() == slang::TypeReflection::Kind::Pointer;
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

TEST_CASE_GPU("tensor write_to_cursor binds tensor and gradient fields")
{
    TensorTestType type = make_tensor_test_type(ctx.device);
    slangpy::Shape shape({2, 2});

    ref<func::Tensor> tensor = make_tensor(ctx.device, type, shape);
    ref<func::Tensor> grad_in = make_tensor(ctx.device, type, shape);
    ref<func::Tensor> grad_out = make_tensor(ctx.device, type, shape);
    ref<func::Tensor> with_grads = tensor->with_grads(grad_in, grad_out, false);

    ref<ShaderProgram> program = make_tensor_cursor_program(ctx.device);
    ref<ShaderObject> root_object = ctx.device->create_root_shader_object(program);
    ShaderCursor entry_point = ShaderCursor(root_object.get()).find_entry_point(0);

    CHECK_NOTHROW(entry_point["plain"] = tensor);
    CHECK_THROWS(entry_point["diff"] = tensor);
    CHECK_NOTHROW(entry_point["diff"] = with_grads);
    CHECK_NOTHROW(entry_point["wdiff"] = with_grads);
    CHECK_NOTHROW(entry_point["rwdiff"] = with_grads);
}

TEST_CASE_GPU("tensor write_to_cursor supports pointer-backed buffer cursor")
{
    TensorTestType type = make_tensor_test_type(ctx.device);
    slangpy::Shape shape({2, 2});

    ref<func::Tensor> tensor = make_tensor(
        ctx.device,
        type,
        shape,
        BufferUsage::shader_resource | BufferUsage::unordered_access,
        MemoryType::device_local,
        5
    );
    ref<func::Tensor> view = tensor->view(shape, shape.calc_contiguous_strides(), 1);

    ref<const TypeLayoutReflection> holder_layout = make_tensor_holder_layout(ctx.device);
    auto cursor = make_ref<BufferCursor>(ctx.device->type(), holder_layout, 1);
    BufferElementCursor tensor_cursor = (*cursor)[0]["tensor"];
    BufferElementCursor data_cursor = tensor_cursor["_data"];

    if (!is_pointer_cursor(data_cursor)) {
        CHECK_THROWS(tensor_cursor = view);
        return;
    }

    CHECK_NOTHROW(tensor_cursor = view);

    uint64_t data_pointer = 0;
    data_cursor._get_data(data_cursor._get_offset(), &data_pointer, sizeof(data_pointer));
    CHECK(data_pointer == tensor->storage()->device_address());
    CHECK(tensor_cursor["_offset"].as<uint32_t>() == 1);
    CHECK(tensor_cursor["_shape"][0].as<uint32_t>() == 2);
    CHECK(tensor_cursor["_shape"][1].as<uint32_t>() == 2);
    CHECK(tensor_cursor["_strides"][0].as<uint32_t>() == 2);
    CHECK(tensor_cursor["_strides"][1].as<uint32_t>() == 1);
}

TEST_SUITE_END();
