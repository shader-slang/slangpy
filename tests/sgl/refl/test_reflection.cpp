// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"

#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/refl/layout.h"
#include "sgl/refl/type.h"

using namespace sgl;

TEST_SUITE_BEGIN("refl");

TEST_CASE_GPU("semantic type lookup")
{
    ref<SlangModule> module = ctx.device->load_module_from_source(
        "refl_semantic_type_lookup",
        R"(
struct Foo {
    float3 value;
};
)"
    );

    ref<refl::Layout> layout = make_ref<refl::Layout>(module->layout());

    ref<refl::ScalarType> float_type = dynamic_ref_cast<refl::ScalarType>(layout->require_type_by_name("float"));
    REQUIRE(float_type);
    CHECK(float_type->name() == "float");
    CHECK(float_type->full_name() == "float");
    CHECK(float_type->shape() == slangpy::Shape(std::vector<int>{}));
    CHECK(float_type->slang_scalar_type() == TypeReflection::ScalarType::float32);

    ref<refl::VectorType> vector_type
        = dynamic_ref_cast<refl::VectorType>(layout->require_type_by_name("vector<float,3>"));
    REQUIRE(vector_type);
    CHECK(vector_type->element_type() == float_type);
    CHECK(vector_type->num_elements() == 3);
    CHECK(vector_type->shape() == slangpy::Shape({3}));
    CHECK(vector_type->vector_type_name() == "vector<float,3>");

    ref<refl::MatrixType> matrix_type
        = dynamic_ref_cast<refl::MatrixType>(layout->require_type_by_name("matrix<float,3,2>"));
    REQUIRE(matrix_type);
    CHECK(matrix_type->rows() == 3);
    CHECK(matrix_type->cols() == 2);
    CHECK(matrix_type->shape() == slangpy::Shape({3, 2}));
    CHECK(matrix_type->inner_element_type() == float_type);

    ref<refl::ArrayType> array_type = dynamic_ref_cast<refl::ArrayType>(layout->require_type_by_name("float[4]"));
    REQUIRE(array_type);
    CHECK(array_type->element_type() == float_type);
    CHECK(array_type->num_elements() == 4);
    CHECK(array_type->array_shape() == slangpy::Shape({4}));
    CHECK(array_type->shape() == slangpy::Shape({4}));

    ref<refl::StructType> struct_type = dynamic_ref_cast<refl::StructType>(layout->require_type_by_name("Foo"));
    REQUIRE(struct_type);
    CHECK(struct_type->name() == "Foo");
    CHECK(struct_type->full_name() == "Foo");
    CHECK(struct_type->shape() == slangpy::Shape(std::vector<int>{}));
}

TEST_CASE_GPU("semantic tensor type lookup")
{
    ref<SlangModule> module = ctx.device->load_module_from_source(
        "refl_semantic_tensor_type_lookup",
        R"(
struct Tensor<T, let D : int> {}
struct WTensor<T, let D : int> {}
struct RWTensor<T, let D : int> {}
struct DiffTensor<T, let D : int> {}
struct WDiffTensor<T, let D : int> {}
struct RWDiffTensor<T, let D : int> {}
struct PrimalTensor<T, let D : int> {}
struct WPrimalTensor<T, let D : int> {}
struct RWPrimalTensor<T, let D : int> {}
struct ITensor<T, let D : int> {}
struct IWTensor<T, let D : int> {}
struct IRWTensor<T, let D : int> {}
struct IDiffTensor<T, let D : int> {}
struct IWDiffTensor<T, let D : int> {}
struct IRWDiffTensor<T, let D : int> {}
struct AtomicTensor<T, let D : int> {}
struct TensorView<T> {}
struct DiffTensorView<T> {}
void use_tensor(
    RWTensor<float, 2> tensor,
    DiffTensor<float, 3> diff_tensor,
    TensorView<float> tensor_view,
    DiffTensorView<float> diff_tensor_view
) {}
)"
    );

    ref<refl::Layout> layout = make_ref<refl::Layout>(module->layout());
    ref<refl::ScalarType> float_type = layout->scalar_type(TypeReflection::ScalarType::float32);
    REQUIRE(float_type);

    ref<const FunctionReflection> function
        = const_cast<ProgramLayout*>(module->layout().get())->find_function_by_name("use_tensor");
    REQUIRE(function);
    auto parameters = function->parameters();
    REQUIRE(parameters.size() == 4);

    ref<refl::TensorType> tensor_type = dynamic_ref_cast<refl::TensorType>(layout->find_type(parameters[0]->type()));
    REQUIRE(tensor_type);
    CHECK(tensor_type->full_name() == "RWTensor<float, 2>");
    CHECK(tensor_type->dtype() == float_type);
    CHECK(tensor_type->dims() == 2);
    CHECK(tensor_type->readable());
    CHECK(tensor_type->writable());
    CHECK(!tensor_type->diff_tensor());
    CHECK(tensor_type->shape() == slangpy::Shape({-1, -1}));

    ref<refl::TensorType> diff_tensor_type
        = dynamic_ref_cast<refl::TensorType>(layout->find_type(parameters[1]->type()));
    REQUIRE(diff_tensor_type);
    CHECK(diff_tensor_type->full_name() == "DiffTensor<float, 3>");
    CHECK(diff_tensor_type->diff_tensor());
    CHECK(!diff_tensor_type->has_grad_in());
    CHECK(diff_tensor_type->has_grad_out());

    ref<refl::TensorViewType> tensor_view_type
        = dynamic_ref_cast<refl::TensorViewType>(layout->find_type(parameters[2]->type()));
    REQUIRE(tensor_view_type);
    CHECK(tensor_view_type->dtype() == float_type);

    ref<refl::DiffTensorViewType> diff_tensor_view_type
        = dynamic_ref_cast<refl::DiffTensorViewType>(layout->find_type(parameters[3]->type()));
    REQUIRE(diff_tensor_view_type);
    CHECK(diff_tensor_view_type->dtype() == float_type);
}

TEST_SUITE_END();
