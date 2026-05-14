// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/refl/type.h"

#include "sgl/refl/function.h"
#include "sgl/refl/layout.h"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <utility>

namespace sgl::refl {

//------------------------------------
// Internal helpers
//------------------------------------

namespace {

    slangpy::Shape shape_from_vector(std::vector<int> shape)
    {
        return slangpy::Shape(std::optional<std::vector<int>>(std::move(shape)));
    }

    slangpy::Shape empty_shape()
    {
        return shape_from_vector({});
    }

    std::string c_string(const char* value)
    {
        return value ? std::string(value) : std::string();
    }

    std::string trim(std::string_view value)
    {
        size_t begin = 0;
        while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])))
            ++begin;

        size_t end = value.size();
        while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])))
            --end;

        return std::string(value.substr(begin, end - begin));
    }

    bool can_convert_to_int(std::string_view value)
    {
        std::string normalized = trim(value);
        if (normalized.empty())
            return false;

        size_t index = 0;
        if (normalized[index] == '+' || normalized[index] == '-')
            ++index;
        if (index == normalized.size())
            return false;

        for (; index < normalized.size(); ++index) {
            if (!std::isdigit(static_cast<unsigned char>(normalized[index])))
                return false;
        }
        return true;
    }

    bool can_convert_to_bool(std::string_view value)
    {
        std::string normalized = trim(value);
        std::transform(
            normalized.begin(),
            normalized.end(),
            normalized.begin(),
            [](unsigned char c)
            {
                return static_cast<char>(std::tolower(c));
            }
        );
        return normalized == "true" || normalized == "false" || normalized == "bool(1)" || normalized == "bool(0)";
    }

    int convert_bool_to_int(std::string_view value)
    {
        std::string normalized = trim(value);
        std::transform(
            normalized.begin(),
            normalized.end(),
            normalized.begin(),
            [](unsigned char c)
            {
                return static_cast<char>(std::tolower(c));
            }
        );
        return (normalized == "true" || normalized == "bool(1)") ? 1 : 0;
    }

    std::vector<std::string> split_generic_args(std::string_view full_name)
    {
        SGL_CHECK(
            !full_name.empty() && full_name.back() == '>',
            "Type '{}' is not a generic specialization",
            full_name
        );

        int nesting = 0;
        int open_index = -1;
        for (int i = int(full_name.size()) - 2; i >= 0; --i) {
            char ch = full_name[size_t(i)];
            if (ch == '>')
                ++nesting;
            else if (ch == '<') {
                if (nesting == 0) {
                    open_index = i;
                    break;
                }
                --nesting;
            }
        }

        SGL_CHECK(open_index >= 0, "Unable to parse generic '{}'", full_name);

        std::string_view args = full_name.substr(size_t(open_index + 1), full_name.size() - size_t(open_index + 2));
        std::vector<std::string> pieces;
        size_t start = 0;
        nesting = 0;
        for (size_t i = 0; i < args.size(); ++i) {
            char ch = args[i];
            if (ch == '<')
                ++nesting;
            else if (ch == '>')
                --nesting;
            else if (ch == ',' && nesting == 0) {
                pieces.push_back(trim(args.substr(start, i - start)));
                start = i + 1;
            }
        }
        pieces.push_back(trim(args.substr(start)));
        return pieces;
    }

    std::string scalar_type_name(TypeReflection::ScalarType scalar_type)
    {
        switch (scalar_type) {
        case TypeReflection::ScalarType::none_:
            return "Unknown";
        case TypeReflection::ScalarType::void_:
            return "void";
        case TypeReflection::ScalarType::bool_:
            return "bool";
        case TypeReflection::ScalarType::int8:
            return "int8_t";
        case TypeReflection::ScalarType::int16:
            return "int16_t";
        case TypeReflection::ScalarType::int32:
            return "int";
        case TypeReflection::ScalarType::int64:
            return "int64_t";
        case TypeReflection::ScalarType::uint8:
            return "uint8_t";
        case TypeReflection::ScalarType::uint16:
            return "uint16_t";
        case TypeReflection::ScalarType::uint32:
            return "uint";
        case TypeReflection::ScalarType::uint64:
            return "uint64_t";
        case TypeReflection::ScalarType::float16:
            return "half";
        case TypeReflection::ScalarType::float32:
            return "float";
        case TypeReflection::ScalarType::float64:
            return "double";
        default:
            SGL_THROW("Unsupported scalar type '{}'", scalar_type);
        }
    }

    bool is_unknown_type(const ref<Type>& type)
    {
        return !type || type->full_name() == "Unknown";
    }

    bool is_generic_type(const ref<Type>& type)
    {
        return !type || is_unknown_type(type) || type->is_generic();
    }

    TensorType::Kind tensor_kind_from_name(std::string_view name)
    {
        if (name == "ITensor" || name == "IWTensor" || name == "IRWTensor")
            return TensorType::Kind::itensor;
        if (name == "DiffTensor" || name == "WDiffTensor" || name == "RWDiffTensor")
            return TensorType::Kind::diff_tensor;
        if (name == "IDiffTensor" || name == "IWDiffTensor" || name == "IRWDiffTensor")
            return TensorType::Kind::idiff_tensor;
        if (name == "PrimalTensor" || name == "WPrimalTensor" || name == "RWPrimalTensor")
            return TensorType::Kind::primal_tensor;
        if (name == "AtomicTensor")
            return TensorType::Kind::atomic;
        return TensorType::Kind::tensor;
    }

    TensorType::Access tensor_access_from_name(std::string_view name)
    {
        if (name == "WTensor" || name == "WDiffTensor" || name == "WPrimalTensor" || name == "IWTensor"
            || name == "IWDiffTensor")
            return TensorType::Access::write;
        if (name == "RWTensor" || name == "RWDiffTensor" || name == "RWPrimalTensor" || name == "IRWTensor"
            || name == "IRWDiffTensor" || name == "AtomicTensor")
            return TensorType::Access::read_write;
        return TensorType::Access::read;
    }

    bool is_tensor_type_name(std::string_view name)
    {
        return name == "Tensor" || name == "WTensor" || name == "RWTensor" || name == "DiffTensor"
            || name == "WDiffTensor" || name == "RWDiffTensor" || name == "PrimalTensor" || name == "WPrimalTensor"
            || name == "RWPrimalTensor" || name == "ITensor" || name == "IWTensor" || name == "IRWTensor"
            || name == "IDiffTensor" || name == "IWDiffTensor" || name == "IRWDiffTensor" || name == "AtomicTensor";
    }

    int resource_texture_dims(TypeReflection::ResourceShape shape)
    {
        switch (shape) {
        case TypeReflection::ResourceShape::texture_1d:
            return 1;
        case TypeReflection::ResourceShape::texture_2d:
        case TypeReflection::ResourceShape::texture_2d_multisample:
        case TypeReflection::ResourceShape::texture_1d_array:
            return 2;
        case TypeReflection::ResourceShape::texture_3d:
        case TypeReflection::ResourceShape::texture_cube:
        case TypeReflection::ResourceShape::texture_2d_array:
        case TypeReflection::ResourceShape::texture_2d_multisample_array:
            return 3;
        case TypeReflection::ResourceShape::texture_cube_array:
            return 4;
        default:
            return 0;
        }
    }

} // namespace

//------------------------------------
// TypeLayout
//------------------------------------

TypeLayout::TypeLayout(ref<const TypeLayoutReflection> reflection)
    : m_reflection(std::move(reflection))
{
    SGL_CHECK(m_reflection, "TypeLayout requires a reflection object");
}

std::string TypeLayout::to_string() const
{
    return fmt::format("refl::TypeLayout(size={}, alignment={}, stride={})", size(), alignment(), stride());
}

//------------------------------------
// Type
//------------------------------------

Type::Type(ref<Layout> layout, ref<const TypeReflection> reflection, ref<Type> element_type, slangpy::Shape local_shape)
    : m_layout(std::move(layout))
    , m_reflection(std::move(reflection))
    , m_element_type(std::move(element_type))
    , m_local_shape(std::move(local_shape))
{
    SGL_CHECK(m_layout, "Type requires a semantic layout");
    SGL_CHECK(m_reflection, "Type requires a reflection object");
    update_shape();
}

std::string Type::name() const
{
    return c_string(m_reflection->name());
}

std::string Type::full_name() const
{
    return m_reflection->full_name();
}

bool Type::is_generic() const
{
    return m_reflection->kind() == TypeReflection::Kind::none;
}

std::string Type::vector_type_name() const
{
    if (!m_vector_type_name)
        m_vector_type_name = full_name();
    return *m_vector_type_name;
}

ref<Type> Type::derivative()
{
    if (!m_derivative)
        m_derivative = find_type_by_name(full_name() + ".Differential");
    return m_derivative;
}

const std::unordered_map<std::string, ref<Field>>& Type::fields()
{
    if (!m_fields)
        m_fields = build_fields();
    return *m_fields;
}

std::unordered_map<std::string, ref<Field>> Type::build_fields()
{
    return {};
}

ref<TypeLayout> Type::uniform_layout()
{
    if (!m_uniform_layout) {
        ref<const TypeLayoutReflection> type_layout = m_layout->low_level_layout()->get_type_layout(m_reflection.get());
        SGL_CHECK(type_layout, "Unable to get uniform layout for '{}'", full_name());
        m_uniform_layout = make_ref<TypeLayout>(std::move(type_layout));
    }
    return m_uniform_layout;
}

ref<TypeLayout> Type::buffer_layout()
{
    if (!m_buffer_layout) {
        ref<Type> buffer_type = find_type_by_name(fmt::format("StructuredBuffer<{}>", full_name()));
        SGL_CHECK(buffer_type, "Unable to get buffer layout for '{}'", full_name());

        ref<const TypeLayoutReflection> buffer_layout
            = m_layout->low_level_layout()->get_type_layout(buffer_type->reflection());
        SGL_CHECK(buffer_layout, "Unable to get buffer layout for '{}'", full_name());

        m_buffer_layout = make_ref<TypeLayout>(buffer_layout->element_type_layout());
    }
    return m_buffer_layout;
}

void Type::on_hot_reload(ref<const TypeReflection> reflection)
{
    SGL_CHECK(reflection, "Type hot reload requires a reflection object");
    m_reflection = std::move(reflection);
    m_uniform_layout = nullptr;
    m_buffer_layout = nullptr;
    m_derivative = nullptr;
    m_fields.reset();
    m_vector_type_name.reset();
}

std::string Type::to_string() const
{
    return fmt::format(
        "refl::Type(full_name=\"{}\", kind={}, shape={})",
        full_name(),
        m_reflection->kind(),
        m_shape.to_string()
    );
}

void Type::set_element_type(ref<Type> element_type)
{
    m_element_type = std::move(element_type);
    update_shape();
}

void Type::set_local_shape(slangpy::Shape local_shape)
{
    m_local_shape = std::move(local_shape);
    update_shape();
}

void Type::update_shape()
{
    if (m_local_shape.valid() && m_element_type)
        m_shape = m_local_shape + m_element_type->shape();
    else
        m_shape = m_local_shape;
}

ref<Type> Type::find_type_by_name(std::string_view name) const
{
    return m_layout->find_type_by_name(name);
}

//------------------------------------
// Fundamental types
//------------------------------------

UnknownType::UnknownType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
}

VoidType::VoidType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
}

PointerType::PointerType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
    auto args = m_layout->get_resolved_generic_args(m_reflection.get());
    if (args && !args->empty() && (*args)[0].is_type())
        m_target_type = (*args)[0].type();
}

bool PointerType::is_generic() const
{
    return is_generic_type(m_target_type);
}

std::string PointerType::vector_type_name() const
{
    return fmt::format("Ptr<{}>", m_target_type ? m_target_type->vector_type_name() : "Unknown");
}

ScalarType::ScalarType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
    SGL_CHECK(m_reflection->scalar_type() != TypeReflection::ScalarType::none_, "ScalarType cannot wrap none");
    SGL_CHECK(m_reflection->scalar_type() != TypeReflection::ScalarType::void_, "ScalarType cannot wrap void");
}

//------------------------------------
// Vector and matrix types
//------------------------------------

VectorType::VectorType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
    m_num_elements = int(m_reflection->col_count());
    set_element_type(m_layout->scalar_type(m_reflection->scalar_type()));
    set_local_shape(shape_from_vector({m_num_elements}));
}

ref<ScalarType> VectorType::scalar_type() const
{
    return dynamic_ref_cast<ScalarType>(m_element_type);
}

std::string VectorType::vector_type_name() const
{
    return fmt::format(
        "vector<{},{}>",
        m_element_type ? m_element_type->vector_type_name() : "Unknown",
        m_num_elements
    );
}

std::unordered_map<std::string, ref<Field>> VectorType::build_fields()
{
    static constexpr std::array<const char*, 4> names = {"x", "y", "z", "w"};

    std::unordered_map<std::string, ref<Field>> fields;
    ref<ScalarType> scalar_type = this->scalar_type();
    const int count = std::min<int>(m_num_elements, int(names.size()));
    for (int i = 0; i < count; ++i)
        fields.emplace(names[size_t(i)], make_ref<Field>(m_layout, scalar_type, names[size_t(i)]));
    return fields;
}

MatrixType::MatrixType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
    m_cols = int(m_reflection->col_count());
    m_rows = int(m_reflection->row_count());
    if (m_cols > 0 && m_rows > 0) {
        set_element_type(m_layout->vector_type(m_reflection->scalar_type(), m_cols));
        set_local_shape(shape_from_vector({m_rows}));
    } else {
        set_element_type(m_layout->scalar_type(m_reflection->scalar_type()));
        set_local_shape(shape_from_vector({m_rows, m_cols}));
    }
}

ref<ScalarType> MatrixType::scalar_type() const
{
    if (auto vector = dynamic_ref_cast<VectorType>(m_element_type))
        return vector->scalar_type();
    return dynamic_ref_cast<ScalarType>(m_element_type);
}

ref<Type> MatrixType::inner_element_type() const
{
    if (auto vector = dynamic_ref_cast<VectorType>(m_element_type))
        return vector->element_type();
    return m_element_type;
}

std::string MatrixType::vector_type_name() const
{
    ref<Type> inner = inner_element_type();
    return fmt::format("matrix<{},{},{}>", inner ? inner->vector_type_name() : "Unknown", m_rows, m_cols);
}

//------------------------------------
// Array and aggregate types
//------------------------------------

ArrayType::ArrayType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
    set_element_type(m_layout->find_type(m_reflection->element_type()));
    m_num_elements = int(m_reflection->element_count());
    set_local_shape(shape_from_vector({m_num_elements}));
}

slangpy::Shape ArrayType::array_shape() const
{
    std::vector<int> dims;
    const ArrayType* array = this;
    while (array) {
        dims.push_back(array->num_elements());
        array = dynamic_cast<const ArrayType*>(array->element_type().get());
    }
    return shape_from_vector(std::move(dims));
}

bool ArrayType::any_generic_dims() const
{
    const ArrayType* array = this;
    while (array) {
        if (array->num_elements() == 0)
            return true;
        array = dynamic_cast<const ArrayType*>(array->element_type().get());
    }
    return false;
}

ref<Type> ArrayType::inner_element_type() const
{
    ref<Type> type = m_element_type;
    while (auto array = dynamic_ref_cast<ArrayType>(type))
        type = array->element_type();
    return type;
}

std::string ArrayType::vector_type_name() const
{
    return fmt::format("Array<{},{}>", m_element_type ? m_element_type->vector_type_name() : "Unknown", m_num_elements);
}

StructType::StructType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
    auto args = m_layout->get_resolved_generic_args(m_reflection.get());
    if (args) {
        for (const GenericArg& arg : *args) {
            if (arg.is_type() && is_unknown_type(arg.type())) {
                m_is_generic = true;
                break;
            }
        }
    }
}

bool StructType::is_generic() const
{
    return m_is_generic;
}

std::string StructType::vector_type_name() const
{
    if (full_name().find('<') != std::string::npos)
        return "Unknown";
    return full_name();
}

std::unordered_map<std::string, ref<Field>> StructType::build_fields()
{
    std::unordered_map<std::string, ref<Field>> fields;
    for (ref<const VariableReflection> field_reflection : m_reflection->fields()) {
        if (field_reflection)
            fields.emplace(c_string(field_reflection->name()), make_ref<Field>(m_layout, field_reflection));
    }
    return fields;
}

InterfaceType::InterfaceType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
    auto args = m_layout->get_resolved_generic_args(m_reflection.get());
    if (args) {
        for (const GenericArg& arg : *args) {
            if (arg.is_type() && is_unknown_type(arg.type())) {
                m_is_generic = true;
                break;
            }
        }
    }
}

//------------------------------------
// Resource types
//------------------------------------

ResourceType::ResourceType(
    ref<Layout> layout,
    ref<const TypeReflection> reflection,
    ref<Type> element_type,
    slangpy::Shape local_shape
)
    : Type(std::move(layout), std::move(reflection), std::move(element_type), std::move(local_shape))
{
}

bool ResourceType::writable() const
{
    return resource_access() == TypeReflection::ResourceAccess::read_write
        || resource_access() == TypeReflection::ResourceAccess::access_write;
}

TextureType::TextureType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : ResourceType(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
    m_texture_dims = resource_texture_dims(resource_shape());
    set_element_type(m_layout->find_type(m_reflection->resource_result_type()));
    set_local_shape(shape_from_vector(std::vector<int>(size_t(m_texture_dims), -1)));
}

StructuredBufferType::StructuredBufferType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : ResourceType(std::move(layout), std::move(reflection), nullptr, shape_from_vector({-1}))
{
    ref<const TypeReflection> result_reflection = m_reflection->resource_result_type();
    if (!result_reflection || result_reflection->kind() == TypeReflection::Kind::none)
        set_element_type(m_layout->require_type_by_name("Unknown"));
    else
        set_element_type(m_layout->find_type(result_reflection));
}

std::string StructuredBufferType::vector_type_name() const
{
    return fmt::format("{}<{}>", name(), m_element_type ? m_element_type->full_name() : "Unknown");
}

ByteAddressBufferType::ByteAddressBufferType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : ResourceType(std::move(layout), std::move(reflection), nullptr, shape_from_vector({-1}))
{
    set_element_type(m_layout->scalar_type(TypeReflection::ScalarType::uint8));
}

//------------------------------------
// Differential and singleton types
//------------------------------------

DifferentialPairType::DifferentialPairType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
    auto args = m_layout->get_resolved_generic_args(m_reflection.get());
    SGL_CHECK(args && args->size() == 1 && (*args)[0].is_type(), "DifferentialPair requires one type argument");
    m_primal = (*args)[0].type();
}

ref<Type> DifferentialPairType::derivative()
{
    SGL_CHECK(m_primal, "DifferentialPair has no primal type");
    ref<Type> primal_derivative = m_primal->derivative();
    SGL_CHECK(primal_derivative, "DifferentialPair primal type '{}' is not differentiable", m_primal->full_name());
    return find_type_by_name(fmt::format("DifferentialPair<{}>", primal_derivative->full_name()));
}

RaytracingAccelerationStructureType::RaytracingAccelerationStructureType(
    ref<Layout> layout,
    ref<const TypeReflection> reflection
)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
}

SamplerStateType::SamplerStateType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
}

//------------------------------------
// Tensor types
//------------------------------------

TensorType::TensorType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
    auto args = m_layout->get_resolved_generic_args(m_reflection.get());
    SGL_CHECK(args && args->size() == 2 && (*args)[0].is_type(), "Tensor type '{}' requires T and dims", full_name());

    m_kind = tensor_kind_from_name(name());
    m_access = tensor_access_from_name(name());
    set_element_type((*args)[0].type());

    if ((*args)[1].is_integer()) {
        m_dims = (*args)[1].integer();
        set_local_shape(shape_from_vector(std::vector<int>(size_t(m_dims), -1)));
    } else {
        m_dims = 0;
        set_local_shape(empty_shape());
    }
}

bool TensorType::is_generic() const
{
    return is_generic_type(m_element_type) || m_dims == 0;
}

std::string TensorType::build_tensor_name(const Type& element_type, int dims, Access access, Kind tensor_kind)
{
    if (tensor_kind == Kind::atomic)
        return fmt::format("AtomicTensor<{}, {}>", element_type.full_name(), dims);

    std::string prefix;
    if (tensor_kind == Kind::itensor || tensor_kind == Kind::idiff_tensor)
        prefix += "I";

    if (access == Access::read_write)
        prefix += "RW";
    else if (access == Access::write)
        prefix += "W";
    else
        SGL_CHECK(access == Access::read, "Tensor must be readable, writable, or read-write");

    if (tensor_kind == Kind::diff_tensor || tensor_kind == Kind::idiff_tensor)
        prefix += "Diff";
    if (tensor_kind == Kind::primal_tensor)
        prefix += "Primal";

    return fmt::format("{}Tensor<{}, {}>", prefix, element_type.full_name(), dims);
}

TensorViewType::TensorViewType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
    auto args = m_layout->get_resolved_generic_args(m_reflection.get());
    SGL_CHECK(args && args->size() == 1 && (*args)[0].is_type(), "TensorView requires one type argument");
    set_element_type((*args)[0].type());
}

bool TensorViewType::is_generic() const
{
    return is_generic_type(m_element_type);
}

std::string TensorViewType::build_tensorview_name(const Type& element_type)
{
    return fmt::format("TensorView<{}>", element_type.full_name());
}

DiffTensorViewType::DiffTensorViewType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
    auto args = m_layout->get_resolved_generic_args(m_reflection.get());
    SGL_CHECK(args && !args->empty() && (*args)[0].is_type(), "DiffTensorView requires at least one type argument");
    set_element_type((*args)[0].type());
    if (args->size() > 1 && (*args)[1].is_type())
        m_wrapper_type = (*args)[1].type();
}

bool DiffTensorViewType::is_generic() const
{
    return is_generic_type(m_element_type);
}

std::string DiffTensorViewType::build_difftensorview_name(const Type& element_type)
{
    return fmt::format("DiffTensorView<{}>", element_type.full_name());
}

//------------------------------------
// Fallback and generic argument helpers
//------------------------------------

UnhandledType::UnhandledType(ref<Layout> layout, ref<const TypeReflection> reflection)
    : Type(std::move(layout), std::move(reflection), nullptr, empty_shape())
{
}

GenericArg GenericArg::integer(int value)
{
    GenericArg arg;
    arg.m_kind = Kind::integer;
    arg.m_integer = value;
    return arg;
}

GenericArg GenericArg::type(ref<Type> value)
{
    GenericArg arg;
    arg.m_kind = Kind::type;
    arg.m_type = std::move(value);
    return arg;
}

//------------------------------------
// Type factories
//------------------------------------

ref<Type> create_builtin_type(Layout& layout, ref<const TypeReflection> reflection)
{
    ref<Layout> layout_ref = ref(&layout);
    std::string name = c_string(reflection->name());

    if (name == "Unknown")
        return make_ref<UnknownType>(layout_ref, std::move(reflection));
    if (name == "DifferentialPair")
        return make_ref<DifferentialPairType>(layout_ref, std::move(reflection));
    if (is_tensor_type_name(name))
        return make_ref<TensorType>(layout_ref, std::move(reflection));
    if (name == "TensorView")
        return make_ref<TensorViewType>(layout_ref, std::move(reflection));
    if (name == "DiffTensorView")
        return make_ref<DiffTensorViewType>(layout_ref, std::move(reflection));

    switch (reflection->kind()) {
    case TypeReflection::Kind::scalar:
        if (reflection->scalar_type() == TypeReflection::ScalarType::void_)
            return make_ref<VoidType>(layout_ref, std::move(reflection));
        return make_ref<ScalarType>(layout_ref, std::move(reflection));
    case TypeReflection::Kind::vector:
        return make_ref<VectorType>(layout_ref, std::move(reflection));
    case TypeReflection::Kind::matrix:
        return make_ref<MatrixType>(layout_ref, std::move(reflection));
    case TypeReflection::Kind::array:
        return make_ref<ArrayType>(layout_ref, std::move(reflection));
    case TypeReflection::Kind::resource:
        switch (reflection->resource_shape()) {
        case TypeReflection::ResourceShape::structured_buffer:
            return make_ref<StructuredBufferType>(layout_ref, std::move(reflection));
        case TypeReflection::ResourceShape::byte_address_buffer:
            return make_ref<ByteAddressBufferType>(layout_ref, std::move(reflection));
        case TypeReflection::ResourceShape::acceleration_structure:
            return make_ref<RaytracingAccelerationStructureType>(layout_ref, std::move(reflection));
        default:
            if (resource_texture_dims(reflection->resource_shape()) > 0)
                return make_ref<TextureType>(layout_ref, std::move(reflection));
            return make_ref<ResourceType>(layout_ref, std::move(reflection));
        }
    case TypeReflection::Kind::sampler_state:
        return make_ref<SamplerStateType>(layout_ref, std::move(reflection));
    case TypeReflection::Kind::pointer:
        return make_ref<PointerType>(layout_ref, std::move(reflection));
    case TypeReflection::Kind::none:
        return make_ref<UnknownType>(layout_ref, std::move(reflection));
    case TypeReflection::Kind::struct_:
        return make_ref<StructType>(layout_ref, std::move(reflection));
    case TypeReflection::Kind::interface:
        return make_ref<InterfaceType>(layout_ref, std::move(reflection));
    default:
        return make_ref<UnhandledType>(layout_ref, std::move(reflection));
    }
}

std::optional<GenericArgs> parse_generic_args(Layout& layout, const TypeReflection* reflection)
{
    std::string full_name = reflection->full_name();
    if (full_name.empty() || full_name.back() != '>')
        return std::nullopt;

    std::vector<std::string> pieces = split_generic_args(full_name);
    GenericArgs result;
    result.reserve(pieces.size());

    for (const std::string& piece : pieces) {
        if (can_convert_to_int(piece)) {
            result.push_back(GenericArg::integer(std::stoi(piece)));
        } else if (can_convert_to_bool(piece)) {
            result.push_back(GenericArg::integer(convert_bool_to_int(piece)));
        } else {
            ref<Type> type = layout.find_type_by_name(piece);
            if (!type)
                type = layout.require_type_by_name("Unknown");
            result.push_back(GenericArg::type(std::move(type)));
        }
    }

    return result;
}

std::string name_for_scalar_type(TypeReflection::ScalarType scalar_type)
{
    return scalar_type_name(scalar_type);
}

} // namespace sgl::refl
