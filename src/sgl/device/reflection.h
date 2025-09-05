// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/types.h"
#include "sgl/device/fwd.h"

#include "sgl/core/macros.h"
#include "sgl/core/enum.h"
#include "sgl/core/error.h"
#include "sgl/core/type_utils.h"

#include "sgl/math/vector_types.h"

#include <slang.h>

#include <map>
#include <string>
#include <vector>

namespace sgl {

namespace detail {

    SGL_API ref<const DeclReflection> from_slang(ref<const Object> owner, slang::DeclReflection* decl_reflection);
    SGL_API ref<const TypeReflection> from_slang(ref<const Object> owner, slang::TypeReflection* type_reflection);
    SGL_API ref<const TypeLayoutReflection>
    from_slang(ref<const Object> owner, slang::TypeLayoutReflection* type_layout_reflection);
    SGL_API ref<const FunctionReflection>
    from_slang(ref<const Object> owner, slang::FunctionReflection* variable_reflection);
    SGL_API ref<const VariableReflection>
    from_slang(ref<const Object> owner, slang::VariableReflection* variable_reflection);
    SGL_API ref<const VariableLayoutReflection>
    from_slang(ref<const Object> owner, slang::VariableLayoutReflection* variable_layout_reflection);
    SGL_API ref<const EntryPointLayout>
    from_slang(ref<const Object> owner, slang::EntryPointLayout* entry_point_reflection);
    SGL_API ref<const ProgramLayout> from_slang(ref<const Object> owner, slang::ProgramLayout* program_layout);

    SGL_API void on_slang_wrapper_destroyed(void* slang_reflection);

    SGL_API void invalidate_all_reflection_data();
} // namespace detail


enum class ModifierID {
    shared = slang::Modifier::ID::Shared,
    nodiff = slang::Modifier::ID::NoDiff,
    static_ = slang::Modifier::ID::Static,
    const_ = slang::Modifier::ID::Const,
    export_ = slang::Modifier::ID::Export,
    extern_ = slang::Modifier::ID::Extern,
    differentiable = slang::Modifier::ID::Differentiable,
    mutating = slang::Modifier::ID::Mutating,
    in = slang::Modifier::ID::In,
    out = slang::Modifier::ID::Out,
    inout = slang::Modifier::ID::InOut,
};

SGL_ENUM_INFO(
    ModifierID,
    {
        {ModifierID::shared, "shared"},
        {ModifierID::nodiff, "nodiff"},
        {ModifierID::static_, "static"},
        {ModifierID::const_, "const"},
        {ModifierID::export_, "export"},
        {ModifierID::extern_, "extern"},
        {ModifierID::differentiable, "differentiable"},
        {ModifierID::mutating, "mutating"},
        {ModifierID::in, "inn"},
        {ModifierID::out, "out"},
        {ModifierID::inout, "inout"},
    }
);
SGL_ENUM_REGISTER(ModifierID);

/// Base class for read-only lazy evaluation list. This only maintains
/// a reference to the parent, and allocates/returns children on demand.
template<class ParentType, class ChildType>
class BaseReflectionList {
public:
    class Iterator {
    public:
        using value_type = ref<const ChildType>;

        explicit Iterator(const BaseReflectionList* list, uint32_t index)
            : m_list(list)
            , m_index(index)
        {
        }
        Iterator& operator++()
        {
            if (m_index >= m_list->size())
                SGL_THROW("Iterator out of range");
            m_index++;
            return *this;
        }
        Iterator operator++(int)
        {
            Iterator retval = *this;
            ++(*this);
            return retval;
        }
        bool operator==(Iterator other) const { return m_list == other.m_list && m_index == other.m_index; }
        bool operator!=(Iterator other) const { return !(*this == other); }
        ref<const ChildType> operator*() const { return (*m_list)[m_index]; }

    private:
        const BaseReflectionList* m_list;
        uint32_t m_index;
    };

    /// Iterator type def for STL
    using iterator = BaseReflectionList<ParentType, ChildType>::Iterator;

    BaseReflectionList(ref<const ParentType> owner)
        : m_owner(std::move(owner))
    {
    }

    BaseReflectionList(const BaseReflectionList& other) noexcept
        : m_owner(other.m_owner)
    {
    }

    BaseReflectionList(BaseReflectionList&& other) noexcept
        : m_owner(std::move(other.m_owner))
    {
    }

    virtual ~BaseReflectionList() = default;

    /// Number of entries in list.
    virtual uint32_t size() const = 0;

    /// Index operator
    virtual ref<const ChildType> operator[](uint32_t index) const { return evaluate(index); }

    /// Begin iterator
    Iterator begin() const { return Iterator(this, 0); }

    /// End iterator
    Iterator end() const { return Iterator(this, size()); }

protected:
    ref<const ParentType> m_owner;

    /// Evaluate and return a specific child in the parent object.
    virtual ref<const ChildType> evaluate(uint32_t child_index) const = 0;
};

/// Base class for read-only lazy evaluation list of search results.
/// To use it, the search function (e.g. children_of_kind) fills out the indices of
/// the children matching the search. The list can then be accessed as usual, and only
/// allocates/evaluates children as they're requested.
template<class ParentType, class ChildType>
class BaseReflectionIndexedList : public BaseReflectionList<ParentType, ChildType> {
public:
    BaseReflectionIndexedList(ref<const ParentType> owner, std::vector<uint32_t> indices)
        : BaseReflectionList<ParentType, ChildType>(std::move(owner))
        , m_indices(std::move(indices))
    {
    }

    BaseReflectionIndexedList(const BaseReflectionIndexedList& other) noexcept
        : BaseReflectionList<ParentType, ChildType>(other.m_owner)
        , m_indices(other.m_indices)
    {
    }

    BaseReflectionIndexedList(BaseReflectionIndexedList&& other) noexcept
        : BaseReflectionList<ParentType, ChildType>(std::move(other.m_owner))
        , m_indices(std::move(other.m_indices))
    {
    }

    virtual ~BaseReflectionIndexedList() = default;

    /// Number of search results in list.
    uint32_t size() const override { return static_cast<uint32_t>(m_indices.size()); }

    /// Index operator
    ref<const ChildType> operator[](uint32_t index) const override { return this->evaluate(m_indices[index]); }

private:
    std::vector<uint32_t> m_indices;
};

class SGL_API BaseReflectionObject : public Object {
    SGL_OBJECT(BaseReflectionObject)
public:
    BaseReflectionObject(ref<const Object> owner)
        : m_owner(std::move(owner))
    {
    }

    virtual void _hot_reload_invalidate() { m_owner = nullptr; }

    bool is_valid() const { return m_owner != nullptr; }

protected:
    ref<const Object> m_owner;
};

template<class SlangType>
class SGL_API BaseReflectionObjectImpl : public BaseReflectionObject {
public:
    BaseReflectionObjectImpl(ref<const Object> owner, SlangType* target)
        : BaseReflectionObject(std::move(owner))
        , m_target(target)
    {
    }
    ~BaseReflectionObjectImpl() { detail::on_slang_wrapper_destroyed(m_target); }

    SlangType* slang_target() const
    {
        SGL_CHECK(m_target, "Reflection object has been invalidated");
        return m_target;
    }

    void _hot_reload_invalidate() override
    {
        BaseReflectionObject::_hot_reload_invalidate();
        m_target = nullptr;
    }

private:
    SlangType* m_target;
};

class SGL_API DeclReflection : public BaseReflectionObjectImpl<slang::DeclReflection> {
    SGL_OBJECT(DeclReflection)

public:
    DeclReflection(ref<const Object> owner, slang::DeclReflection* target)
        : BaseReflectionObjectImpl(std::move(owner), target)
    {
    }


    /// Different kinds of decl slang can return.
    enum class Kind {
        unsupported = SLANG_DECL_KIND_UNSUPPORTED_FOR_REFLECTION,
        struct_ = SLANG_DECL_KIND_STRUCT,
        func = SLANG_DECL_KIND_FUNC,
        module = SLANG_DECL_KIND_MODULE,
        generic = SLANG_DECL_KIND_GENERIC,
        variable = SLANG_DECL_KIND_VARIABLE,
    };
    SGL_ENUM_INFO(
        Kind,
        {
            {Kind::unsupported, "unsupported"},
            {Kind::struct_, "struct"},
            {Kind::func, "func"},
            {Kind::module, "module"},
            {Kind::generic, "generic"},
            {Kind::variable, "variable"},
        }
    );

    /// Decl kind (struct/function/module/generic/variable).
    Kind kind() const { return static_cast<Kind>(slang_target()->getKind()); }

    /// List of children of this cursor.
    DeclReflectionChildList children() const;

    /// Get number of children.
    uint32_t child_count() const { return slang_target()->getChildrenCount(); }

    /// Get a child by index.
    ref<const DeclReflection> child(uint32_t index) const
    {
        if (index > child_count())
            SGL_THROW("Child index out of range: {}", index);
        return detail::from_slang(m_owner, slang_target()->getChild(index));
    }

    /// Get the name of this decl (if it is of a kind that has a name).
    /// Note: Only supported for types, functions and variables.
    std::string name() const;

    /// List of children of this cursor of a specific kind.
    DeclReflectionIndexedChildList children_of_kind(Kind kind) const;

    /// Index operator to get nth child.
    ref<const DeclReflection> operator[](uint32_t index) const
    {
        return detail::from_slang(m_owner, slang_target()->getChild(index));
    }

    /// Description as string.
    std::string to_string() const override;

    /// Get type corresponding to this decl ref.
    ref<const TypeReflection> as_type() const;

    /// Get variable corresponding to this decl ref.
    ref<const VariableReflection> as_variable() const
    {
        return detail::from_slang(m_owner, slang_target()->asVariable());
    }

    /// Get function corresponding to this decl ref.
    ref<const FunctionReflection> as_function() const
    {
        return detail::from_slang(m_owner, slang_target()->asFunction());
    }

    /// Finds all children of a specific kind with a given name.
    /// Note: Only supported for types, functions and variables.
    DeclReflectionIndexedChildList find_children_of_kind(Kind kind, std::string_view child_name) const;

    /// Finds the first child of a specific kind with a given name.
    /// Note: Only supported for types, functions and variables.
    ref<const DeclReflection> find_first_child_of_kind(Kind kind, std::string_view child_name) const;
};
SGL_ENUM_REGISTER(DeclReflection::Kind);

/// DeclReflection lazy child list evaluation.
class SGL_API DeclReflectionChildList : public BaseReflectionList<DeclReflection, DeclReflection> {
public:
    DeclReflectionChildList(ref<const DeclReflection> owner)
        : BaseReflectionList(std::move(owner))
    {
    }

    /// Number of entries in list.
    uint32_t size() const override { return m_owner->child_count(); }

protected:
    /// Get a specific search result.
    ref<const DeclReflection> evaluate(uint32_t index) const override { return m_owner->child(index); }
};

/// DeclReflection lazy search result evaluation.
class SGL_API DeclReflectionIndexedChildList : public BaseReflectionIndexedList<DeclReflection, DeclReflection> {
public:
    DeclReflectionIndexedChildList(ref<const DeclReflection> owner, std::vector<uint32_t> results)
        : BaseReflectionIndexedList(std::move(owner), std::move(results))
    {
    }

protected:
    /// Get a specific search result.
    ref<const DeclReflection> evaluate(uint32_t index) const override { return m_owner->child(index); }
};

class SGL_API TypeReflection : public BaseReflectionObjectImpl<slang::TypeReflection> {
    SGL_OBJECT(TypeReflection)
public:
    enum class Kind {
        none = SLANG_TYPE_KIND_NONE,
        struct_ = SLANG_TYPE_KIND_STRUCT,
        array = SLANG_TYPE_KIND_ARRAY,
        matrix = SLANG_TYPE_KIND_MATRIX,
        vector = SLANG_TYPE_KIND_VECTOR,
        scalar = SLANG_TYPE_KIND_SCALAR,
        constant_buffer = SLANG_TYPE_KIND_CONSTANT_BUFFER,
        resource = SLANG_TYPE_KIND_RESOURCE,
        sampler_state = SLANG_TYPE_KIND_SAMPLER_STATE,
        texture_buffer = SLANG_TYPE_KIND_TEXTURE_BUFFER,
        shader_storage_buffer = SLANG_TYPE_KIND_SHADER_STORAGE_BUFFER,
        parameter_block = SLANG_TYPE_KIND_PARAMETER_BLOCK,
        generic_type_parameter = SLANG_TYPE_KIND_GENERIC_TYPE_PARAMETER,
        interface = SLANG_TYPE_KIND_INTERFACE,
        output_stream = SLANG_TYPE_KIND_OUTPUT_STREAM,
        specialized = SLANG_TYPE_KIND_SPECIALIZED,
        feedback = SLANG_TYPE_KIND_FEEDBACK,
        pointer = SLANG_TYPE_KIND_POINTER,
    };

    SGL_ENUM_INFO(
        Kind,
        {
            {Kind::none, "none"},
            {Kind::struct_, "struct"},
            {Kind::array, "array"},
            {Kind::matrix, "matrix"},
            {Kind::vector, "vector"},
            {Kind::scalar, "scalar"},
            {Kind::constant_buffer, "constant_buffer"},
            {Kind::resource, "resource"},
            {Kind::sampler_state, "sampler_state"},
            {Kind::texture_buffer, "texture_buffer"},
            {Kind::shader_storage_buffer, "shader_storage_buffer"},
            {Kind::parameter_block, "parameter_block"},
            {Kind::generic_type_parameter, "generic_type_parameter"},
            {Kind::interface, "interface"},
            {Kind::output_stream, "output_stream"},
            {Kind::specialized, "specialized"},
            {Kind::feedback, "feedback"},
            {Kind::pointer, "pointer"},
        }
    );

    enum class ScalarType {
        none_ = SLANG_SCALAR_TYPE_NONE,
        void_ = SLANG_SCALAR_TYPE_VOID,
        bool_ = SLANG_SCALAR_TYPE_BOOL,
        int32 = SLANG_SCALAR_TYPE_INT32,
        uint32 = SLANG_SCALAR_TYPE_UINT32,
        int64 = SLANG_SCALAR_TYPE_INT64,
        uint64 = SLANG_SCALAR_TYPE_UINT64,
        float16 = SLANG_SCALAR_TYPE_FLOAT16,
        float32 = SLANG_SCALAR_TYPE_FLOAT32,
        float64 = SLANG_SCALAR_TYPE_FLOAT64,
        int8 = SLANG_SCALAR_TYPE_INT8,
        uint8 = SLANG_SCALAR_TYPE_UINT8,
        int16 = SLANG_SCALAR_TYPE_INT16,
        uint16 = SLANG_SCALAR_TYPE_UINT16,
        intptr = SLANG_SCALAR_TYPE_INTPTR,
        uintptr = SLANG_SCALAR_TYPE_UINTPTR,
        COUNT,
    };

    SGL_ENUM_INFO(
        ScalarType,
        {
            {ScalarType::none_, "none"},
            {ScalarType::void_, "void"},
            {ScalarType::bool_, "bool"},
            {ScalarType::int32, "int32"},
            {ScalarType::uint32, "uint32"},
            {ScalarType::int64, "int64"},
            {ScalarType::uint64, "uint64"},
            {ScalarType::float16, "float16"},
            {ScalarType::float32, "float32"},
            {ScalarType::float64, "float64"},
            {ScalarType::int8, "int8"},
            {ScalarType::uint8, "uint8"},
            {ScalarType::int16, "int16"},
            {ScalarType::uint16, "uint16"},
        }
    );

    enum class ResourceShape {
        base_shape_mask = SLANG_RESOURCE_BASE_SHAPE_MASK,

        none = SLANG_RESOURCE_NONE,
        texture_1d = SLANG_TEXTURE_1D,
        texture_2d = SLANG_TEXTURE_2D,
        texture_3d = SLANG_TEXTURE_3D,
        texture_cube = SLANG_TEXTURE_CUBE,
        texture_buffer = SLANG_TEXTURE_BUFFER,

        structured_buffer = SLANG_STRUCTURED_BUFFER,
        byte_address_buffer = SLANG_BYTE_ADDRESS_BUFFER,
        unknown = SLANG_RESOURCE_UNKNOWN,
        acceleration_structure = SLANG_ACCELERATION_STRUCTURE,

        ext_shape_mask = SLANG_RESOURCE_EXT_SHAPE_MASK,

        texture_feedback_flag = SLANG_TEXTURE_FEEDBACK_FLAG,
        texture_array_flag = SLANG_TEXTURE_ARRAY_FLAG,
        texture_multisample_flag = SLANG_TEXTURE_MULTISAMPLE_FLAG,

        texture_1d_array = texture_1d | texture_array_flag,
        texture_2d_array = texture_2d | texture_array_flag,
        texture_cube_array = texture_cube | texture_array_flag,

        texture_2d_multisample = texture_2d | texture_multisample_flag,
        texture_2d_multisample_array = texture_2d | texture_multisample_flag | texture_array_flag,
    };

    SGL_ENUM_INFO(
        ResourceShape,
        {
            {ResourceShape::none, "none"},
            {ResourceShape::texture_1d, "texture_1d"},
            {ResourceShape::texture_2d, "texture_2d"},
            {ResourceShape::texture_3d, "texture_3d"},
            {ResourceShape::texture_cube, "texture_cube"},
            {ResourceShape::texture_buffer, "texture_buffer"},
            {ResourceShape::structured_buffer, "structured_buffer"},
            {ResourceShape::byte_address_buffer, "byte_address_buffer"},
            {ResourceShape::unknown, "unknown"},
            {ResourceShape::acceleration_structure, "acceleration_structure"},
            {ResourceShape::texture_feedback_flag, "texture_feedback_flag"},
            {ResourceShape::texture_array_flag, "texture_array_flag"},
            {ResourceShape::texture_multisample_flag, "texture_multisample_flag"},
            {ResourceShape::texture_1d_array, "texture_1d_array"},
            {ResourceShape::texture_2d_array, "texture_2d_array"},
            {ResourceShape::texture_cube_array, "texture_cube_array"},
            {ResourceShape::texture_2d_multisample, "texture_2d_multisample"},
            {ResourceShape::texture_2d_multisample_array, "texture_2d_multisample_array"},
        }
    );

    enum class ResourceAccess {
        none = SLANG_RESOURCE_ACCESS_NONE,
        read = SLANG_RESOURCE_ACCESS_READ,
        read_write = SLANG_RESOURCE_ACCESS_READ_WRITE,
        raster_ordered = SLANG_RESOURCE_ACCESS_RASTER_ORDERED,
        access_append = SLANG_RESOURCE_ACCESS_APPEND,
        access_consume = SLANG_RESOURCE_ACCESS_CONSUME,
        access_write = SLANG_RESOURCE_ACCESS_WRITE,
    };

    SGL_ENUM_INFO(
        ResourceAccess,
        {
            {ResourceAccess::none, "none"},
            {ResourceAccess::read, "read"},
            {ResourceAccess::read_write, "read_write"},
            {ResourceAccess::raster_ordered, "raster_ordered"},
            {ResourceAccess::access_append, "access_append"},
            {ResourceAccess::access_consume, "access_consume"},
            {ResourceAccess::access_write, "access_write"},
        }
    );

    enum class ParameterCategory {
        none = SLANG_PARAMETER_CATEGORY_NONE,
        mixed = SLANG_PARAMETER_CATEGORY_MIXED,
        constant_buffer = SLANG_PARAMETER_CATEGORY_CONSTANT_BUFFER,
        shader_resource = SLANG_PARAMETER_CATEGORY_SHADER_RESOURCE,
        unordered_access = SLANG_PARAMETER_CATEGORY_UNORDERED_ACCESS,
        varying_input = SLANG_PARAMETER_CATEGORY_VARYING_INPUT,
        varying_output = SLANG_PARAMETER_CATEGORY_VARYING_OUTPUT,
        sampler_state = SLANG_PARAMETER_CATEGORY_SAMPLER_STATE,
        uniform = SLANG_PARAMETER_CATEGORY_UNIFORM,
        descriptor_table_slot = SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT,
        specialization_constant = SLANG_PARAMETER_CATEGORY_SPECIALIZATION_CONSTANT,
        push_constant_buffer = SLANG_PARAMETER_CATEGORY_PUSH_CONSTANT_BUFFER,
        register_space = SLANG_PARAMETER_CATEGORY_REGISTER_SPACE,
        generic = SLANG_PARAMETER_CATEGORY_GENERIC,
        ray_payload = SLANG_PARAMETER_CATEGORY_RAY_PAYLOAD,
        hit_attributes = SLANG_PARAMETER_CATEGORY_HIT_ATTRIBUTES,
        callable_payload = SLANG_PARAMETER_CATEGORY_CALLABLE_PAYLOAD,
        shader_record = SLANG_PARAMETER_CATEGORY_SHADER_RECORD,
        existential_type_param = SLANG_PARAMETER_CATEGORY_EXISTENTIAL_TYPE_PARAM,
        existential_object_param = SLANG_PARAMETER_CATEGORY_EXISTENTIAL_OBJECT_PARAM,
    };

    SGL_ENUM_INFO(
        ParameterCategory,
        {
            {ParameterCategory::none, "none"},
            {ParameterCategory::mixed, "mixed"},
            {ParameterCategory::constant_buffer, "constant_buffer"},
            {ParameterCategory::shader_resource, "shader_resource"},
            {ParameterCategory::unordered_access, "unordered_access"},
            {ParameterCategory::varying_input, "varying_input"},
            {ParameterCategory::varying_output, "varying_output"},
            {ParameterCategory::sampler_state, "sampler_state"},
            {ParameterCategory::uniform, "uniform"},
            {ParameterCategory::descriptor_table_slot, "descriptor_table_slot"},
            {ParameterCategory::specialization_constant, "specialization_constant"},
            {ParameterCategory::push_constant_buffer, "push_constant_buffer"},
            {ParameterCategory::register_space, "register_space"},
            {ParameterCategory::generic, "generic"},
            {ParameterCategory::ray_payload, "ray_payload"},
            {ParameterCategory::hit_attributes, "hit_attributes"},
            {ParameterCategory::callable_payload, "callable_payload"},
            {ParameterCategory::shader_record, "shader_record"},
            {ParameterCategory::existential_type_param, "existential_type_param"},
            {ParameterCategory::existential_object_param, "existential_object_param"},
        }
    );

    TypeReflection(ref<const Object> owner, slang::TypeReflection* target)
        : BaseReflectionObjectImpl(std::move(owner), target)
    {
    }

    Kind kind() const { return static_cast<Kind>(slang_target()->getKind()); }

    const char* name() const { return slang_target()->getName(); }

    std::string full_name() const;

    bool is_struct() const { return kind() == Kind::struct_; }

    uint32_t field_count() const
    {
        SGL_CHECK(is_struct(), "Type is not a struct");
        return slang_target()->getFieldCount();
    }

    ref<const VariableReflection> get_field_by_index(uint32_t index) const
    {
        SGL_CHECK(is_struct(), "Type is not a struct");
        return detail::from_slang(m_owner, slang_target()->getFieldByIndex(index));
    }

    TypeReflectionFieldList fields() const;

    bool is_array() const { return kind() == Kind::array; }

    size_t element_count() const
    {
        SGL_CHECK(is_array(), "Type is not an array");
        return slang_target()->getElementCount();
    }

    size_t total_element_count() const
    {
        SGL_CHECK(is_array(), "Type is not an array");
        size_t result = 1;
        ref<const TypeReflection> type = ref(this);
        while (type->is_array()) {
            result *= type->element_count();
            type = type->element_type();
        }
        return result;
    }

    ref<const TypeReflection> element_type() const
    {
        SGL_CHECK(is_array(), "Type is not an array");
        return detail::from_slang(m_owner, slang_target()->getElementType());
    }

    ref<const TypeReflection> unwrap_array() const
    {
        ref<const TypeReflection> type = ref(this);
        while (type->is_array()) {
            type = type->element_type();
        }
        return type;
    }

    uint32_t row_count() const { return slang_target()->getRowCount(); }

    uint32_t col_count() const { return slang_target()->getColumnCount(); }

    ScalarType scalar_type() const { return static_cast<ScalarType>(slang_target()->getScalarType()); }

    ref<const TypeReflection> resource_result_type() const
    {
        return detail::from_slang(m_owner, slang_target()->getResourceResultType());
    }

    ResourceShape resource_shape() const { return static_cast<ResourceShape>(slang_target()->getResourceShape()); }

    ResourceAccess resource_access() const { return static_cast<ResourceAccess>(slang_target()->getResourceAccess()); }

#if 0
    unsigned int getUserAttributeCount() { return spReflectionType_GetUserAttributeCount((SlangReflectionType*)this); }
    UserAttribute* getUserAttributeByIndex(unsigned int index)
    {
        return (UserAttribute*)spReflectionType_GetUserAttribute((SlangReflectionType*)this, index);
    }
    UserAttribute* findUserAttributeByName(char const* name)
    {
        return (UserAttribute*)spReflectionType_FindUserAttributeByName((SlangReflectionType*)this, name);
    }
#endif

    std::string to_string() const override;
};

SGL_ENUM_CLASS_OPERATORS(TypeReflection::ResourceShape);

SGL_ENUM_REGISTER(TypeReflection::Kind);
SGL_ENUM_REGISTER(TypeReflection::ScalarType);
SGL_ENUM_REGISTER(TypeReflection::ResourceShape);
SGL_ENUM_REGISTER(TypeReflection::ResourceAccess);
SGL_ENUM_REGISTER(TypeReflection::ParameterCategory);

/// TypeReflection lazy field list evaluation.
class SGL_API TypeReflectionFieldList : public BaseReflectionList<TypeReflection, VariableReflection> {

public:
    TypeReflectionFieldList(ref<const TypeReflection> owner)
        : BaseReflectionList(std::move(owner))
    {
    }

    /// Number of entries in list.
    uint32_t size() const override { return m_owner->field_count(); }

protected:
    /// Get a specific child.
    ref<const VariableReflection> evaluate(uint32_t index) const override { return m_owner->get_field_by_index(index); }
};


class SGL_API TypeLayoutReflection : public BaseReflectionObjectImpl<slang::TypeLayoutReflection> {
    SGL_OBJECT(TypeLayoutReflection)
public:
    static ref<const TypeLayoutReflection>
    from_slang(ref<const Object> owner, slang::TypeLayoutReflection* type_layout_reflection)
    {
        return detail::from_slang(std::move(owner), type_layout_reflection);
    }

    TypeLayoutReflection(ref<const Object> owner, slang::TypeLayoutReflection* target)
        : BaseReflectionObjectImpl(std::move(owner), target)
    {
    }

    slang::TypeLayoutReflection* get_slang_type_layout() const { return slang_target(); }

    ref<const TypeReflection> type() const { return detail::from_slang(m_owner, slang_target()->getType()); }

    TypeReflection::Kind kind() const { return static_cast<TypeReflection::Kind>(slang_target()->getKind()); }

    const char* name() const { return slang_target()->getName(); }

    size_t size() const { return slang_target()->getSize(SlangParameterCategory::SLANG_PARAMETER_CATEGORY_UNIFORM); }
    size_t stride() const
    {
        return slang_target()->getStride(SlangParameterCategory::SLANG_PARAMETER_CATEGORY_UNIFORM);
    }
    int32_t alignment() const
    {
        return slang_target()->getAlignment(SlangParameterCategory::SLANG_PARAMETER_CATEGORY_UNIFORM);
    }

    uint32_t field_count() const { return slang_target()->getFieldCount(); }

    ref<const VariableLayoutReflection> get_field_by_index(uint32_t index) const
    {
        // TODO field_count() does not always return the right number of fields
        // SGL_CHECK(index < field_count(), "Field index out of range");
        return detail::from_slang(m_owner, slang_target()->getFieldByIndex(index));
    }

    int32_t find_field_index_by_name(const char* name_begin, const char* name_end = nullptr) const
    {
        return narrow_cast<int32_t>(slang_target()->findFieldIndexByName(name_begin, name_end));
    }

    ref<const VariableLayoutReflection> find_field_by_name(const char* name_begin, const char* name_end = nullptr) const
    {
        if (auto index = slang_target()->findFieldIndexByName(name_begin, name_end); index >= 0)
            return detail::from_slang(m_owner, slang_target()->getFieldByIndex(narrow_cast<unsigned int>(index)));
        return nullptr;
    }

    TypeLayoutReflectionFieldList fields() const;

    bool is_array() const { return kind() == TypeReflection::Kind::array; }

    ref<const TypeLayoutReflection> unwrap_array() const
    {
        ref<const TypeLayoutReflection> type_layout = ref(this);
        while (type_layout->is_array()) {
            type_layout = type_layout->element_type_layout();
        }
        return type_layout;
    }

    size_t element_count() const { return slang_target()->getElementCount(); }

    size_t element_stride(TypeReflection::ParameterCategory category = TypeReflection::ParameterCategory::uniform) const
    {
        return slang_target()->getElementStride(static_cast<SlangParameterCategory>(category));
    }

    ref<const TypeLayoutReflection> element_type_layout() const
    {
        return detail::from_slang(m_owner, slang_target()->getElementTypeLayout());
    }

    ref<const VariableLayoutReflection> element_var_layout() const
    {
        return detail::from_slang(m_owner, slang_target()->getElementVarLayout());
    }

    ref<const VariableLayoutReflection> container_var_layout() const
    {
        return detail::from_slang(m_owner, slang_target()->getContainerVarLayout());
    }

    // How is this type supposed to be bound?
    TypeReflection::ParameterCategory parameter_category() const
    {
        return static_cast<TypeReflection::ParameterCategory>(slang_target()->getParameterCategory());
    }

    uint32_t get_field_binding_range_offset(uint32_t field_index) const
    {
        return narrow_cast<uint32_t>(slang_target()->getFieldBindingRangeOffset(field_index));
    }

    std::string to_string() const override;
};

/// TypeLayoutReflection lazy field list evaluation.
class SGL_API TypeLayoutReflectionFieldList
    : public BaseReflectionList<TypeLayoutReflection, VariableLayoutReflection> {

public:
    TypeLayoutReflectionFieldList(ref<const TypeLayoutReflection> owner)
        : BaseReflectionList(std::move(owner))
    {
    }

    /// Number of entries in list.
    uint32_t size() const override { return m_owner->field_count(); }

protected:
    /// Get a specific child.
    ref<const VariableLayoutReflection> evaluate(uint32_t index) const override
    {
        return m_owner->get_field_by_index(index);
    }
};

class SGL_API FunctionReflection : public BaseReflectionObjectImpl<slang::FunctionReflection> {
    SGL_OBJECT(FunctionReflection)
public:
    FunctionReflection(ref<const Object> owner, slang::FunctionReflection* target)
        : BaseReflectionObjectImpl(std::move(owner), target)
    {
    }

    /// Function name.
    const char* name() const { return slang_target()->getName(); }

    /// Function return type.
    ref<const TypeReflection> return_type() { return detail::from_slang(m_owner, slang_target()->getReturnType()); }

    /// Number of function parameters.
    uint32_t parameter_count() const { return slang_target()->getParameterCount(); }

    /// Get a single parameter.
    ref<const VariableReflection> get_parameter_by_index(uint32_t index) const
    {
        SGL_CHECK(index < parameter_count(), "Parameter index out of range");
        return detail::from_slang(m_owner, slang_target()->getParameterByIndex(index));
    }

    /// List of all function parameters.
    FunctionReflectionParameterList parameters() const;

    /// Check if the function has a given modifier (e.g. 'differentiable').
    bool has_modifier(ModifierID modifier) const
    {
        return slang_target()->findModifier(static_cast<slang::Modifier::ID>(modifier)) != nullptr;
    }

    /// Specialize a generic or interface based function with a set of concrete
    /// argument types. Calling on a none-generic/interface function will simply
    /// validate all argument types can be implicitly converted to their respective
    /// parameter types. Where a function contains multiple overloads, specialize
    /// will identify the correct overload based on the arguments.
    ref<const FunctionReflection> specialize_with_arg_types(const std::vector<ref<TypeReflection>>& types) const
    {
        if (types.empty()) {
            return detail::from_slang(m_owner, slang_target()->specializeWithArgTypes(0, nullptr));
        } else {
            std::vector<slang::TypeReflection*> slang_types;
            slang_types.reserve(types.size());
            for (const auto& type : types) {
                SGL_CHECK(type, "Null type provided to specialize_with_arg_types");
                slang_types.push_back(type->slang_target());
            }

            return detail::from_slang(
                m_owner,
                slang_target()->specializeWithArgTypes((uint32_t)slang_types.size(), &slang_types[0])
            );
        }
    }

    /// Check whether this function object represents a group of overloaded functions,
    /// accessible via the overloads list.
    bool is_overloaded() const { return slang_target()->isOverloaded(); }

    /// Get number of available overloads of this function.
    uint32_t overload_count() const { return is_overloaded() ? slang_target()->getOverloadCount() : 0; }

    /// Get a given overload of this function.
    ref<const FunctionReflection> get_overload_by_index(uint32_t index) const
    {
        return detail::from_slang(m_owner, slang_target()->getOverload(index));
    }

    /// List of all overloads of this function.
    FunctionReflectionOverloadList overloads() const;
};

/// FunctionReflection lazy parameter list evaluation.
class SGL_API FunctionReflectionParameterList : public BaseReflectionList<FunctionReflection, VariableReflection> {

public:
    FunctionReflectionParameterList(ref<const FunctionReflection> owner)
        : BaseReflectionList(std::move(owner))
    {
    }

    /// Number of entries in list.
    uint32_t size() const override { return m_owner->parameter_count(); }

protected:
    /// Get a specific child.
    ref<const VariableReflection> evaluate(uint32_t index) const override
    {
        return m_owner->get_parameter_by_index(index);
    }
};

/// FunctionReflection lazy overload list evaluation.
class SGL_API FunctionReflectionOverloadList : public BaseReflectionList<FunctionReflection, FunctionReflection> {

public:
    FunctionReflectionOverloadList(ref<const FunctionReflection> owner)
        : BaseReflectionList(std::move(owner))
    {
    }

    /// Number of entries in list.
    uint32_t size() const override { return m_owner->overload_count(); }

protected:
    /// Get a specific child.
    ref<const FunctionReflection> evaluate(uint32_t index) const override
    {
        return m_owner->get_overload_by_index(index);
    }
};

class SGL_API VariableReflection : public BaseReflectionObjectImpl<slang::VariableReflection> {
    SGL_OBJECT(VariableReflection)
public:
    static ref<const VariableReflection>
    from_slang(ref<const Object> owner, slang::VariableReflection* variable_reflection)
    {
        return detail::from_slang(std::move(owner), variable_reflection);
    }

    VariableReflection(ref<const Object> owner, slang::VariableReflection* target)
        : BaseReflectionObjectImpl(std::move(owner), target)
    {
    }

    /// Variable name.
    const char* name() const { return slang_target()->getName(); }

    /// Variable type reflection.
    ref<const TypeReflection> type() const { return detail::from_slang(m_owner, slang_target()->getType()); }

    /// Check if variable has a given modifier (e.g. 'inout').
    bool has_modifier(ModifierID modifier) const
    {
        return slang_target()->findModifier(static_cast<slang::Modifier::ID>(modifier)) != nullptr;
    }
};

class SGL_API VariableLayoutReflection : public BaseReflectionObjectImpl<slang::VariableLayoutReflection> {
public:
    VariableLayoutReflection(ref<const Object> owner, slang::VariableLayoutReflection* target)
        : BaseReflectionObjectImpl(std::move(owner), target)
    {
    }

    ref<const VariableReflection> variable() const
    {
        return detail::from_slang(m_owner, slang_target()->getVariable());
    }

    const char* name() const { return slang_target()->getName(); }

    ref<const TypeLayoutReflection> type_layout() const
    {
        return detail::from_slang(m_owner, slang_target()->getTypeLayout());
    }

    size_t offset() const
    {
        return slang_target()->getOffset(SlangParameterCategory::SLANG_PARAMETER_CATEGORY_UNIFORM);
    }

    std::string to_string() const override;
};

class SGL_API EntryPointLayout : public BaseReflectionObjectImpl<slang::EntryPointLayout> {
    SGL_OBJECT(EntryPointLayout)
public:
    static ref<const EntryPointLayout>
    from_slang(ref<const Object> owner, slang::EntryPointLayout* entry_point_reflection)
    {
        return detail::from_slang(std::move(owner), entry_point_reflection);
    }

    EntryPointLayout(ref<const Object> owner, slang::EntryPointLayout* target)
        : BaseReflectionObjectImpl(std::move(owner), target)
    {
    }

    const char* name() const { return slang_target()->getName(); }

    const char* name_override() const { return slang_target()->getNameOverride(); }

    ShaderStage stage() const { return static_cast<ShaderStage>(slang_target()->getStage()); }

    uint32_t parameter_count() const { return slang_target()->getParameterCount(); }

    ref<const VariableLayoutReflection> get_parameter_by_index(uint32_t index) const
    {
        SGL_CHECK(index < parameter_count(), "Parameter index out of range");
        return detail::from_slang(m_owner, slang_target()->getParameterByIndex(index));
    }

    EntryPointLayoutParameterList parameters() const;

    uint3 compute_thread_group_size() const
    {
        SlangUInt size[3];
        slang_target()->getComputeThreadGroupSize(3, size);
        return uint3(narrow_cast<uint32_t>(size[0]), narrow_cast<uint32_t>(size[1]), narrow_cast<uint32_t>(size[2]));
    }

    bool uses_any_sample_rate_input() const { return slang_target()->usesAnySampleRateInput(); }

    std::string to_string() const override;
};


/// EntryPointLayout lazy parameter list evaluation.
class SGL_API EntryPointLayoutParameterList : public BaseReflectionList<EntryPointLayout, VariableLayoutReflection> {

public:
    EntryPointLayoutParameterList(ref<const EntryPointLayout> owner)
        : BaseReflectionList(std::move(owner))
    {
    }

    /// Number of entries in list.
    uint32_t size() const override { return m_owner->parameter_count(); }

protected:
    /// Get a specific child.
    ref<const VariableLayoutReflection> evaluate(uint32_t index) const override
    {
        return m_owner->get_parameter_by_index(index);
    }
};


class SGL_API ProgramLayout : public BaseReflectionObjectImpl<slang::ProgramLayout> {
    SGL_OBJECT(ProgramLayout)
public:
    static ref<const ProgramLayout> from_slang(ref<const Object> owner, slang::ProgramLayout* program_layout)
    {
        return detail::from_slang(owner, program_layout);
    }

    ProgramLayout(ref<const Object> owner, slang::ProgramLayout* target)
        : BaseReflectionObjectImpl(std::move(owner), target)
    {
    }

    ref<const TypeLayoutReflection> globals_type_layout() const
    {
        return detail::from_slang(m_owner, slang_target()->getGlobalParamsTypeLayout());
    }

    ref<const VariableLayoutReflection> globals_variable_layout() const
    {
        return detail::from_slang(m_owner, slang_target()->getGlobalParamsVarLayout());
    }

    uint32_t parameter_count() const { return slang_target()->getParameterCount(); }

    ref<const VariableLayoutReflection> get_parameter_by_index(uint32_t index) const
    {
        SGL_CHECK(index < parameter_count(), "Parameter index out of range");
        return detail::from_slang(m_owner, slang_target()->getParameterByIndex(index));
    }

    ProgramLayoutParameterList parameters() const;

    uint32_t type_parameter_count() const { return slang_target()->getTypeParameterCount(); }

    uint32_t entry_point_count() const { return narrow_cast<uint32_t>(slang_target()->getEntryPointCount()); }

    ref<const EntryPointLayout> get_entry_point_by_index(uint32_t index) const
    {
        SGL_CHECK(index < entry_point_count(), "Entry point index out of range");
        return detail::from_slang(m_owner, slang_target()->getEntryPointByIndex(index));
    }

    ref<const EntryPointLayout> get_entry_point_by_name(const char* name) const
    {
        return detail::from_slang(m_owner, slang_target()->findEntryPointByName(name));
    }

    ref<const EntryPointLayout> get_entry_point_by_name(std::string_view name) const
    {
        std::string name_str(name);
        return get_entry_point_by_name(name_str.c_str());
    }

    ref<const EntryPointLayout> find_entry_point_by_name(const char* name) const
    {
        if (auto entry_point = slang_target()->findEntryPointByName(name)) {
            return detail::from_slang(m_owner, entry_point);
        }
        return nullptr;
    }

    ref<const EntryPointLayout> find_entry_point_by_name(std::string_view name) const
    {
        std::string name_str(name);
        return find_entry_point_by_name(name_str.c_str());
    }

    ProgramLayoutEntryPointList entry_points() const;

    /// Find a given type by name. Handles generic specilization if generic
    /// variable values are provided.
    ref<const TypeReflection> find_type_by_name(const char* name) const
    {
        return detail::from_slang(m_owner, slang_target()->findTypeByName(name));
    }

    /// Find a given function by name. Handles generic specilization if generic
    /// variable values are provided.
    ref<const FunctionReflection> find_function_by_name(const char* name)
    {
        return detail::from_slang(m_owner, slang_target()->findFunctionByName(name));
    }

    /// Find a given function in a type by name. Handles generic specilization if generic
    /// variable values are provided.
    ref<const FunctionReflection> find_function_by_name_in_type(const TypeReflection* type, const char* name)
    {
        return detail::from_slang(m_owner, slang_target()->findFunctionByNameInType(type->slang_target(), name));
    }

    /// Test whether a type is a sub type of another type. Handles both
    /// struct inheritance and interface implementation.
    bool is_sub_type(const TypeReflection* sub_type, const TypeReflection* super_type)
    {
        return slang_target()->isSubType(sub_type->slang_target(), super_type->slang_target());
    }

    /// Get corresponding type layout from a given type.
    ref<const TypeLayoutReflection> get_type_layout(const TypeReflection* type)
    {
        // TODO: Once device is available via session reference, pass metal layout rules for metal target
        return detail::from_slang(
            m_owner,
            slang_target()->getTypeLayout(type->slang_target(), slang::LayoutRules::Default)
        );
    }

    uint32_t hashed_string_count() const { return narrow_cast<uint32_t>(slang_target()->getHashedStringCount()); }

    struct HashedString {
        std::string string;
        uint32_t hash;
    };

    HashedString get_hashed_string(uint32_t index) const
    {
        SGL_CHECK(index < hashed_string_count(), "Hashed string index out of range");
        size_t size;
        const char* str = slang_target()->getHashedString(index, &size);
        return {std::string(str, str + size), spComputeStringHash(str, size)};
    }

    std::vector<HashedString> hashed_strings() const
    {
        std::vector<HashedString> result;
        for (uint32_t i = 0; i < slang_target()->getHashedStringCount(); ++i) {
            size_t size;
            const char* str = slang_target()->getHashedString(i, &size);
            result.push_back({std::string(str, str + size), spComputeStringHash(str, size)});
        }
        return result;
    }

    std::map<uint32_t, std::string> hashed_strings_map() const
    {
        std::map<uint32_t, std::string> result;
        for (uint32_t i = 0; i < slang_target()->getHashedStringCount(); ++i) {
            size_t size;
            const char* str = slang_target()->getHashedString(i, &size);
            uint32_t hash = spComputeStringHash(str, size);
            result.emplace(hash, std::string(str, str + size));
        }
        return result;
    }

    std::string to_string() const override;
};

/// ProgramLayout lazy parameter list evaluation.
class SGL_API ProgramLayoutParameterList : public BaseReflectionList<ProgramLayout, VariableLayoutReflection> {

public:
    ProgramLayoutParameterList(ref<const ProgramLayout> owner)
        : BaseReflectionList(std::move(owner))
    {
    }

    /// Number of entries in list.
    uint32_t size() const override { return m_owner->parameter_count(); }

protected:
    /// Get a specific child.
    ref<const VariableLayoutReflection> evaluate(uint32_t index) const override
    {
        return m_owner->get_parameter_by_index(index);
    }
};

/// ProgramLayout lazy entry point list evaluation.
class SGL_API ProgramLayoutEntryPointList : public BaseReflectionList<ProgramLayout, EntryPointLayout> {

public:
    ProgramLayoutEntryPointList(ref<const ProgramLayout> owner)
        : BaseReflectionList(std::move(owner))
    {
    }

    /// Number of entries in list.
    uint32_t size() const override { return m_owner->entry_point_count(); }

protected:
    /// Get a specific child.
    ref<const EntryPointLayout> evaluate(uint32_t index) const override
    {
        return m_owner->get_entry_point_by_index(index);
    }
};


class ShaderProgram;

class SGL_API ReflectionCursor {
public:
    ReflectionCursor() = default;

    ReflectionCursor(const ShaderProgram* shader_program);
    ReflectionCursor(ref<const EntryPointLayout> entry_point_layout);
    ReflectionCursor(const TypeLayoutReflection* type_layout);

    ref<const TypeLayoutReflection> type_layout() const { return m_type_layout; }
    ref<const TypeReflection> type() const { return m_type_layout->type(); }

    bool is_valid() const { return m_valid; }

    // explicit operator bool() const { return is_valid(); }

    operator ref<const TypeLayoutReflection>() const { return type_layout(); }

    ReflectionCursor operator[](std::string_view name) const;
    ReflectionCursor operator[](uint32_t index) const;

    ReflectionCursor find_field(std::string_view name) const;
    ReflectionCursor find_element(uint32_t index) const;

    bool has_field(std::string_view name) const { return find_field(name).is_valid(); }
    bool has_element(uint32_t index) const { return find_element(index).is_valid(); }

    std::string to_string() const;

private:
    const ShaderProgram* m_shader_program{nullptr};
    ref<const EntryPointLayout> m_entry_point_layout{nullptr};
    ref<const TypeLayoutReflection> m_type_layout{nullptr};
    bool m_valid{false};
};

} // namespace sgl
