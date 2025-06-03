// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/device/cursor_utils.h"

namespace sgl {

namespace cursor_utils {

    // Helper class for checking if implicit conversion between scalar types is allowed.
    // Note that only conversion between types of the same size is allowed.
    struct ScalarConversionTable {
        static_assert(size_t(TypeReflection::ScalarType::COUNT) < 32, "Not enough bits to represent all scalar types");
        constexpr ScalarConversionTable()
        {
            for (uint32_t i = 0; i < uint32_t(TypeReflection::ScalarType::COUNT); ++i)
                table[i] = 1 << i;

            auto add_conversion = [&](TypeReflection::ScalarType from, auto... to)
            {
                uint32_t flags{0};
                ((flags |= 1 << uint32_t(to)), ...);
                table[uint32_t(from)] |= flags;
            };

            using ST = TypeReflection::ScalarType;
            add_conversion(ST::int32, ST::uint32, ST::bool_);
            add_conversion(ST::uint32, ST::int32, ST::bool_);
            add_conversion(ST::int64, ST::uint64);
            add_conversion(ST::uint64, ST::int64);
            add_conversion(ST::int8, ST::uint8);
            add_conversion(ST::uint8, ST::int8);
            add_conversion(ST::int16, ST::uint16);
            add_conversion(ST::uint16, ST::int16);
        }

        constexpr bool allow_conversion(TypeReflection::ScalarType from, TypeReflection::ScalarType to) const
        {
            return (table[uint32_t(from)] & (1 << uint32_t(to))) != 0;
        }

        uint32_t table[size_t(TypeReflection::ScalarType::COUNT)]{};
    };

    bool allow_scalar_conversion(TypeReflection::ScalarType from, TypeReflection::ScalarType to)
    {
        static constexpr ScalarConversionTable table;
        return table.allow_conversion(from, to);
    }

    size_t get_scalar_type_size(TypeReflection::ScalarType type)
    {
        switch (type) {
        case TypeReflection::ScalarType::int8:
        case TypeReflection::ScalarType::uint8:
            return 1;
        case TypeReflection::ScalarType::int16:
        case TypeReflection::ScalarType::uint16:
        case TypeReflection::ScalarType::float16:
            return 2;
        case TypeReflection::ScalarType::bool_:
        case TypeReflection::ScalarType::int32:
        case TypeReflection::ScalarType::uint32:
        case TypeReflection::ScalarType::float32:
            return 4;
        case TypeReflection::ScalarType::int64:
        case TypeReflection::ScalarType::uint64:
        case TypeReflection::ScalarType::float64:
            return 8;
        default:
            return 0;
        }
    }

    slang::TypeLayoutReflection* unwrap_array(slang::TypeLayoutReflection* layout)
    {
        while (layout->isArray()) {
            layout = layout->getElementTypeLayout();
        }
        return layout;
    }

    void check_array(
        slang::TypeLayoutReflection* type_layout,
        size_t size,
        TypeReflection::ScalarType scalar_type,
        size_t element_count
    )
    {
        slang::TypeReflection* type = type_layout->getType();
        slang::TypeReflection* element_type = unwrap_array(type_layout)->getType();
        size_t element_size = get_scalar_type_size((TypeReflection::ScalarType)element_type->getScalarType());

        SGL_CHECK(type->isArray(), "\"{}\" cannot bind an array", type_layout->getName());
        SGL_CHECK(
            allow_scalar_conversion(scalar_type, (TypeReflection::ScalarType)element_type->getScalarType()),
            "\"{}\" expects scalar type {} (no implicit conversion from type {})",
            type_layout->getName(),
            (TypeReflection::ScalarType)element_type->getScalarType(),
            scalar_type
        );
        SGL_CHECK(
            element_count <= type->getElementCount(),
            "\"{}\" expects an array with at most {} elements (got {})",
            type_layout->getName(),
            type->getElementCount(),
            element_count
        );
        SGL_ASSERT(element_count * element_size == size);
    }

    void check_scalar(slang::TypeLayoutReflection* type_layout, size_t size, TypeReflection::ScalarType scalar_type)
    {
        slang::TypeReflection* type = unwrap_array(type_layout)->getType();

        SGL_CHECK(
            (TypeReflection::Kind)type->getKind() == TypeReflection::Kind::scalar,
            "\"{}\" cannot bind a scalar value",
            type_layout->getName()
        );
        SGL_CHECK(
            allow_scalar_conversion(scalar_type, (TypeReflection::ScalarType)type->getScalarType()),
            "\"{}\" expects scalar type {} (no implicit conversion from type {})",
            type_layout->getName(),
            (TypeReflection::ScalarType)type->getScalarType(),
            scalar_type
        );
        SGL_UNUSED(size);
    }

    void check_vector(
        slang::TypeLayoutReflection* type_layout,
        size_t size,
        TypeReflection::ScalarType scalar_type,
        int dimension
    )
    {
        slang::TypeReflection* type = unwrap_array(type_layout)->getType();

        SGL_CHECK(
            (TypeReflection::Kind)type->getKind() == TypeReflection::Kind::vector,
            "\"{}\" cannot bind a vector value",
            type_layout->getName()
        );
        SGL_CHECK(
            type->getColumnCount() == uint32_t(dimension),
            "\"{}\" expects a vector with dimension {} (got dimension {})",
            type_layout->getName(),
            type->getColumnCount(),
            dimension
        );
        SGL_CHECK(
            allow_scalar_conversion(scalar_type, (TypeReflection::ScalarType)type->getScalarType()),
            "\"{}\" expects a vector with scalar type {} (no implicit conversion from type {})",
            type_layout->getName(),
            (TypeReflection::ScalarType)type->getScalarType(),
            scalar_type
        );
        SGL_UNUSED(size);
    }

    void check_matrix(
        slang::TypeLayoutReflection* type_layout,
        size_t size,
        TypeReflection::ScalarType scalar_type,
        int rows,
        int cols
    )
    {
        slang::TypeReflection* type = unwrap_array(type_layout)->getType();

        SGL_CHECK(
            (TypeReflection::Kind)type->getKind() == TypeReflection::Kind::matrix,
            "\"{}\" cannot bind a matrix value",
            type_layout->getName()
        );

        bool dimensionCondition = type->getRowCount() == uint32_t(rows) && type->getColumnCount() == uint32_t(cols);

        SGL_CHECK(
            dimensionCondition,
            "\"{}\" expects a matrix with dimension {}x{} (got dimension {}x{})",
            type_layout->getName(),
            type->getRowCount(),
            type->getColumnCount(),
            rows,
            cols
        );
        SGL_CHECK(
            allow_scalar_conversion(scalar_type, (TypeReflection::ScalarType)type->getScalarType()),
            "\"{}\" expects a matrix with scalar type {} (no implicit conversion from type {})",
            type_layout->getName(),
            (TypeReflection::ScalarType)type->getScalarType(),
            scalar_type
        );
        SGL_UNUSED(size);
    }


} // namespace cursor_utils


} // namespace sgl
