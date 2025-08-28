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
            add_conversion(ST::int32, ST::uint32);
            add_conversion(ST::uint32, ST::int32);
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
        TypeReflection::ScalarType src_scalar_type,
        size_t element_count
    )
    {
        SGL_CHECK(type_layout->isArray(), "\"{}\" cannot bind a non-array", type_layout->getName());

        slang::TypeLayoutReflection* element_type_layout = type_layout->getElementTypeLayout();
        size_t element_size = element_type_layout->getSize();
        size_t element_stride = type_layout->getElementStride(SlangParameterCategory::SLANG_PARAMETER_CATEGORY_UNIFORM);

        SGL_CHECK(
            allow_scalar_conversion(src_scalar_type, (TypeReflection::ScalarType)element_type_layout->getScalarType()),
            "\"{}\" expects scalar type {} (no implicit conversion from type {})",
            type_layout->getName(),
            (TypeReflection::ScalarType)element_type_layout->getScalarType(),
            src_scalar_type
        );
        SGL_CHECK(
            element_count <= type_layout->getElementCount(),
            "\"{}\" expects an array with at most {} elements (got {})",
            type_layout->getName(),
            type_layout->getElementCount(),
            element_count
        );
        SGL_CHECK(
            type_layout->getSize() >= size,
            "Mismatched size, writing {} B into backend type ({}) of only {} B.",
            size,
            type_layout->getName(),
            type_layout->getSize()
        );
    }

    void check_scalar(slang::TypeLayoutReflection* type_layout, size_t size, TypeReflection::ScalarType src_scalar_type)
    {
        SGL_CHECK(
            allow_scalar_conversion(src_scalar_type, (TypeReflection::ScalarType)type_layout->getScalarType()),
            "\"{}\" expects scalar type {} (no implicit conversion from type {})",
            type_layout->getName(),
            (TypeReflection::ScalarType)type_layout->getScalarType(),
            src_scalar_type
        );
        SGL_CHECK(
            type_layout->getSize() >= size,
            "Mismatched size, writing {} B into backend type ({}) of only {} B.",
            size,
            type_layout->getName(),
            type_layout->getSize()
        );
    }

    void check_vector(
        slang::TypeLayoutReflection* type_layout,
        size_t size,
        TypeReflection::ScalarType src_scalar_type,
        int dimension
    )
    {
        SGL_CHECK(
            (TypeReflection::Kind)type_layout->getKind() == TypeReflection::Kind::vector,
            "\"{}\" cannot bind a non-vector value",
            type_layout->getName()
        );
        SGL_CHECK(
            type_layout->getColumnCount() == uint32_t(dimension),
            "\"{}\" expects a vector with dimension {} (got dimension {})",
            type_layout->getName(),
            type_layout->getColumnCount(),
            dimension
        );
        SGL_CHECK(
            allow_scalar_conversion(src_scalar_type, (TypeReflection::ScalarType)type_layout->getScalarType()),
            "\"{}\" expects a vector with scalar type {} (no implicit conversion from type {})",
            type_layout->getName(),
            (TypeReflection::ScalarType)type_layout->getScalarType(),
            src_scalar_type
        );
        SGL_CHECK(
            type_layout->getSize() >= size,
            "Mismatched size, writing {} B into backend type ({}) of only {} B.",
            size,
            type_layout->getName(),
            type_layout->getSize()
        );
    }

    void check_matrix(
        slang::TypeLayoutReflection* type_layout,
        size_t size,
        TypeReflection::ScalarType src_scalar_type,
        int rows,
        int cols
    )
    {
        SGL_CHECK(
            (TypeReflection::Kind)type_layout->getKind() == TypeReflection::Kind::matrix,
            "\"{}\" cannot bind a non-matrix value",
            type_layout->getName()
        );

        bool dimensionCondition
            = type_layout->getRowCount() == uint32_t(rows) && type_layout->getColumnCount() == uint32_t(cols);

        SGL_CHECK(
            dimensionCondition,
            "\"{}\" expects a matrix with dimension {}x{} (got dimension {}x{})",
            type_layout->getName(),
            type_layout->getRowCount(),
            type_layout->getColumnCount(),
            rows,
            cols
        );
        SGL_CHECK(
            allow_scalar_conversion(src_scalar_type, (TypeReflection::ScalarType)type_layout->getScalarType()),
            "\"{}\" expects a matrix with scalar type {} (no implicit conversion from type {})",
            type_layout->getName(),
            (TypeReflection::ScalarType)type_layout->getScalarType(),
            src_scalar_type
        );
        SGL_CHECK(
            type_layout->getSize() >= size,
            "Mismatched size, writing {} B into backend type ({}) of only {} B.",
            size,
            type_layout->getName(),
            type_layout->getSize()
        );
    }


} // namespace cursor_utils


} // namespace sgl
