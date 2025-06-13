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
            // bool can be converted to 32 or 8b int types, but needs additional size check.
            add_conversion(ST::int32, ST::uint32, ST::bool_);
            add_conversion(ST::uint32, ST::int32, ST::bool_);
            add_conversion(ST::int64, ST::uint64);
            add_conversion(ST::uint64, ST::int64);
            add_conversion(ST::int8, ST::uint8, ST::bool_);
            add_conversion(ST::uint8, ST::int8, ST::bool_);
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
        slang::TypeLayoutReflection* dst_type_layout,
        size_t src_size,
        TypeReflection::ScalarType src_scalar_type,
        size_t src_element_count
    )
    {
        slang::TypeLayoutReflection* element_type_layout = unwrap_array(dst_type_layout);
        size_t dst_element_size = element_type_layout->getSize();
        size_t src_element_size = src_size / src_element_count;

        SGL_CHECK(dst_type_layout->isArray(), "\"{}\" cannot bind an array", dst_type_layout->getName());
        SGL_CHECK(
            dst_element_size == src_element_size
                && allow_scalar_conversion(
                    src_scalar_type,
                    (TypeReflection::ScalarType)element_type_layout->getScalarType()
                ),
            "\"{}\" expects scalar type {} ({}B) (no implicit conversion from type {} ({}B))",
            dst_type_layout->getName(),
            (TypeReflection::ScalarType)element_type_layout->getScalarType(),
            dst_element_size,
            src_scalar_type,
            src_element_size
        );
        SGL_CHECK(
            src_element_count == dst_type_layout->getElementCount(),
            "\"{}\" expects an array with at most {} elements (got {})",
            dst_type_layout->getName(),
            dst_type_layout->getElementCount(),
            src_element_count
        );
        SGL_ASSERT(src_element_count * dst_element_size == src_size);
    }

    void check_scalar(
        slang::TypeLayoutReflection* dst_type_layout,
        size_t src_size,
        TypeReflection::ScalarType src_scalar_type
    )
    {
        size_t dst_size = dst_type_layout->getSize();

        SGL_CHECK(
            (TypeReflection::Kind)dst_type_layout->getKind() == TypeReflection::Kind::scalar,
            "\"{}\" cannot bind a scalar value",
            dst_type_layout->getName()
        );
        SGL_CHECK(
            dst_size == src_size
                && allow_scalar_conversion(
                    src_scalar_type,
                    (TypeReflection::ScalarType)dst_type_layout->getScalarType()
                ),
            "\"{}\" expects scalar type {} ({}B) (no implicit conversion from type {} ({}B))",
            dst_type_layout->getName(),
            (TypeReflection::ScalarType)dst_type_layout->getScalarType(),
            dst_size,
            src_scalar_type,
            src_size
        );
        SGL_CHECK(
            src_size <= dst_type_layout->getSize(),
            "Mismatched size, writing {} B into backend type ({}) of only {} B.",
            src_size,
            dst_type_layout->getName(),
            dst_type_layout->getSize()
        );
    }

    void check_vector(
        slang::TypeLayoutReflection* dst_type_layout,
        size_t src_size,
        TypeReflection::ScalarType src_scalar_type,
        int src_dimension
    )
    {
        slang::TypeLayoutReflection* element_type_layout = dst_type_layout->getElementTypeLayout();
        size_t dst_element_size = element_type_layout->getSize();
        size_t src_element_size = src_size / src_dimension;

        SGL_CHECK(
            (TypeReflection::Kind)dst_type_layout->getKind() == TypeReflection::Kind::vector,
            "\"{}\" cannot bind a vector value",
            dst_type_layout->getName()
        );
        SGL_CHECK(
            dst_type_layout->getColumnCount() == uint32_t(src_dimension),
            "\"{}\" expects a vector with dimension {} (got dimension {})",
            dst_type_layout->getName(),
            dst_type_layout->getColumnCount(),
            src_dimension
        );
        SGL_CHECK(
            dst_element_size == src_element_size
                && allow_scalar_conversion(
                    src_scalar_type,
                    (TypeReflection::ScalarType)dst_type_layout->getScalarType()
                ),
            "\"{}\" expects a vector with scalar type {} ({}B) (no implicit conversion from type {} ({}B))",
            dst_type_layout->getName(),
            (TypeReflection::ScalarType)element_type_layout->getScalarType(),
            dst_element_size,
            src_scalar_type,
            src_element_size
        );
        SGL_CHECK(
            src_size <= dst_type_layout->getSize(),
            "Mismatched size, writing {} B into backend type ({}) of only {} B.",
            src_size,
            dst_type_layout->getName(),
            dst_type_layout->getSize()
        );
    }

    void check_matrix(
        slang::TypeLayoutReflection* dst_type_layout,
        size_t src_size,
        TypeReflection::ScalarType src_scalar_type,
        int src_rows,
        int src_cols
    )
    {
        // Element of `matrix` is a vector, so the `scalar` is element applied twice.
        slang::TypeLayoutReflection* element_type_layout
            = dst_type_layout->getElementTypeLayout()->getElementTypeLayout();
        size_t dst_element_size = element_type_layout->getSize();
        size_t src_element_size = src_size / (src_rows * src_cols);

        SGL_CHECK(
            (TypeReflection::Kind)dst_type_layout->getKind() == TypeReflection::Kind::matrix,
            "\"{}\" cannot bind a matrix value",
            dst_type_layout->getName()
        );

        bool dimensionCondition = dst_type_layout->getRowCount() == uint32_t(src_rows)
            && dst_type_layout->getColumnCount() == uint32_t(src_cols);

        SGL_CHECK(
            dimensionCondition,
            "\"{}\" expects a matrix with dimension {}x{} (got dimension {}x{})",
            dst_type_layout->getName(),
            element_type_layout->getRowCount(),
            element_type_layout->getColumnCount(),
            src_rows,
            src_cols
        );
        SGL_CHECK(
            dst_element_size == src_element_size
                && allow_scalar_conversion(
                    src_scalar_type,
                    (TypeReflection::ScalarType)element_type_layout->getScalarType()
                ),
            "\"{}\" expects a matrix with scalar type {} ({}B) (no implicit conversion from type {} ({}B))",
            dst_type_layout->getName(),
            (TypeReflection::ScalarType)element_type_layout->getScalarType(),
            dst_element_size,
            src_scalar_type,
            src_element_size
        );
        SGL_CHECK(
            src_size <= dst_type_layout->getSize(),
            "Mismatched size, writing {} B into backend type ({}) of only {} B.",
            src_size,
            dst_type_layout->getName(),
            dst_type_layout->getSize()
        );
    }


} // namespace cursor_utils


} // namespace sgl
