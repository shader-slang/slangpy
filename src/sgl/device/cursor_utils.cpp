// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/device/cursor_utils.h"

namespace sgl {

namespace cursor_utils {

    namespace {

        std::vector<CursorWriterTypeInfo>& cursor_writer_type_info_registry()
        {
            static std::vector<CursorWriterTypeInfo> infos;
            return infos;
        }

    } // namespace

    void register_cursor_writer_type(CursorWriterTypeInfo info)
    {
        SGL_CHECK(info.type != nullptr, "Cursor writer type info must specify a type.");
        SGL_CHECK(
            bool(info.write_shader_cursor) || bool(info.write_buffer_cursor),
            "Cursor writer type info for type \"{}\" must provide at least one cursor writer.",
            info.type->name()
        );
        if (info.has_functional_metadata) {
            SGL_CHECK(
                bool(info.slang_type_name),
                "Cursor writer type info for type \"{}\" has functional metadata but no Slang type name.",
                info.type->name()
            );
            SGL_CHECK(
                bool(info.write_signature),
                "Cursor writer type info for type \"{}\" has functional metadata but no signature writer.",
                info.type->name()
            );
        }

        auto& infos = cursor_writer_type_info_registry();
        for (auto& entry : infos) {
            if (!(*entry.type == *info.type))
                continue;

            if (info.write_shader_cursor) {
                if (entry.write_shader_cursor) {
                    SGL_THROW("ShaderCursor object writer for type \"{}\" is already registered.", info.type->name());
                }
                entry.write_shader_cursor = std::move(info.write_shader_cursor);
            }

            if (info.write_buffer_cursor) {
                if (entry.write_buffer_cursor) {
                    SGL_THROW(
                        "BufferElementCursor object writer for type \"{}\" is already registered.",
                        info.type->name()
                    );
                }
                entry.write_buffer_cursor = std::move(info.write_buffer_cursor);
            }

            if (info.has_functional_metadata) {
                if (entry.has_functional_metadata) {
                    SGL_THROW("Cursor writer metadata for type \"{}\" is already registered.", info.type->name());
                }
                entry.has_functional_metadata = true;
                entry.slang_type_name = std::move(info.slang_type_name);
                entry.write_signature = std::move(info.write_signature);
                entry.imports = std::move(info.imports);
            }

            return;
        }

        infos.push_back(std::move(info));
    }

    std::span<const CursorWriterTypeInfo> cursor_writer_type_infos()
    {
        return cursor_writer_type_info_registry();
    }

    const CursorWriterTypeInfo* find_cursor_writer_type_info(const std::type_info& type)
    {
        for (const auto& info : cursor_writer_type_info_registry()) {
            if (*info.type == type) {
                return &info;
            }
        }
        return nullptr;
    }

    void register_shader_cursor_object_writer(const std::type_info& type, ShaderCursorObjectWriteFunc write)
    {
        SGL_CHECK(bool(write), "ShaderCursor object writer must be callable.");

        CursorWriterTypeInfo info;
        info.type = &type;
        info.write_shader_cursor = std::move(write);
        register_cursor_writer_type(std::move(info));
    }

    void
    register_buffer_element_cursor_object_writer(const std::type_info& type, BufferElementCursorObjectWriteFunc write)
    {
        SGL_CHECK(bool(write), "BufferElementCursor object writer must be callable.");

        CursorWriterTypeInfo info;
        info.type = &type;
        info.write_buffer_cursor = std::move(write);
        register_cursor_writer_type(std::move(info));
    }

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
