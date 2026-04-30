// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/buffer_cursor.h"
#include "sgl/device/shader_cursor.h"
#include <fstream>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

using namespace sgl;

namespace {

struct NestedTestStruct {
    uint32_t data = 123;

    template<typename TCursor>
    void write_to_cursor(TCursor& cursor) const
    {
        cursor["data"] = data;
    }
};

struct TestStruct {
    float f0 = 0;
    NestedTestStruct nested;

    template<typename TCursor>
    void write_to_cursor(TCursor& cursor) const
    {
        cursor["f0"] = f0;
        cursor["nested"] = nested;
    }
};

struct ShaderCursorOnlyStruct {
    static inline bool wrote = false;

    void write_to_cursor(ShaderCursor& cursor) const
    {
        (void)cursor;
        wrote = true;
    }
};

struct BufferCursorOnlyStruct {
    static inline bool wrote = false;

    void write_to_cursor(BufferElementCursor& cursor) const
    {
        (void)cursor;
        wrote = true;
    }
};

struct BothCursorStruct {
    void write_to_cursor(ShaderCursor& cursor) const { (void)cursor; }
    void write_to_cursor(BufferElementCursor& cursor) const { (void)cursor; }
};

struct NoWriteToCursorStruct { };

static_assert(HasWriteToCursor<ShaderCursorOnlyStruct, ShaderCursor>);
static_assert(!HasWriteToCursor<ShaderCursorOnlyStruct, BufferElementCursor>);
static_assert(HasWriteToCursor<BufferCursorOnlyStruct, BufferElementCursor>);
static_assert(!HasWriteToCursor<BufferCursorOnlyStruct, ShaderCursor>);
static_assert(HasWriteToCursor<BothCursorStruct, ShaderCursor>);
static_assert(HasWriteToCursor<BothCursorStruct, BufferElementCursor>);
static_assert(!HasWriteToCursor<NoWriteToCursorStruct, ShaderCursor>);
static_assert(!HasWriteToCursor<NoWriteToCursorStruct, BufferElementCursor>);

} // namespace

namespace sgl::cursor_tests {

struct RegistryShaderOnlyStruct {
    static inline bool wrote_shader = false;

    void write_to_cursor(ShaderCursor& cursor) const
    {
        (void)cursor;
        wrote_shader = true;
    }
};

struct RegistryBufferOnlyStruct {
    static inline bool wrote_buffer = false;

    void write_to_cursor(BufferElementCursor& cursor) const
    {
        (void)cursor;
        wrote_buffer = true;
    }
};

struct RegistryMetadataStruct {
    static inline bool wrote_shader = false;
    static inline bool wrote_buffer = false;
    static inline int imports_calls = 0;

    void write_to_cursor(ShaderCursor& cursor) const
    {
        (void)cursor;
        wrote_shader = true;
    }

    void write_to_cursor(BufferElementCursor& cursor) const
    {
        (void)cursor;
        wrote_buffer = true;
    }
};

struct RegistryDuplicateStruct {
    void write_to_cursor(ShaderCursor& cursor) const { (void)cursor; }
};

struct RegistryLegacyMergeStruct { };

} // namespace sgl::cursor_tests

namespace sgl::cursor_utils {

template<>
struct CursorWriterTraits<cursor_tests::RegistryMetadataStruct> {
    static constexpr std::string_view slang_type_name = "RegistryMetadata";

    static void write_slangpy_signature(SignatureBuffer& sig) { sig.add("sig:RegistryMetadata"); }

    static std::vector<std::string_view> slangpy_imports()
    {
        ++cursor_tests::RegistryMetadataStruct::imports_calls;
        return {"module/a.slang", "module/b.slang"};
    }
};

} // namespace sgl::cursor_utils

TEST_SUITE_BEGIN("cursors");

TEST_CASE("shader_cursor_set_uses_shader_cursor_contract")
{
    ShaderCursorOnlyStruct::wrote = false;

    ShaderCursor cursor{};
    cursor.set(ShaderCursorOnlyStruct{});

    CHECK(ShaderCursorOnlyStruct::wrote);
}

TEST_CASE("buffer_element_cursor_set_uses_buffer_cursor_contract")
{
    BufferCursorOnlyStruct::wrote = false;

    BufferElementCursor cursor;
    cursor.set(BufferCursorOnlyStruct{});

    CHECK(BufferCursorOnlyStruct::wrote);
}

TEST_CASE("register_cursor_writer_shader_only")
{
    using namespace sgl::cursor_tests;

    cursor_utils::register_cursor_writer<RegistryShaderOnlyStruct>();

    const cursor_utils::CursorWriterTypeInfo* info
        = cursor_utils::find_cursor_writer_type_info(typeid(RegistryShaderOnlyStruct));
    REQUIRE(info);
    CHECK(info->write_shader_cursor);
    CHECK(!info->write_buffer_cursor);
    CHECK(!info->has_functional_metadata);

    ShaderCursor cursor{};
    RegistryShaderOnlyStruct value;
    RegistryShaderOnlyStruct::wrote_shader = false;
    CHECK(info->write_shader_cursor(cursor, &value));
    CHECK(RegistryShaderOnlyStruct::wrote_shader);
}

TEST_CASE("register_cursor_writer_buffer_only")
{
    using namespace sgl::cursor_tests;

    cursor_utils::register_cursor_writer<RegistryBufferOnlyStruct>();

    const cursor_utils::CursorWriterTypeInfo* info
        = cursor_utils::find_cursor_writer_type_info(typeid(RegistryBufferOnlyStruct));
    REQUIRE(info);
    CHECK(!info->write_shader_cursor);
    CHECK(info->write_buffer_cursor);
    CHECK(!info->has_functional_metadata);

    BufferElementCursor cursor;
    RegistryBufferOnlyStruct value;
    RegistryBufferOnlyStruct::wrote_buffer = false;
    CHECK(info->write_buffer_cursor(cursor, &value));
    CHECK(RegistryBufferOnlyStruct::wrote_buffer);
}

TEST_CASE("register_cursor_writer_static_metadata")
{
    using namespace sgl::cursor_tests;

    RegistryMetadataStruct::imports_calls = 0;
    cursor_utils::register_cursor_writer<RegistryMetadataStruct>();

    const cursor_utils::CursorWriterTypeInfo* info
        = cursor_utils::find_cursor_writer_type_info(typeid(RegistryMetadataStruct));
    REQUIRE(info);
    CHECK(info->write_shader_cursor);
    CHECK(info->write_buffer_cursor);
    CHECK(info->has_functional_metadata);
    CHECK_EQ(RegistryMetadataStruct::imports_calls, 1);

    RegistryMetadataStruct value;
    CHECK_EQ(info->slang_type_name(&value), "RegistryMetadata");

    SignatureBuffer sig;
    info->write_signature(sig, &value);
    CHECK_EQ(std::string(sig.view()), "sig:RegistryMetadata");

    REQUIRE_EQ(info->imports.size(), 2);
    CHECK_EQ(info->imports[0], "module/a.slang");
    CHECK_EQ(info->imports[1], "module/b.slang");

    ShaderCursor shader_cursor{};
    BufferElementCursor buffer_cursor;
    RegistryMetadataStruct::wrote_shader = false;
    RegistryMetadataStruct::wrote_buffer = false;
    CHECK(info->write_shader_cursor(shader_cursor, &value));
    CHECK(info->write_buffer_cursor(buffer_cursor, &value));
    CHECK(RegistryMetadataStruct::wrote_shader);
    CHECK(RegistryMetadataStruct::wrote_buffer);
}

TEST_CASE("register_cursor_writer_duplicate_rejected")
{
    using namespace sgl::cursor_tests;

    cursor_utils::register_cursor_writer<RegistryDuplicateStruct>();
    CHECK_THROWS(cursor_utils::register_cursor_writer<RegistryDuplicateStruct>());
}

TEST_CASE("legacy_cursor_writer_registration_merges_into_combined_registry")
{
    using namespace sgl::cursor_tests;

    cursor_utils::register_shader_cursor_object_writer<RegistryLegacyMergeStruct>(
        [](ShaderCursor& cursor, const RegistryLegacyMergeStruct& value)
        {
            (void)cursor;
            (void)value;
            return true;
        }
    );
    cursor_utils::register_buffer_element_cursor_object_writer<RegistryLegacyMergeStruct>(
        [](BufferElementCursor& cursor, const RegistryLegacyMergeStruct& value)
        {
            (void)cursor;
            (void)value;
            return true;
        }
    );

    const cursor_utils::CursorWriterTypeInfo* info
        = cursor_utils::find_cursor_writer_type_info(typeid(RegistryLegacyMergeStruct));
    REQUIRE(info);
    CHECK(info->write_shader_cursor);
    CHECK(info->write_buffer_cursor);
    CHECK_THROWS(
        cursor_utils::register_shader_cursor_object_writer<RegistryLegacyMergeStruct>(
            [](ShaderCursor& cursor, const RegistryLegacyMergeStruct& value)
            {
                (void)cursor;
                (void)value;
                return true;
            }
        )
    );
}

TEST_CASE_GPU("write_to_cursor")
{

    auto shader = R"SHADER(
struct NestedTestStruct
{
    uint32_t data;
};

struct TestStruct
{
    float f0;
    NestedTestStruct nested;
};
)SHADER";

    // Just verify module loads.
    SUBCASE("buffer_cursor")
    {
        ref<SlangModule> module = ctx.device->load_module_from_source("test", shader);
        CHECK(module);

        TestStruct cpu_struct;

        auto layout = module->layout();
        auto element_type = layout->find_type_by_name("TestStruct");
        auto element_type_layout = layout->get_type_layout(element_type);

        // This is how much we actually use in the memory for one element.
        size_t element_size = element_type_layout->stride();

        std::vector<uint8_t> from_direct(element_size, 0);
        auto direct_buffer_cursor = make_ref<sgl::BufferCursor>(
            ctx.device->type(),
            element_type_layout,
            from_direct.data(),
            from_direct.size()
        );

        (*direct_buffer_cursor)[0]["f0"] = cpu_struct.f0;
        (*direct_buffer_cursor)[0]["nested"]["data"] = cpu_struct.nested.data;

        std::vector<uint8_t> from_tocursor(element_size, 0);
        auto tocursor_cursor = make_ref<sgl::BufferCursor>(
            ctx.device->type(),
            element_type_layout,
            from_tocursor.data(),
            from_tocursor.size()
        );

        (*tocursor_cursor)[0] = cpu_struct;

        CHECK(memcmp(from_direct.data(), from_tocursor.data(), from_direct.size()) == 0);
    }
}

TEST_SUITE_END();
