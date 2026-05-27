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
#include <typeinfo>
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
    static constexpr std::string_view slang_type_name = "RegistryShaderOnly";

    void write_to_cursor(ShaderCursor& cursor) const { (void)cursor; }
};

struct RegistryBufferOnlyStruct {
    static constexpr std::string_view slang_type_name = "RegistryBufferOnly";

    void write_to_cursor(BufferElementCursor& cursor) const { (void)cursor; }
};

struct RegistryDefaultSignatureStruct {
    static constexpr std::string_view slang_type_name = "RegistryDefault";
    static inline bool wrote_shader = false;
    static inline bool wrote_buffer = false;

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

struct RegistryMetadataStruct {
    static constexpr std::string_view slang_type_name = "RegistryMetadata";
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

    static void write_slangpy_signature(SignatureBuffer& sig) { sig.add("sig:RegistryMetadata"); }

    static std::vector<std::string_view> slangpy_imports()
    {
        ++cursor_tests::RegistryMetadataStruct::imports_calls;
        return {"module/a.slang", "module/b.slang"};
    }
};

struct RegistryStaticStringSignatureStruct {
    static constexpr std::string_view slang_type_name = "RegistryStaticString";
    static constexpr std::string_view slangpy_signature = "sig:static-string";

    void write_to_cursor(ShaderCursor& cursor) const { (void)cursor; }
    void write_to_cursor(BufferElementCursor& cursor) const { (void)cursor; }
};

struct RegistryDynamicSignatureStruct {
    static constexpr std::string_view slang_type_name = "RegistryDynamic";

    uint32_t kind = 0;

    void write_to_cursor(ShaderCursor& cursor) const { (void)cursor; }
    void write_to_cursor(BufferElementCursor& cursor) const { (void)cursor; }

    void write_slangpy_signature(SignatureBuffer& sig) const
    {
        sig.add("sig:dynamic:");
        sig.add(kind);
    }
};

struct RegistryMissingSlangTypeNameStruct {
    static inline bool wrote_shader = false;
    static inline bool wrote_buffer = false;

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
    static constexpr std::string_view slang_type_name = "RegistryDuplicate";

    void write_to_cursor(ShaderCursor& cursor) const { (void)cursor; }
    void write_to_cursor(BufferElementCursor& cursor) const { (void)cursor; }
};

struct RegistryPartialMetadataStruct { };
struct RegistryImportsWithoutTypeStruct { };

static_assert(cursor_utils::CanRegisterCursorWriter<RegistryMissingSlangTypeNameStruct>);
static_assert(!cursor_utils::CanRegisterFunctionalCursorWriter<RegistryMissingSlangTypeNameStruct>);
static_assert(!cursor_utils::CanRegisterCursorWriter<RegistryShaderOnlyStruct>);
static_assert(!cursor_utils::CanRegisterCursorWriter<RegistryBufferOnlyStruct>);
static_assert(cursor_utils::CanRegisterCursorWriter<RegistryDefaultSignatureStruct>);
static_assert(cursor_utils::CanRegisterFunctionalCursorWriter<RegistryDefaultSignatureStruct>);

} // namespace sgl::cursor_tests

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

TEST_CASE("register_cursor_writer_default_signature_writes_both_cursors")
{
    using namespace sgl::cursor_tests;

    cursor_utils::register_cursor_writer<RegistryDefaultSignatureStruct>();

    const cursor_utils::CursorWriterTypeInfo* info
        = cursor_utils::find_cursor_writer_type_info(typeid(RegistryDefaultSignatureStruct));
    REQUIRE(info);
    CHECK(info->write_shader_cursor);
    CHECK(info->write_buffer_cursor);
    CHECK(info->has_functional_metadata());
    CHECK_EQ(info->slang_type_name, "RegistryDefault");

    RegistryDefaultSignatureStruct value;
    SignatureBuffer sig;
    info->write_signature(sig, &value);
    CHECK(std::string(sig.view()).find("RegistryDefaultSignatureStruct") != std::string::npos);

    ShaderCursor shader_cursor{};
    BufferElementCursor buffer_cursor;
    RegistryDefaultSignatureStruct::wrote_shader = false;
    RegistryDefaultSignatureStruct::wrote_buffer = false;
    CHECK(info->write_shader_cursor(shader_cursor, &value));
    CHECK(info->write_buffer_cursor(buffer_cursor, &value));
    CHECK(RegistryDefaultSignatureStruct::wrote_shader);
    CHECK(RegistryDefaultSignatureStruct::wrote_buffer);
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
    CHECK(info->has_functional_metadata());
    CHECK_EQ(RegistryMetadataStruct::imports_calls, 1);

    RegistryMetadataStruct value;
    CHECK_EQ(info->slang_type_name, "RegistryMetadata");

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

TEST_CASE("register_cursor_writer_without_functional_metadata")
{
    using namespace sgl::cursor_tests;

    cursor_utils::register_cursor_writer<RegistryMissingSlangTypeNameStruct>();

    const cursor_utils::CursorWriterTypeInfo* info
        = cursor_utils::find_cursor_writer_type_info(typeid(RegistryMissingSlangTypeNameStruct));
    REQUIRE(info);
    CHECK(info->write_shader_cursor);
    CHECK(info->write_buffer_cursor);
    CHECK_FALSE(info->has_functional_metadata());
    CHECK(info->slang_type_name.empty());
    CHECK_FALSE(bool(info->write_signature));
    CHECK(info->imports.empty());

    ShaderCursor shader_cursor{};
    BufferElementCursor buffer_cursor;
    RegistryMissingSlangTypeNameStruct value;
    RegistryMissingSlangTypeNameStruct::wrote_shader = false;
    RegistryMissingSlangTypeNameStruct::wrote_buffer = false;
    CHECK(info->write_shader_cursor(shader_cursor, &value));
    CHECK(info->write_buffer_cursor(buffer_cursor, &value));
    CHECK(RegistryMissingSlangTypeNameStruct::wrote_shader);
    CHECK(RegistryMissingSlangTypeNameStruct::wrote_buffer);
}

TEST_CASE("register_cursor_writer_type_rejects_partial_functional_metadata")
{
    using namespace sgl::cursor_tests;

    auto make_info = [](const std::type_info& type)
    {
        cursor_utils::CursorWriterTypeInfo info;
        info.type = &type;
        info.write_shader_cursor = [](ShaderCursor& cursor, const void* value)
        {
            (void)cursor;
            (void)value;
            return true;
        };
        info.write_buffer_cursor = [](BufferElementCursor& cursor, const void* value)
        {
            (void)cursor;
            (void)value;
            return true;
        };
        return info;
    };

    auto missing_signature = make_info(typeid(RegistryPartialMetadataStruct));
    missing_signature.slang_type_name = "RegistryPartial";
    CHECK_THROWS(cursor_utils::register_cursor_writer_type(std::move(missing_signature)));

    auto imports_without_type = make_info(typeid(RegistryImportsWithoutTypeStruct));
    imports_without_type.imports.emplace_back("module/a.slang");
    CHECK_THROWS(cursor_utils::register_cursor_writer_type(std::move(imports_without_type)));
}

TEST_CASE("register_cursor_writer_static_string_signature")
{
    using namespace sgl::cursor_tests;

    cursor_utils::register_cursor_writer<RegistryStaticStringSignatureStruct>();

    const cursor_utils::CursorWriterTypeInfo* info
        = cursor_utils::find_cursor_writer_type_info(typeid(RegistryStaticStringSignatureStruct));
    REQUIRE(info);
    CHECK_EQ(info->slang_type_name, "RegistryStaticString");

    RegistryStaticStringSignatureStruct value;
    SignatureBuffer sig;
    info->write_signature(sig, &value);
    CHECK_EQ(std::string(sig.view()), "sig:static-string");
}

TEST_CASE("register_cursor_writer_dynamic_signature")
{
    using namespace sgl::cursor_tests;

    cursor_utils::register_cursor_writer<RegistryDynamicSignatureStruct>();

    const cursor_utils::CursorWriterTypeInfo* info
        = cursor_utils::find_cursor_writer_type_info(typeid(RegistryDynamicSignatureStruct));
    REQUIRE(info);
    CHECK_EQ(info->slang_type_name, "RegistryDynamic");

    RegistryDynamicSignatureStruct value;
    value.kind = 7;
    SignatureBuffer sig_a;
    info->write_signature(sig_a, &value);
    CHECK_EQ(std::string(sig_a.view()), "sig:dynamic:00000007");

    value.kind = 9;
    SignatureBuffer sig_b;
    info->write_signature(sig_b, &value);
    CHECK_EQ(std::string(sig_b.view()), "sig:dynamic:00000009");
}

TEST_CASE("register_cursor_writer_duplicate_rejected")
{
    using namespace sgl::cursor_tests;

    cursor_utils::register_cursor_writer<RegistryDuplicateStruct>();
    CHECK_THROWS(cursor_utils::register_cursor_writer<RegistryDuplicateStruct>());
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

struct Pair
{
    uint16_t x;
    uint16_t y;
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

    // Reinterpret cursor to a different type and write to it.
    SUBCASE("buffer_cursor_reinterpret")
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

        // reinterpret uint32_t data as uint16_t[2]
        uint16_t pair[] = {55, 67};
        auto reinterpret_element_type = layout->find_type_by_name("Pair");
        auto reinterpret_element_type_layout = layout->get_type_layout(reinterpret_element_type);
        auto reinterpret_cursor
            = (*direct_buffer_cursor)[0]["nested"]["data"].reinterpret(reinterpret_element_type_layout);
        reinterpret_cursor["x"] = pair[0];
        reinterpret_cursor["y"] = pair[1];

        std::vector<uint8_t> from_tocursor(element_size, 0);
        auto tocursor_cursor = make_ref<sgl::BufferCursor>(
            ctx.device->type(),
            element_type_layout,
            from_tocursor.data(),
            from_tocursor.size()
        );

        reinterpret_cast<uint16_t*>(&cpu_struct.nested.data)[0] = pair[0];
        reinterpret_cast<uint16_t*>(&cpu_struct.nested.data)[1] = pair[1];
        (*tocursor_cursor)[0] = cpu_struct;

        CHECK(memcmp(from_direct.data(), from_tocursor.data(), from_direct.size()) == 0);
    }
}

TEST_SUITE_END();
