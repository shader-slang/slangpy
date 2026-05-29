// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/buffer_cursor.h"
#include "sgl/device/raytracing.h"
#include "sgl/device/resource.h"
#include "sgl/device/sampler.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/shader_object.h"
#include "sgl/func/tensor.h"
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
    static void write_to_cursor(const TCursor& cursor, const NestedTestStruct* value)
    {
        cursor["data"] = value->data;
    }
};

struct TestStruct {
    float f0 = 0;
    NestedTestStruct nested;

    template<typename TCursor>
    static void write_to_cursor(const TCursor& cursor, const TestStruct* value)
    {
        cursor["f0"] = value->f0;
        cursor["nested"] = value->nested;
    }
};

struct ShaderCursorOnlyStruct {
    static inline bool wrote = false;

    static void write_to_cursor(const ShaderCursor& cursor, const ShaderCursorOnlyStruct* value)
    {
        (void)cursor;
        (void)value;
        wrote = true;
    }
};

struct BufferCursorOnlyStruct {
    static inline bool wrote = false;

    static void write_to_cursor(const BufferElementCursor& cursor, const BufferCursorOnlyStruct* value)
    {
        (void)cursor;
        (void)value;
        wrote = true;
    }
};

struct BothCursorStruct {
    static void write_to_cursor(const ShaderCursor& cursor, const BothCursorStruct* value)
    {
        (void)cursor;
        (void)value;
    }
    static void write_to_cursor(const BufferElementCursor& cursor, const BothCursorStruct* value)
    {
        (void)cursor;
        (void)value;
    }
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
static_assert(HasWriteToCursor<BufferView, ShaderCursor>);
static_assert(!HasWriteToCursor<BufferView, BufferElementCursor>);
static_assert(HasWriteToCursor<TextureView, ShaderCursor>);
static_assert(!HasWriteToCursor<TextureView, BufferElementCursor>);
static_assert(HasWriteToCursor<Sampler, ShaderCursor>);
static_assert(!HasWriteToCursor<Sampler, BufferElementCursor>);
static_assert(HasWriteToCursor<AccelerationStructure, ShaderCursor>);
static_assert(!HasWriteToCursor<AccelerationStructure, BufferElementCursor>);
static_assert(HasWriteToCursor<DescriptorHandle, ShaderCursor>);
static_assert(HasWriteToCursor<DescriptorHandle, BufferElementCursor>);
static_assert(HasWriteToCursor<func::Tensor, ShaderCursor>);
static_assert(HasWriteToCursor<func::Tensor, BufferElementCursor>);
static_assert(!HasWriteToCursor<ref<Buffer>, ShaderCursor>);
static_assert(!HasWriteToCursor<ref<func::Tensor>, BufferElementCursor>);
static_assert(requires(const ShaderCursor& cursor, const ref<Buffer>& value) { cursor.set(value); });
static_assert(requires(const ShaderCursor& cursor, const ref<BufferView>& value) { cursor.set(value); });
static_assert(requires(const ShaderCursor& cursor, const ref<Texture>& value) { cursor.set(value); });
static_assert(requires(const ShaderCursor& cursor, const ref<TextureView>& value) { cursor.set(value); });
static_assert(requires(const ShaderCursor& cursor, const ref<Sampler>& value) { cursor.set(value); });
static_assert(requires(const ShaderCursor& cursor, const ref<AccelerationStructure>& value) { cursor.set(value); });
static_assert(requires(const ShaderCursor& cursor, const func::Tensor& value) { cursor.set(value); });
static_assert(requires(const BufferElementCursor& cursor, const ref<func::Tensor>& value) { cursor.set(value); });

} // namespace

namespace sgl::cursor_tests {

struct RegistryShaderOnlyStruct {
    static constexpr std::string_view slang_type_name = "RegistryShaderOnly";

    static void write_to_cursor(const ShaderCursor& cursor, const RegistryShaderOnlyStruct* value)
    {
        (void)cursor;
        (void)value;
    }
};

struct RegistryBufferOnlyStruct {
    static constexpr std::string_view slang_type_name = "RegistryBufferOnly";

    static void write_to_cursor(const BufferElementCursor& cursor, const RegistryBufferOnlyStruct* value)
    {
        (void)cursor;
        (void)value;
    }
};

struct RegistryDefaultSignatureStruct {
    static constexpr std::string_view slang_type_name = "RegistryDefault";
    static inline bool wrote_shader = false;
    static inline bool wrote_buffer = false;

    static void write_to_cursor(const ShaderCursor& cursor, const RegistryDefaultSignatureStruct* value)
    {
        (void)cursor;
        (void)value;
        wrote_shader = true;
    }

    static void write_to_cursor(const BufferElementCursor& cursor, const RegistryDefaultSignatureStruct* value)
    {
        (void)cursor;
        (void)value;
        wrote_buffer = true;
    }
};

struct RegistryMetadataStruct {
    static constexpr std::string_view slang_type_name = "RegistryMetadata";
    static inline bool wrote_shader = false;
    static inline bool wrote_buffer = false;
    static inline int imports_calls = 0;

    static void write_to_cursor(const ShaderCursor& cursor, const RegistryMetadataStruct* value)
    {
        (void)cursor;
        (void)value;
        wrote_shader = true;
    }

    static void write_to_cursor(const BufferElementCursor& cursor, const RegistryMetadataStruct* value)
    {
        (void)cursor;
        (void)value;
        wrote_buffer = true;
    }

    static void write_slangpy_signature(SignatureBuffer& sig, const RegistryMetadataStruct* value)
    {
        (void)value;
        sig.add("sig:RegistryMetadata");
    }

    static std::vector<std::string_view> slangpy_imports()
    {
        ++cursor_tests::RegistryMetadataStruct::imports_calls;
        return {"module/a.slang", "module/b.slang"};
    }
};

struct RegistryStaticStringSignatureStruct {
    static constexpr std::string_view slang_type_name = "RegistryStaticString";
    static constexpr std::string_view slangpy_signature = "sig:static-string";

    static void write_to_cursor(const ShaderCursor& cursor, const RegistryStaticStringSignatureStruct* value)
    {
        (void)cursor;
        (void)value;
    }
    static void write_to_cursor(const BufferElementCursor& cursor, const RegistryStaticStringSignatureStruct* value)
    {
        (void)cursor;
        (void)value;
    }
};

struct RegistryDynamicSignatureStruct {
    static constexpr std::string_view slang_type_name = "RegistryDynamic";

    uint32_t kind = 0;

    static void write_to_cursor(const ShaderCursor& cursor, const RegistryDynamicSignatureStruct* value)
    {
        (void)cursor;
        (void)value;
    }
    static void write_to_cursor(const BufferElementCursor& cursor, const RegistryDynamicSignatureStruct* value)
    {
        (void)cursor;
        (void)value;
    }

    static void write_slangpy_signature(SignatureBuffer& sig, const RegistryDynamicSignatureStruct* value)
    {
        sig.add("sig:dynamic:");
        sig.add(value->kind);
    }
};

struct RegistryMissingSlangTypeNameStruct {
    static inline bool wrote_shader = false;
    static inline bool wrote_buffer = false;

    static void write_to_cursor(const ShaderCursor& cursor, const RegistryMissingSlangTypeNameStruct* value)
    {
        (void)cursor;
        (void)value;
        wrote_shader = true;
    }

    static void write_to_cursor(const BufferElementCursor& cursor, const RegistryMissingSlangTypeNameStruct* value)
    {
        (void)cursor;
        (void)value;
        wrote_buffer = true;
    }
};

struct RegistryWriterOnlyStaticSignatureStruct {
    static constexpr std::string_view slangpy_signature = "sig:writer-only-static";

    static void write_to_cursor(const ShaderCursor& cursor, const RegistryWriterOnlyStaticSignatureStruct* value)
    {
        (void)cursor;
        (void)value;
    }
    static void write_to_cursor(const BufferElementCursor& cursor, const RegistryWriterOnlyStaticSignatureStruct* value)
    {
        (void)cursor;
        (void)value;
    }
};

struct RegistryDuplicateStruct {
    static constexpr std::string_view slang_type_name = "RegistryDuplicate";

    static void write_to_cursor(const ShaderCursor& cursor, const RegistryDuplicateStruct* value)
    {
        (void)cursor;
        (void)value;
    }
    static void write_to_cursor(const BufferElementCursor& cursor, const RegistryDuplicateStruct* value)
    {
        (void)cursor;
        (void)value;
    }
};

struct RegistryPartialMetadataStruct { };
struct RegistryImportsWithoutTypeStruct { };
struct RegistrySignatureOnlyStruct { };

static_assert(cursor_utils::CanRegisterCursorWriter<RegistryMissingSlangTypeNameStruct>);
static_assert(!cursor_utils::CanRegisterFunctionalCursorWriter<RegistryMissingSlangTypeNameStruct>);
static_assert(cursor_utils::CanRegisterCursorWriter<RegistryWriterOnlyStaticSignatureStruct>);
static_assert(!cursor_utils::CanRegisterFunctionalCursorWriter<RegistryWriterOnlyStaticSignatureStruct>);
static_assert(cursor_utils::CanRegisterCursorWriter<RegistryShaderOnlyStruct>);
static_assert(cursor_utils::CanRegisterCursorWriter<RegistryBufferOnlyStruct>);
static_assert(cursor_utils::CanRegisterCursorWriter<RegistryDefaultSignatureStruct>);
static_assert(cursor_utils::CanRegisterFunctionalCursorWriter<RegistryDefaultSignatureStruct>);
static_assert(cursor_utils::CanRegisterCursorWriter<Buffer>);
static_assert(cursor_utils::CanRegisterCursorWriter<Texture>);

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
    CHECK(info->write_signature);
    CHECK_FALSE(info->slang_type_name.empty());
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
    CHECK(info->write_signature);
    CHECK_FALSE(info->slang_type_name.empty());
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
    CHECK(info->write_signature);
    CHECK(info->slang_type_name.empty());
    CHECK(info->imports.empty());

    RegistryMissingSlangTypeNameStruct value;
    SignatureBuffer sig;
    info->write_signature(sig, &value);
    CHECK(std::string(sig.view()).find("RegistryMissingSlangTypeNameStruct") != std::string::npos);

    ShaderCursor shader_cursor{};
    BufferElementCursor buffer_cursor;
    RegistryMissingSlangTypeNameStruct::wrote_shader = false;
    RegistryMissingSlangTypeNameStruct::wrote_buffer = false;
    CHECK(info->write_shader_cursor(shader_cursor, &value));
    CHECK(info->write_buffer_cursor(buffer_cursor, &value));
    CHECK(RegistryMissingSlangTypeNameStruct::wrote_shader);
    CHECK(RegistryMissingSlangTypeNameStruct::wrote_buffer);
}

TEST_CASE("register_cursor_writer_without_functional_metadata_uses_explicit_signature")
{
    using namespace sgl::cursor_tests;

    cursor_utils::register_cursor_writer<RegistryWriterOnlyStaticSignatureStruct>();

    const cursor_utils::CursorWriterTypeInfo* info
        = cursor_utils::find_cursor_writer_type_info(typeid(RegistryWriterOnlyStaticSignatureStruct));
    REQUIRE(info);
    CHECK(info->write_shader_cursor);
    CHECK(info->write_buffer_cursor);
    CHECK(info->write_signature);
    CHECK(info->slang_type_name.empty());

    RegistryWriterOnlyStaticSignatureStruct value;
    SignatureBuffer sig;
    info->write_signature(sig, &value);
    CHECK_EQ(std::string(sig.view()), "sig:writer-only-static");
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
    imports_without_type.write_signature = [](SignatureBuffer& sig, const void* value)
    {
        (void)value;
        sig.add("sig:imports-without-type");
    };
    imports_without_type.imports.emplace_back("module/a.slang");
    CHECK_THROWS(cursor_utils::register_cursor_writer_type(std::move(imports_without_type)));
}

TEST_CASE("register_cursor_writer_type_allows_signature_without_functional_metadata")
{
    using namespace sgl::cursor_tests;

    cursor_utils::CursorWriterTypeInfo info;
    info.type = &typeid(RegistrySignatureOnlyStruct);
    info.write_shader_cursor = [](ShaderCursor& cursor, const void* value)
    {
        (void)cursor;
        (void)value;
        return true;
    };
    info.write_signature = [](SignatureBuffer& sig, const void* value)
    {
        (void)value;
        sig.add("sig:signature-only");
    };

    cursor_utils::register_cursor_writer_type(std::move(info));

    const cursor_utils::CursorWriterTypeInfo* registered
        = cursor_utils::find_cursor_writer_type_info(typeid(RegistrySignatureOnlyStruct));
    REQUIRE(registered);
    CHECK(registered->write_shader_cursor);
    CHECK_FALSE(bool(registered->write_buffer_cursor));
    CHECK(registered->write_signature);
    CHECK(registered->slang_type_name.empty());

    RegistrySignatureOnlyStruct value;
    SignatureBuffer sig;
    registered->write_signature(sig, &value);
    CHECK_EQ(std::string(sig.view()), "sig:signature-only");
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

TEST_CASE_GPU("shader_cursor_set_allows_null_resource_refs")
{
    ref<SlangModule> module = ctx.device->load_module_from_source(
        "shader_cursor_null_resource_refs",
        R"(
[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main(StructuredBuffer<uint> buffer)
{
}
)"
    );
    ref<ShaderProgram> program = ctx.device->link_program({module}, {module->entry_point("compute_main")});
    ref<ShaderObject> root_object = ctx.device->create_root_shader_object(program);
    ShaderCursor entry_point = ShaderCursor(root_object.get()).find_entry_point(0);

    ref<Buffer> buffer;
    CHECK_NOTHROW(entry_point["buffer"].set(buffer));
}

TEST_SUITE_END();
