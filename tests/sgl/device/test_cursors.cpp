// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/buffer_cursor.h"
#include "sgl/device/shader_cursor.h"
#include <fstream>
#include <filesystem>

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
