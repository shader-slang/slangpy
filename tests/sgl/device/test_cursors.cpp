// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/buffer_cursor.h"
#include <fstream>
#include <filesystem>

using namespace sgl;

namespace {

struct NestedTestStruct {
    uint32_t data = 123;

    template<typename TCursor>
    void to_cursor(TCursor& cursor) const
    {
        cursor["data"] = data;
    }
};

struct TestStruct {
    float f0 = 0;
    NestedTestStruct nested;

    template<typename TCursor>
    void to_cursor(TCursor& cursor) const
    {
        cursor["f0"] = f0;
        cursor["nested"] = nested;
    }
};

} // namespace

TEST_SUITE_BEGIN("cursors");

TEST_CASE_GPU("to_cursor")
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

        size_t element_size = element_type_layout->size();

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
