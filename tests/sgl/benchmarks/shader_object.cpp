// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/sgl.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/shader_object.h"
#include "sgl/device/agility_sdk.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

SGL_EXPORT_AGILITY_SDK

using namespace sgl;

namespace {

constexpr char shader[] = R"(
struct ModuleParams { uint value; }
struct ProgramParams { uint value; }
struct ObjectParams { uint value; }
struct ColdParams { uint value; }
ParameterBlock<ProgramParams> params;
RWStructuredBuffer<uint> output;
[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main() { output[0] = params.value; }
)";

template<typename F>
void measure(std::string_view name, std::string_view filter, int iterations, F operation)
{
    if (filter != "all" && filter != name)
        return;
    for (int i = 0; i < 256; ++i)
        operation(i);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
        operation(i);
    auto elapsed = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start);
    std::cout << name << ',' << iterations << ',' << elapsed.count() / iterations << '\n';
}

template<typename F>
void measure_child_phases(
    const char* name,
    Device* device,
    ShaderProgram* program,
    const std::vector<ref<const ShaderObject>>& children,
    int iterations,
    F operation
)
{
    using Clock = std::chrono::steady_clock;
    Clock::duration work{}, teardown{};
    int batches = std::max(1, iterations / int(children.size()));
    for (int i = -32; i < batches; ++i) {
        auto parent = device->create_root_shader_object(program);
        auto start = Clock::now();
        for (const auto& child : children)
            operation(parent.get(), child);
        auto end = Clock::now();
        parent.reset();
        auto released = Clock::now();
        if (i >= 0) {
            work += end - start;
            teardown += released - end;
        }
    }
    auto count = batches * children.size();
    using Ns = std::chrono::duration<double, std::nano>;
    std::cout << name << "-work," << count << ',' << Ns(work).count() / count << '\n';
    std::cout << name << "-release," << count << ',' << Ns(teardown).count() / count << '\n';
}

void benchmark(DeviceType type, bool validation, std::string_view filter, int iterations)
{
    auto device = Device::create({.type = type, .enable_rhi_validation = validation});
    auto module = device->load_module_from_source("shader_object_benchmark", shader);
    auto program = device->link_program({module}, {module->entry_point("compute_main")});
    auto module_layout = module->layout()->get_type_layout(module->layout()->find_type_by_name("ModuleParams"));
    auto program_layout = program->layout()->get_type_layout(program->layout()->find_type_by_name("ProgramParams"));
    auto source = device->create_shader_object(
        module->layout()->get_type_layout(module->layout()->find_type_by_name("ObjectParams"))
    );
    auto object_layout = source->element_type_layout();
    auto reflection = program->layout();
    auto cold_type = reflection->find_type_by_name("ColdParams");
    SGL_CHECK(cold_type, "Missing cold type");
    SGL_CHECK(
        module_layout->owner() == module.get() && program_layout->owner() == program.get()
            && object_layout->owner() == source.get(),
        "Unexpected benchmark reflection owner"
    );
    auto root = device->create_root_shader_object(program);
    auto offset = ShaderCursor(root)["params"].offset();
    if (filter == "phases") {
        std::vector<ref<const ShaderObject>> children;
        for (int i = 0; i < 1024; ++i)
            children.push_back(device->create_shader_object(program_layout));
        const rhi::ShaderOffset native_offset{
            offset.uniform_offset,
            offset.binding_range_index,
            offset.binding_array_index
        };
        measure_child_phases(
            "sgl-bind",
            device,
            program,
            children,
            iterations,
            [&](ShaderObject* parent, const auto& child)
            {
                parent->set_object(offset, child);
            }
        );
        measure_child_phases(
            "sgl-bind-get",
            device,
            program,
            children,
            iterations,
            [&](ShaderObject* parent, const auto& child)
            {
                parent->set_object(offset, child);
                parent->get_object(offset);
            }
        );
        measure_child_phases(
            "rhi-bind-get",
            device,
            program,
            children,
            iterations,
            [&](ShaderObject* parent, const auto& child)
            {
                auto* native = parent->rhi_shader_object();
                SGL_CHECK(
                    SLANG_SUCCEEDED(native->setObject(native_offset, child->rhi_shader_object())),
                    "RHI bind failed"
                );
                auto object = native->getObject(native_offset);
                SGL_CHECK(object, "RHI read failed");
            }
        );
        return;
    }
    ref<const ShaderObject> values[] = {
        device->create_shader_object(program_layout),
        device->create_shader_object(program_layout),
    };

    // Keep the reflection, shader compilation, and GPU submission out of the timed region.
    // Each creation includes destruction of the returned object.
    measure(
        "reflect-and-create",
        filter,
        iterations,
        [&](int)
        {
            device->create_shader_object(reflection->get_type_layout(cold_type));
        }
    );
    measure(
        "create-module",
        filter,
        iterations,
        [&](int)
        {
            device->create_shader_object(module_layout);
        }
    );
    measure(
        "create-program",
        filter,
        iterations,
        [&](int)
        {
            device->create_shader_object(program_layout);
        }
    );
    measure(
        "create-object",
        filter,
        iterations,
        [&](int)
        {
            device->create_shader_object(object_layout);
        }
    );
    auto* rhi_device = device->rhi_device();
    auto* component = program->rhi_shader_program()->getSlangComponentType();
    auto* layout = program_layout->get_slang_type_layout();
    measure(
        "rhi-create",
        filter,
        iterations,
        [&](int)
        {
            Slang::ComPtr<rhi::IShaderObject> object;
            SGL_CHECK(
                SLANG_SUCCEEDED(rhi_device->createShaderObjectFromTypeLayout(component, layout, object.writeRef())),
                "RHI shader object creation failed"
            );
        }
    );
    measure(
        "root",
        filter,
        iterations,
        [&](int)
        {
            device->create_root_shader_object(program);
        }
    );
    measure(
        "root-first-get",
        filter,
        iterations,
        [&](int)
        {
            auto parent = device->create_root_shader_object(program);
            parent->get_object(offset);
        }
    );
    measure(
        "root-first-bind",
        filter,
        iterations,
        [&](int)
        {
            auto parent = device->create_root_shader_object(program);
            parent->set_object(offset, values[0]);
        }
    );
    measure(
        "rebind",
        filter,
        iterations,
        [&](int i)
        {
            root->set_object(offset, values[i % 2]);
        }
    );
    // Bound the lifetime of wrappers retained for shader cursors on both revisions.
    measure(
        "root-32-gets",
        filter,
        iterations,
        [&](int)
        {
            auto parent = device->create_root_shader_object(program);
            for (int i = 0; i < 32; ++i)
                parent->get_object(offset);
        }
    );

    // Rebind one slot to successive distinct children and read it after each write.
    // A fresh parent bounds wrapper lifetimes in both implementations.
    for (int width : {16, 128, 1024}) {
        auto name = "distinct-" + std::to_string(width);
        if (filter != "all" && filter != name)
            continue;
        std::vector<ref<const ShaderObject>> children;
        for (int i = 0; i < width; ++i)
            children.push_back(device->create_shader_object(program_layout));
        measure(
            name,
            filter,
            std::max(iterations / width, 1),
            [&](int)
            {
                auto parent = device->create_root_shader_object(program);
                for (const auto& child : children) {
                    parent->set_object(offset, child);
                    parent->get_object(offset);
                }
            }
        );
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 5) {
        std::cerr
            << "Usage: sgl_shader_object_benchmark <d3d12|vulkan> <validation|no-validation> <case|all> <iterations>\n";
        return 1;
    }
    try {
        auto type = string_to_enum<DeviceType>(argv[1]);
        std::string_view validation = argv[2];
        int iterations = std::stoi(argv[4]);
        SGL_CHECK(validation == "validation" || validation == "no-validation", "Invalid validation mode");
        SGL_CHECK(iterations > 0, "Iterations must be positive");
        sgl::static_init();
        benchmark(type, validation == "validation", argv[3], iterations);
        sgl::static_shutdown();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
