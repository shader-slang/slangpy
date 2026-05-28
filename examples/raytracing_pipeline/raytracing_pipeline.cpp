// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/sgl.h"
#include "sgl/core/platform.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/pipeline.h"
#include "sgl/device/command.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/shader_object.h"
#include "sgl/device/kernel.h"
#include "sgl/device/agility_sdk.h"
#include "sgl/utils/tev.h"

SGL_EXPORT_AGILITY_SDK

static const std::filesystem::path EXAMPLE_DIR(SGL_EXAMPLE_DIR);

using namespace sgl;

int main()
{
    sgl::static_init();

    {
        ref<Device> device = Device::create({
            .enable_debug_layers = true,
            .compiler_options = {.include_paths = {EXAMPLE_DIR}},
        });

        std::vector<float3> vertices{{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}};
        std::vector<uint32_t> indices{0, 1, 2};

        ref<Buffer> vertex_buffer = device->create_buffer({
            .usage = BufferUsage::acceleration_structure_build_input,
            .label = "vertex_buffer",
            .data = vertices.data(),
            .data_size = vertices.size() * sizeof(float3),
        });

        ref<Buffer> index_buffer = device->create_buffer({
            .usage = BufferUsage::acceleration_structure_build_input,
            .label = "index_buffer",
            .data = indices.data(),
            .data_size = indices.size() * sizeof(uint32_t),
        });

        AccelerationStructureBuildInputTriangles blas_input_triangles{
            .vertex_buffers = {vertex_buffer},
            .vertex_format = Format::rgb32_float,
            .vertex_count = narrow_cast<uint32_t>(vertices.size()),
            .vertex_stride = sizeof(float3),
            .index_buffer = index_buffer,
            .index_format = IndexFormat::uint32,
            .index_count = narrow_cast<uint32_t>(indices.size()),
            .flags = AccelerationStructureGeometryFlags::opaque,
        };

        AccelerationStructureBuildDesc blas_build_desc{
            .inputs = {blas_input_triangles},
        };

        AccelerationStructureSizes blas_sizes = device->get_acceleration_structure_sizes(blas_build_desc);

        ref<Buffer> blas_scratch_buffer = device->create_buffer({
            .size = blas_sizes.scratch_size,
            .usage = BufferUsage::unordered_access,
            .label = "blas_scratch_buffer",
        });

        ref<AccelerationStructure> blas = device->create_acceleration_structure({
            .kind = AccelerationStructureKind::bottom_level,
            .size = blas_sizes.acceleration_structure_size,
            .label = "blas",
        });

        // Build bottom level acceleration structure.
        {
            ref<CommandEncoder> command_encoder = device->create_command_encoder();
            command_encoder->build_acceleration_structure(blas_build_desc, blas, nullptr, blas_scratch_buffer);
            device->submit_command_buffer(command_encoder->finish());
        }

        ref<AccelerationStructureInstanceList> instance_list = device->create_acceleration_structure_instance_list(1);
        instance_list->write(
            0,
            AccelerationStructureInstanceDesc{
                .transform = float3x4::identity(),
                .instance_id = 0,
                .instance_mask = 0xff,
                .instance_contribution_to_hit_group_index = 0,
                .flags = AccelerationStructureInstanceFlags::none,
                .acceleration_structure = blas->handle(),
            }
        );

        AccelerationStructureBuildDesc tlas_build_desc{
            .inputs = {instance_list->build_input_instances()},
        };

        AccelerationStructureSizes tlas_sizes = device->get_acceleration_structure_sizes(tlas_build_desc);

        ref<Buffer> tlas_scratch_buffer = device->create_buffer({
            .size = tlas_sizes.scratch_size,
            .usage = BufferUsage::unordered_access,
            .label = "tlas_scratch_buffer",
        });

        ref<AccelerationStructure> tlas = device->create_acceleration_structure({
            .kind = AccelerationStructureKind::top_level,
            .size = tlas_sizes.acceleration_structure_size,
            .label = "tlas",
        });

        // Build top level acceleration structure.
        {
            ref<CommandEncoder> command_encoder = device->create_command_encoder();
            command_encoder->build_acceleration_structure(tlas_build_desc, tlas, nullptr, tlas_scratch_buffer);
            device->submit_command_buffer(command_encoder->finish());
        }

        ref<Texture> render_texture = device->create_texture({
            .format = Format::rgba32_float,
            .width = 1024,
            .height = 1024,
            .usage = TextureUsage::unordered_access,
            .label = "render_texture",
        });

        ref<ShaderProgram> program
            = device->load_program("raytracing_pipeline.slang", {"ray_gen", "miss", "closest_hit"});
        ref<RayTracingPipeline> pipeline = device->create_ray_tracing_pipeline({
            .program = program,
            .hit_groups = {{
                .hit_group_name = "hit_group",
                .closest_hit_entry_point = "closest_hit",
            }},
            .max_recursion = 1,
            .max_ray_payload_size = 12,
        });

        ref<ShaderTable> shader_table = device->create_shader_table({
            .program = program,
            .ray_gen_entry_points = {"ray_gen"},
            .miss_entry_points = {"miss"},
            .hit_group_names = {"hit_group"},
        });

        {
            ref<CommandEncoder> command_encoder = device->create_command_encoder();
            {
                auto pass_encoder = command_encoder->begin_ray_tracing_pass();
                auto shader_object = pass_encoder->bind_pipeline(pipeline, shader_table);
                auto cursor = ShaderCursor(shader_object);
                cursor["tlas"] = tlas;
                cursor["render_texture"] = render_texture;
                pass_encoder->dispatch_rays(0, uint3{1024, 1024, 1});
                pass_encoder->end();
            }
            device->submit_command_buffer(command_encoder->finish());
        }

        tev::show(render_texture, "raytracing_pipeline");

        device->close();
    }

    sgl::static_shutdown();
    return 0;
}
