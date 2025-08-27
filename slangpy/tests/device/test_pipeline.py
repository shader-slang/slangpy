# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np
import sys

import slangpy as spy
from slangpy.testing import helpers

from typing import Any, Optional, Sequence


class PipelineTestContext:
    def __init__(self, device_type: spy.DeviceType, size: int = 128) -> None:
        super().__init__()
        self.device = helpers.get_device(type=device_type)
        self.output_texture = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=size,
            height=size,
            usage=spy.TextureUsage.unordered_access
            | spy.TextureUsage.shader_resource
            | spy.TextureUsage.render_target,
            label="render_texture",
        )
        self.count_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            size=16,
            label="count_buffer",
            data=np.array([0, 0, 0, 0], dtype=np.uint32),
        )

        self.clear_kernel = self.device.create_compute_kernel(
            self.device.load_program("test_pipeline_utils.slang", ["clear"])
        )
        self.count_kernel = self.device.create_compute_kernel(
            self.device.load_program("test_pipeline_utils.slang", ["count"])
        )

        self.clear()

    def clear(self):
        self.clear_kernel.dispatch(
            thread_count=[self.output_texture.width, self.output_texture.height, 1],
            render_texture=self.output_texture,
        )

    def count(self):
        self.count_buffer.copy_from_numpy(np.array([0, 0, 0, 0], dtype=np.uint32))
        self.count_kernel.dispatch(
            thread_count=[self.output_texture.width, self.output_texture.height, 1],
            render_texture=self.output_texture,
            count_buffer=self.count_buffer,
        )

    def expect_counts(self, expected: Sequence[int]):
        self.count()
        count = self.count_buffer.to_numpy().view(np.uint32)
        assert np.all(count == expected)

    def create_quad_mesh(self):
        vertices = np.array([-1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1], dtype=np.float32)
        indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)

        vertex_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.vertex_buffer,
            label="vertex_buffer",
            data=vertices,
        )
        input_layout = self.device.create_input_layout(
            input_elements=[
                {
                    "semantic_name": "POSITION",
                    "semantic_index": 0,
                    "format": spy.Format.rgb32_float,
                    "offset": 0,
                },
            ],
            vertex_streams=[{"stride": 12}],
        )
        index_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.index_buffer,
            label="index_buffer",
            data=indices,
        )

        return vertex_buffer, index_buffer, input_layout


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_clear_and_count(device_type: spy.DeviceType):
    ctx = PipelineTestContext(device_type)
    ctx.expect_counts([0, 0, 0, 0])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_compute_set_square(device_type: spy.DeviceType):
    ctx = PipelineTestContext(device_type)
    prog = ctx.device.load_program("test_pipeline_utils.slang", ["setcolor"])
    set_kernel = ctx.device.create_compute_kernel(prog)

    pos = spy.int2(32, 32)
    size = spy.int2(16, 16)
    set_kernel.dispatch(
        thread_count=[ctx.output_texture.width, ctx.output_texture.height, 1],
        render_texture=ctx.output_texture,
        pos=pos,
        size=size,
        color=spy.float4(1, 0, 0, 1),
    )

    area = size.x * size.y
    ctx.expect_counts([area, 0, 0, area])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_compute_set_and_overwrite(device_type: spy.DeviceType):
    ctx = PipelineTestContext(device_type)
    prog = ctx.device.load_program("test_pipeline_utils.slang", ["setcolor"])
    set_kernel = ctx.device.create_compute_kernel(prog)

    pos1 = spy.int2(0, 0)
    size1 = spy.int2(128, 128)
    set_kernel.dispatch(
        thread_count=[ctx.output_texture.width, ctx.output_texture.height, 1],
        render_texture=ctx.output_texture,
        pos=pos1,
        size=size1,
        color=spy.float4(1, 0, 0, 0),
    )

    pos2 = spy.int2(32, 32)
    size2 = spy.int2(16, 16)
    set_kernel.dispatch(
        thread_count=[ctx.output_texture.width, ctx.output_texture.height, 1],
        render_texture=ctx.output_texture,
        pos=pos2,
        size=size2,
        color=spy.float4(0, 1, 0, 0),
    )

    area1 = size1.x * size1.y
    area2 = size2.x * size2.y
    ctx.expect_counts([area1 - area2, area2, 0, 0])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gfx_clear(device_type: spy.DeviceType):
    ctx = PipelineTestContext(device_type)

    command_encoder = ctx.device.create_command_encoder()
    command_encoder.clear_texture_float(ctx.output_texture, clear_value=[1.0, 0.0, 1.0, 0.0])
    ctx.device.submit_command_buffer(command_encoder.finish())

    area = ctx.output_texture.width * ctx.output_texture.height

    ctx.expect_counts([area, 0, area, 0])


class GfxContext:
    def __init__(self, ctx: PipelineTestContext) -> None:
        super().__init__()
        if not ctx.device.has_feature(spy.Feature.rasterization):
            pytest.skip("Rasterization not supported on this device")

        self.ctx = ctx
        self.program = ctx.device.load_program(
            "test_pipeline_raster.slang", ["vertex_main", "fragment_main"]
        )
        self.vertex_buffer, self.index_buffer, self.input_layout = ctx.create_quad_mesh()

    # Draw a quad with the given pipeline and color, optionally clearing to black first.
    # The quad is [-1,-1]->[1,1] so if offset/scale aren't specified will fill the whole screen.
    def draw(
        self,
        pipeline: spy.RenderPipeline,
        vert_offset: spy.float2 = spy.float2(0, 0),
        vert_scale: spy.float2 = spy.float2(1, 1),
        vert_z: float = 0.0,
        color: spy.float4 = spy.float4(0, 0, 0, 0),
        viewport: Optional[spy.Viewport] = None,
        scissor_rect: Optional[spy.ScissorRect] = None,
        clear: bool = True,
        depth_texture: Optional[spy.Texture] = None,
    ):
        command_encoder = self.ctx.device.create_command_encoder()

        rp_args: Any = {
            "color_attachments": [
                {
                    "view": self.ctx.output_texture.create_view({}),
                    "clear_value": [0.0, 0.0, 0.0, 1.0],
                    "load_op": spy.LoadOp.clear if clear else spy.LoadOp.load,
                    "store_op": spy.StoreOp.store,
                }
            ]
        }
        if depth_texture:
            rp_args["depth_stencil_attachment"] = {
                "view": depth_texture.create_view({}),
                "depth_load_op": spy.LoadOp.load,
                "depth_store_op": spy.StoreOp.store,
            }

        with command_encoder.begin_render_pass(rp_args) as encoder:
            encoder.set_render_state(
                {
                    "vertex_buffers": [self.vertex_buffer],
                    "index_buffer": self.index_buffer,
                    "index_format": spy.IndexFormat.uint32,
                    "viewports": [
                        (
                            viewport
                            if viewport
                            else spy.Viewport.from_size(
                                self.ctx.output_texture.width,
                                self.ctx.output_texture.height,
                            )
                        )
                    ],
                    "scissor_rects": [
                        (
                            scissor_rect
                            if scissor_rect
                            else spy.ScissorRect.from_size(
                                self.ctx.output_texture.width,
                                self.ctx.output_texture.height,
                            )
                        )
                    ],
                }
            )
            shader_object = encoder.bind_pipeline(pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.vert_offset = vert_offset
            cursor.vert_scale = vert_scale
            cursor.vert_z = float(vert_z)
            cursor.frag_color = color
            encoder.draw_indexed({"vertex_count": self.index_buffer.size // 4})
        self.ctx.device.submit_command_buffer(command_encoder.finish())

    # Helper to create pipeline with given set of args + correct program/layouts.
    def create_render_pipeline(self, **kwargs: Any):
        base_args = {
            "primitive_topology": spy.PrimitiveTopology.triangle_list,
            "targets": [{"format": spy.Format.rgba32_float}],
        }
        base_args.update(kwargs)

        return self.ctx.device.create_render_pipeline(
            program=self.program,
            input_layout=self.input_layout,
            **base_args,
        )

    # Helper to both create pipeline and then use it to draw quad.
    def draw_graphics_pipeline(
        self,
        vert_offset: spy.float2 = spy.float2(0, 0),
        vert_scale: spy.float2 = spy.float2(1, 1),
        vert_z: float = 0,
        color: spy.float4 = spy.float4(0, 0, 0, 0),
        clear: bool = True,
        viewport: Optional[spy.Viewport] = None,
        depth_texture: Optional[spy.Texture] = None,
        **kwargs: Any,
    ):
        pipeline = self.create_render_pipeline(**kwargs)
        self.draw(
            pipeline,
            color=color,
            clear=clear,
            vert_offset=vert_offset,
            vert_scale=vert_scale,
            vert_z=vert_z,
            viewport=viewport,
            depth_texture=depth_texture,
        )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gfx_simple_primitive(device_type: spy.DeviceType):
    ctx = PipelineTestContext(device_type)
    gfx = GfxContext(ctx)

    area = ctx.output_texture.width * ctx.output_texture.height
    scale = spy.float2(0.5)

    # Clear and fill red, then verify 1/4 pixels are red and all solid.
    gfx.draw_graphics_pipeline(
        color=spy.float4(1, 0, 0, 1),
        vert_scale=scale,
        rasterizer={"cull_mode": spy.CullMode.back},
    )
    ctx.expect_counts([int(area / 4), 0, 0, area])

    # Repeat with no culling, so should get same result.
    gfx.draw_graphics_pipeline(
        color=spy.float4(0, 1, 0, 1),
        vert_scale=scale,
        rasterizer={"cull_mode": spy.CullMode.none},
    )
    ctx.expect_counts([0, int(area / 4), 0, area])

    # Repeat with front face culling, so should get all black.
    gfx.draw_graphics_pipeline(
        color=spy.float4(1, 1, 1, 1),
        vert_scale=scale,
        rasterizer={"cull_mode": spy.CullMode.front},
    )
    ctx.expect_counts([0, 0, 0, area])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gfx_viewport(device_type: spy.DeviceType):
    ctx = PipelineTestContext(device_type)
    gfx = GfxContext(ctx)

    area = ctx.output_texture.width * ctx.output_texture.height
    scale = spy.float2(0.5)

    # Clear and fill red, and verify it filled the whole screen.
    gfx.draw_graphics_pipeline(
        color=spy.float4(1, 0, 0, 1), rasterizer={"cull_mode": spy.CullMode.back}
    )
    ctx.expect_counts([area, 0, 0, area])

    # Use viewport to clear half the screen.
    gfx.draw_graphics_pipeline(
        color=spy.float4(0, 1, 0, 1),
        rasterizer={"cull_mode": spy.CullMode.back},
        viewport=spy.Viewport(
            {
                "width": int(ctx.output_texture.width / 2),
                "height": ctx.output_texture.height,
            }
        ),
    )
    ctx.expect_counts([0, int(area / 2), 0, area])

    # Same using horiontal clip instead.
    gfx.draw_graphics_pipeline(
        color=spy.float4(0, 1, 0, 1),
        rasterizer={"cull_mode": spy.CullMode.back},
        viewport=spy.Viewport(
            {
                "width": ctx.output_texture.width,
                "height": int(ctx.output_texture.height / 2),
            }
        ),
    )
    ctx.expect_counts([0, int(area / 2), 0, area])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gfx_depth(device_type: spy.DeviceType):
    ctx = PipelineTestContext(device_type)
    gfx = GfxContext(ctx)

    # Create a depth texture and re-create frame buffer that uses depth.
    depth_texture = ctx.device.create_texture(
        format=spy.Format.d32_float,
        width=ctx.output_texture.width,
        height=ctx.output_texture.height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.depth_stencil,
        label="depth_texture",
    )

    area = ctx.output_texture.width * ctx.output_texture.height

    # Manually clear both buffers and verify results.
    command_encoder = ctx.device.create_command_encoder()
    command_encoder.clear_texture_float(ctx.output_texture, clear_value=[0.0, 0.0, 0.0, 1.0])
    command_encoder.clear_texture_depth_stencil(depth_texture, depth_value=0.5)
    ctx.device.submit_command_buffer(command_encoder.finish())
    ctx.expect_counts([0, 0, 0, area])

    # Write quad with z=0.25, which is close than the z buffer clear value of 0.5 so should come through.
    gfx.draw_graphics_pipeline(
        color=spy.float4(1, 0, 0, 1),
        clear=False,
        vert_scale=spy.float2(0.5),
        vert_z=0.25,
        rasterizer={"cull_mode": spy.CullMode.back},
        depth_stencil={
            "depth_test_enable": True,
            "depth_write_enable": True,
            "depth_func": spy.ComparisonFunc.less,
            "format": depth_texture.format,
        },
        depth_texture=depth_texture,
    )
    ctx.expect_counts([int(area / 4), 0, 0, area])

    # Write a great big quad at z=0.75, which should do nothing.
    gfx.draw_graphics_pipeline(
        color=spy.float4(1, 1, 0, 1),
        clear=False,
        vert_z=0.75,
        rasterizer={"cull_mode": spy.CullMode.back},
        depth_stencil={
            "depth_test_enable": True,
            "depth_write_enable": True,
            "depth_func": spy.ComparisonFunc.less,
            "format": depth_texture.format,
        },
        depth_texture=depth_texture,
    )
    ctx.expect_counts([int(area / 4), 0, 0, area])

    # Write a great big quad at z=0.4, which should overwrite the background but not the foreground.
    gfx.draw_graphics_pipeline(
        color=spy.float4(1, 1, 1, 1),
        clear=False,
        vert_z=0.4,
        rasterizer={"cull_mode": spy.CullMode.back},
        depth_stencil={
            "depth_test_enable": True,
            "depth_write_enable": True,
            "depth_func": spy.ComparisonFunc.less,
            "format": depth_texture.format,
        },
        depth_texture=depth_texture,
    )
    ctx.expect_counts([area, area - int(area / 4), area - int(area / 4), area])

    # Write a great big quad at z=0.75 with depth func always, which should just blat the lot.
    gfx.draw_graphics_pipeline(
        color=spy.float4(0, 0, 1, 1),
        clear=False,
        vert_z=0.75,
        rasterizer={"cull_mode": spy.CullMode.back},
        depth_stencil={
            "depth_test_enable": True,
            "depth_write_enable": True,
            "depth_func": spy.ComparisonFunc.always,
            "format": depth_texture.format,
        },
        depth_texture=depth_texture,
    )
    ctx.expect_counts([0, 0, area, area])

    # Quick check that the depth write happened correctly
    # Metal doesn't support reading from depth textures, so skip this check.
    if device_type != spy.DeviceType.metal:
        dt = depth_texture.to_numpy()
        assert np.all(dt == 0.75)

    # Try again at z=0.8, which should do nothing as z write was still enabled with the previous one.
    gfx.draw_graphics_pipeline(
        color=spy.float4(1, 1, 1, 1),
        clear=False,
        vert_z=0.8,
        rasterizer={"cull_mode": spy.CullMode.back},
        depth_stencil={
            "depth_test_enable": True,
            "depth_write_enable": True,
            "depth_func": spy.ComparisonFunc.less,
            "format": depth_texture.format,
        },
        depth_texture=depth_texture,
    )
    ctx.expect_counts([0, 0, area, area])

    # Write out a full quad at z=0.25, with z write turned off, so should work but not affect z buffer.
    gfx.draw_graphics_pipeline(
        color=spy.float4(1, 0, 0, 1),
        clear=False,
        vert_z=0.25,
        rasterizer={"cull_mode": spy.CullMode.back},
        depth_stencil={
            "depth_test_enable": True,
            "depth_write_enable": True,
            "depth_func": spy.ComparisonFunc.less,
            "format": depth_texture.format,
        },
        depth_texture=depth_texture,
    )
    ctx.expect_counts([area, 0, 0, area])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gfx_blend(device_type: spy.DeviceType):
    ctx = PipelineTestContext(device_type)
    gfx = GfxContext(ctx)
    area = ctx.output_texture.width * ctx.output_texture.height

    ctdesc: spy.ColorTargetDesc = spy.ColorTargetDesc(
        {
            "format": spy.Format.rgba32_float,
            "enable_blend": True,
            "color": {
                "src_factor": spy.BlendFactor.src_alpha,
                "dst_factor": spy.BlendFactor.inv_src_alpha,
                "op": spy.BlendOp.add,
            },
            "alpha": {
                "src_factor": spy.BlendFactor.zero,
                "dst_factor": spy.BlendFactor.one,
                "op": spy.BlendOp.add,
            },
        }
    )

    # Clear and then draw semi transparent red quad, and should get 1/4 dark red pixels.
    gfx.draw_graphics_pipeline(
        clear=True,
        color=spy.float4(1, 0, 0, 0.5),
        vert_scale=spy.float2(0.5),
        rasterizer={"cull_mode": spy.CullMode.back},
        targets=[ctdesc],
    )
    pixels = ctx.output_texture.to_numpy()
    is_pixel_red = np.all(pixels[:, :, :3] == [0.5, 0, 0], axis=2)
    assert np.sum(is_pixel_red) == int(area / 4)


# On Vulkan using 50% alpha coverage we get a checkerboard effect.
@pytest.mark.parametrize("device_type", [spy.DeviceType.vulkan])
def test_rhi_alpha_coverage(device_type: spy.DeviceType):
    if device_type == spy.DeviceType.vulkan and sys.platform == "darwin":
        pytest.skip("MoltenVK alpha coverage not working as expected")

    ctx = PipelineTestContext(device_type)
    gfx = GfxContext(ctx)
    area = ctx.output_texture.width * ctx.output_texture.height

    ctdesc: spy.ColorTargetDesc = spy.ColorTargetDesc(
        {
            "format": spy.Format.rgba32_float,
            "enable_blend": True,
            "color": {
                "src_factor": spy.BlendFactor.src_alpha,
            },
        }
    )

    # Clear and then draw semi transparent red quad, and should end up
    # with 1/8 of the pixels red due to alpha coverage.
    gfx.draw_graphics_pipeline(
        clear=True,
        color=spy.float4(1, 0, 0, 0.5),
        vert_scale=spy.float2(0.5),
        rasterizer={"cull_mode": spy.CullMode.back},
        targets=[ctdesc],
        multisample=spy.MultisampleDesc(
            {
                "alpha_to_coverage_enable": True,
            }
        ),
    )

    pixels = ctx.output_texture.to_numpy()
    is_pixel_red = np.all(pixels[:, :, :3] == [0.5, 0, 0], axis=2)
    assert np.sum(is_pixel_red) == int(area / 8)


class RayContext:
    def __init__(self, ctx: PipelineTestContext) -> None:
        super().__init__()
        if not ctx.device.has_feature(spy.Feature.acceleration_structure):
            pytest.skip("Acceleration structures not supported on this device")
        if not ctx.device.has_feature(spy.Feature.ray_tracing):
            pytest.skip("Ray tracing not supported on this device")

        self.ctx = ctx

        vertices = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], dtype=np.float32)
        indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)

        vertex_buffer = ctx.device.create_buffer(
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.acceleration_structure_build_input,
            label="vertex_buffer",
            data=vertices,
        )

        index_buffer = ctx.device.create_buffer(
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.acceleration_structure_build_input,
            label="index_buffer",
            data=indices,
        )

        blas_geometry_desc = spy.AccelerationStructureBuildInputTriangles()
        blas_geometry_desc.flags = spy.AccelerationStructureGeometryFlags.opaque
        blas_geometry_desc.vertex_buffers = [spy.BufferOffsetPair(vertex_buffer)]
        blas_geometry_desc.vertex_format = spy.Format.rgb32_float
        blas_geometry_desc.vertex_count = vertices.size // 3
        blas_geometry_desc.vertex_stride = vertices.itemsize * 3
        blas_geometry_desc.index_buffer = index_buffer
        blas_geometry_desc.index_format = spy.IndexFormat.uint32
        blas_geometry_desc.index_count = indices.size

        blas_build_desc = spy.AccelerationStructureBuildDesc()
        blas_build_desc.inputs = [blas_geometry_desc]

        blas_sizes = ctx.device.get_acceleration_structure_sizes(blas_build_desc)

        blas_scratch_buffer = ctx.device.create_buffer(
            size=blas_sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
            label="blas_scratch_buffer",
        )

        blas_buffer = ctx.device.create_buffer(
            size=blas_sizes.acceleration_structure_size,
            usage=spy.BufferUsage.acceleration_structure,
            label="blas_buffer",
        )

        blas = ctx.device.create_acceleration_structure(
            size=blas_buffer.size,
            label="blas",
        )

        command_encoder = ctx.device.create_command_encoder()
        command_encoder.build_acceleration_structure(
            desc=blas_build_desc, dst=blas, src=None, scratch_buffer=blas_scratch_buffer
        )
        ctx.device.submit_command_buffer(command_encoder.finish())

        self.blas = blas

    def create_instances(self, instance_transforms: Any):

        instance_list = self.ctx.device.create_acceleration_structure_instance_list(
            len(instance_transforms)
        )
        for idx, trans in enumerate(instance_transforms):
            instance_list.write(
                idx,
                {
                    "transform": trans,
                    "instance_id": idx,
                    "instance_mask": 0xFF,
                    "instance_contribution_to_hit_group_index": 0,
                    "flags": spy.AccelerationStructureInstanceFlags.none,
                    "acceleration_structure": self.blas.handle,
                },
            )

        tlas_build_desc = spy.AccelerationStructureBuildDesc(
            {
                "inputs": [instance_list.build_input_instances()],
            }
        )

        tlas_sizes = self.ctx.device.get_acceleration_structure_sizes(tlas_build_desc)

        tlas_scratch_buffer = self.ctx.device.create_buffer(
            size=tlas_sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
            label="tlas_scratch_buffer",
        )

        tlas = self.ctx.device.create_acceleration_structure(
            size=tlas_sizes.acceleration_structure_size,
            label="tlas",
        )

        command_encoder = self.ctx.device.create_command_encoder()
        command_encoder.build_acceleration_structure(
            desc=tlas_build_desc, dst=tlas, src=None, scratch_buffer=tlas_scratch_buffer
        )
        self.ctx.device.submit_command_buffer(command_encoder.finish())

        return tlas

    def dispatch_ray_grid(self, tlas: spy.AccelerationStructure, mode: str):
        if mode == "compute":
            if not self.ctx.device.has_feature(spy.Feature.ray_query):
                pytest.skip("Ray queries not supported on this device")
            self.dispatch_ray_grid_compute(tlas)
        elif mode == "ray":
            self.dispatch_ray_grid_rtp(tlas)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def dispatch_ray_grid_compute(self, tlas: spy.AccelerationStructure):
        program = self.ctx.device.load_program("test_pipeline_rt.slang", ["raygrid"])
        kernel = self.ctx.device.create_compute_kernel(program)
        kernel.dispatch(
            thread_count=[
                self.ctx.output_texture.width,
                self.ctx.output_texture.height,
                1,
            ],
            render_texture=self.ctx.output_texture,
            tlas=tlas,
            pos=spy.int2(0, 0),
            size=spy.int2(self.ctx.output_texture.width, self.ctx.output_texture.height),
            dist=float(2),
        )

    def dispatch_ray_grid_rtp(self, tlas: spy.AccelerationStructure):
        program = self.ctx.device.load_program(
            "test_pipeline_rt.slang", ["rt_ray_gen", "rt_miss", "rt_closest_hit"]
        )
        pipeline = self.ctx.device.create_ray_tracing_pipeline(
            program=program,
            hit_groups=[
                spy.HitGroupDesc(
                    hit_group_name="hit_group", closest_hit_entry_point="rt_closest_hit"
                )
            ],
            max_recursion=1,
            max_ray_payload_size=16,
        )

        shader_table = self.ctx.device.create_shader_table(
            program=program,
            ray_gen_entry_points=["rt_ray_gen"],
            miss_entry_points=["rt_miss"],
            hit_group_names=["hit_group"],
        )

        command_encoder = self.ctx.device.create_command_encoder()
        with command_encoder.begin_ray_tracing_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(pipeline, shader_table)
            cursor = spy.ShaderCursor(shader_object)
            cursor.rt_tlas = tlas
            cursor.rt_render_texture = self.ctx.output_texture
            pass_encoder.dispatch_rays(0, [1024, 1024, 1])
        self.ctx.device.submit_command_buffer(command_encoder.finish())


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("mode", ["compute", "ray"])
def test_raytrace_simple(device_type: spy.DeviceType, mode: str):
    ctx = PipelineTestContext(device_type)
    rtx = RayContext(ctx)

    # Setup instance transform causes the [0-1] quad to cover the top left
    # quarter of the screen. This is basically pixels 0-63, so we scale it up
    # a bit to handle rounding issues. The quad is at z=1 so should be visible.
    tf = spy.math.mul(
        spy.math.matrix_from_translation(spy.float3(-0.05, -0.05, 1)),
        spy.math.matrix_from_scaling(spy.float3(63.1, 63.1, 1)),
    )
    tf = spy.float3x4(tf)
    tlas = rtx.create_instances([tf])

    # Load and run the ray tracing kernel that fires a grid of rays
    # The grid covers the whole texture, and rays have length of 2 so
    # should hit the quad and turn the pixels red.
    rtx.dispatch_ray_grid(tlas, mode)

    # Check the 64x64 pixels are now red
    pixels = ctx.output_texture.to_numpy()
    is_pixel_red = np.all(pixels[:, :, :3] == [1, 0, 0], axis=2)
    num_red = np.sum(is_pixel_red)
    assert num_red == 4096


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("mode", ["compute", "ray"])
def test_raytrace_two_instance(device_type: spy.DeviceType, mode: str):
    ctx = PipelineTestContext(device_type)
    rtx = RayContext(ctx)

    # Ray trace against 2 instances, in top left and bottom right.
    transforms = []
    transforms.append(
        spy.math.mul(
            spy.math.matrix_from_translation(spy.float3(-0.05, -0.05, 1)),
            spy.math.matrix_from_scaling(spy.float3(63.1, 63.1, 1)),
        )
    )
    transforms.append(
        spy.math.mul(
            spy.math.matrix_from_translation(spy.float3(64 - 0.05, 64 - 0.05, 1)),
            spy.math.matrix_from_scaling(spy.float3(63.1, 63.1, 1)),
        )
    )

    tlas = rtx.create_instances([spy.float3x4(x) for x in transforms])
    rtx.dispatch_ray_grid(tlas, mode)

    # Expect 2 64x64 squares, with red from 1st instance and green from 2nd.
    pixels = ctx.output_texture.to_numpy()
    is_pixel_red = np.all(pixels[:, :, :3] == [1, 0, 0], axis=2)
    is_pixel_green = np.all(pixels[:, :, :3] == [0, 1, 0], axis=2)
    assert np.sum(is_pixel_red) == 4096
    assert np.sum(is_pixel_green) == 4096


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("mode", ["compute", "ray"])
def test_raytrace_closest_instance(device_type: spy.DeviceType, mode: str):
    ctx = PipelineTestContext(device_type)
    rtx = RayContext(ctx)

    # Ray trace against 2 instances, slightly overlapping,
    # with centre one closer.
    transforms = []
    transforms.append(
        spy.math.mul(
            spy.math.matrix_from_translation(spy.float3(-0.05, -0.05, 1)),
            spy.math.matrix_from_scaling(spy.float3(63.1, 63.1, 1)),
        )
    )
    transforms.append(
        spy.math.mul(
            spy.math.matrix_from_translation(spy.float3(32 - 0.05, 32 - 0.05, 0.5)),
            spy.math.matrix_from_scaling(spy.float3(63.1, 63.1, 1)),
        )
    )

    tlas = rtx.create_instances([spy.float3x4(x) for x in transforms])
    rtx.dispatch_ray_grid(tlas, mode)

    # Expect full green square, and only 3/4 of red square.
    pixels = ctx.output_texture.to_numpy()
    is_pixel_red = np.all(pixels[:, :, :3] == [1, 0, 0], axis=2)
    is_pixel_green = np.all(pixels[:, :, :3] == [0, 1, 0], axis=2)
    assert np.sum(is_pixel_red) == 3072
    assert np.sum(is_pixel_green) == 4096


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
