# SPDX-License-Identifier: Apache-2.0

import sgl
from sgl import ShaderCursor, float2, float3
from pathlib import Path
import kernelfunctions as kf
from kernelfunctions.instance import InstanceList, InstanceListBuffer
from kernelfunctions.module import Module
from kernelfunctions.struct import Struct
from kernelfunctions.types.buffer import NDBuffer
from kernelfunctions.utils import find_type_layout_for_buffer
import numpy as np

ROOT_DIR = Path(__file__).parent
SLANGPY_DIR = ROOT_DIR.parent.parent / "slang"


class App:
    def __init__(self):
        super().__init__()
        self.window = sgl.Window(
            width=1280, height=720, title="Particles", resizable=True
        )
        self.device = sgl.Device(
            enable_debug_layers=True,
            compiler_options={"include_paths": [ROOT_DIR, SLANGPY_DIR]},
        )
        self.swapchain = self.device.create_swapchain(
            image_count=3,
            width=self.window.width,
            height=self.window.height,
            window=self.window,
            enable_vsync=False,
        )

        self.framebuffers: list[sgl.Framebuffer] = []
        self.create_framebuffers()

        self.load_module()

        self.draw_program = self.device.load_program(
            "particles.slang", ["vertex_main", "fragment_main"]
        )
        self.draw_pipeline = self.device.create_graphics_pipeline(
            program=self.draw_program,
            input_layout=None,
            framebuffer_layout=self.framebuffers[0].layout,
            blend=sgl.BlendDesc(
            {
                "alpha_to_coverage_enable": False,
                "targets": [
                    {
                        "enable_blend": True,
                        "color": {
                            "src_factor": sgl.BlendFactor.one,
                            "dst_factor": sgl.BlendFactor.one,
                            "op": sgl.BlendOp.add,
                        },
                        "alpha": {
                            "src_factor": sgl.BlendFactor.zero,
                            "dst_factor": sgl.BlendFactor.one,
                            "op": sgl.BlendOp.add,
                        },
                    }
                ],
            })
        )

        self.init_sim()

    def load_module(self):
        self.module = Module(self.device.load_module("particles.slang"))
        self.particle_struct = self.module["Particle"].as_struct()

        self.particle_shapes = (10000000,)
        self.particles = InstanceListBuffer(self.particle_struct, self.particle_shapes)

    def init_sim(self):
        self.particles.init(
            pos={
                'x': kf.RandFloatArg(0, self.window.width, 1, 121),
                'y': kf.RandFloatArg(0, self.window.height, 1, 4342),
            },
            vel=kf.RandFloatArg(-30, 30, 2, 321),
            lt=kf.RandFloatArg(10, 20, 1, 65),
            col=kf.RandFloatArg(0, 1, 3, 47342)
        )

    def update_sim(self):
        self.particles.add_point_gravity(
            float2(self.window.width/2, self.window.height/2), 1000000.0)
        # constant random impulse each frame
        self.particles.add_impulse(kf.RandFloatArg(-1, 1, 2))
        self.particles.add_wind(1.0)  # sin wave based wind
        self.particles.update(1.0/60.0)  # update

    def load_module_explicit(self):
        self.module = Module(self.device.load_module("particles.slang"))
        self.particle_struct = self.module["Particle"].as_struct()

        self.particle_shapes = (10000000,)

        # Some form of 'vectorize' specification has to happen here to describe how to treat
        # the 'this' parameter passed into each Particle method
        self.particles = InstanceListBuffer(
            self.particle_struct.vectorize(0,), self.particle_shapes)

    def init_sim_explicit(self):

        # Vectorize all arguments of init
        # Also need some form of explicit specification that the pos is a float2
        self.particles.init.vectorize(
            pos={
                'x': (0,),
                'y': (0,),
            },
            vel=(0,),
            lt=(0,),
            col=(0,)
        )(
            pos=_float2({
                'x': kf.RandFloatArg(0, self.window.width, 1, 121),
                'y': kf.RandFloatArg(0, self.window.height, 1, 4342),
            }),
            vel=kf.RandFloatArg(-30, 30, 2, 321),
            lt=kf.RandFloatArg(10, 20, 1, 65),
            col=kf.RandFloatArg(0, 1, 3, 47342)
        )

    def update_sim_explicit(self):
        self.particles.add_point_gravity(
            float2(self.window.width/2, self.window.height/2), 1000000.0)

        # Vectorize add_impulse
        self.particles.add_impulse.vectorize((0,))(
            kf.RandFloatArg(-1, 1, 2))  # constant random impulse each frame

        self.particles.add_wind(1.0)  # sin wave based wind
        self.particles.update(1.0/60.0)  # update

    def create_framebuffers(self):
        self.framebuffers = [
            self.device.create_framebuffer(render_targets=[image.get_rtv()])
            for image in self.swapchain.images
        ]

    def run(self):

        self.temp_buffer = NDBuffer(self.device, shape=self.particle_shapes,
                                    element_type=self.particle_struct.get_struct_layout())

        while not self.window.should_close():
            self.window.process_events()

            self.update_sim()

            image_index = self.swapchain.acquire_next_image()
            if image_index < 0:
                continue

            image = self.swapchain.get_image(image_index)

            command_buffer = self.device.create_command_buffer()
            command_buffer.clear_texture(image, sgl.float4(0.0, 0.0, 0.0, 1.0))

            self.particle_struct["__init"](self.particles, _result=self.temp_buffer)
            with command_buffer.encode_render_commands(self.framebuffers[image_index]) as encoder:
                shader_object = encoder.bind_pipeline(self.draw_pipeline)
                cursor = ShaderCursor(shader_object)
                cursor["draw_particles"] = self.temp_buffer.buffer
                cursor["screen_size"] = float2(image.width, image.height)
                encoder.set_primitive_topology(sgl.PrimitiveTopology.triangle_list)
                encoder.set_viewport_and_scissor_rect(
                    {"width": image.width, "height": image.height}
                )
                encoder.draw(self.temp_buffer.shape[0]*6)

            command_buffer.set_texture_state(image, sgl.ResourceState.present)

            command_buffer.submit()

            del image

            self.swapchain.present()
            self.device.run_garbage_collection()

            # Valid even though in_tex is a float, as slang can convert
            # in_tex = float texture
            # out_tex = float3 texture
            tonemap(kf.float1(in_tex), 1.0, 0.1, 0.5, 0.5, _result=out_tex)

            # Invalid because in_tex is a float2, which slang can not convert
            in_tex = self.device.create_texture(
                format=sgl.Format.rg32_float, width=256, height=256)
            out_tex = self.device.create_texture(
                format=sgl.Format.rgb32_float, width=256, height=256)
            tonemap(kf.float3(in_tex), 1.0, 0.1, 0.5, 0.5, _result=kf.float3(out_tex))


if __name__ == "__main__":
    app = App()
    app.run()

# Vectorize tonemap
tonemap.vectorize((0, 1), (), (), (), (), (0, 1))(
    my_texture, 1.0, 0.1, 0.5, 0.5, _result=framebuffer_texture)

# Vectorize trace ray
trace_ray.vectorize(
    {'position': (0,), 'direction': ()}, (0,))(
    Ray({'position': positions_buffer, 'direction': float3(0, 0, 1)}), results_buffer))
