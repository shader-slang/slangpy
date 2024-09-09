# SPDX-License-Identifier: Apache-2.0

import sgl
from pathlib import Path
import kernelfunctions as kf

ROOT_DIR = Path(__file__).parent


class App:
    def __init__(self):
        super().__init__()
        self.window = sgl.Window(
            width=1280, height=720, title="Slime Mold", resizable=True
        )
        self.device = sgl.Device(
            enable_debug_layers=True,
            compiler_options={"include_paths": [ROOT_DIR]},
        )
        self.swapchain = self.device.create_swapchain(
            image_count=3,
            width=self.window.width,
            height=self.window.height,
            window=self.window,
            enable_vsync=False,
        )

        self.framebuffers = []
        self.create_framebuffers()

        self.ui = sgl.ui.Context(self.device)

        self.output_texture = None

        self.core_module = self.device.load_module("smcore.slang")
        self.particle_type = self.core_module.layout.find_type_by_name("Particle")
        self.particle_layout = self.core_module.layout.get_type_layout(self.particle_type)

        self.particles: list[kf.NDBuffer] = []
        for i in range(0, 2):
            self.particles.append(kf.NDBuffer(
                element_count=100,
                device=self.device,
                element_type=self.particle_layout,
            ))
        self.read_idx = 0

        self.init_particle = kf.Function(self.core_module, "init_particle")
        self.write_particle_pixels = kf.Function(
            self.core_module, "write_particle_pixels")

        self.mouse_pos = sgl.float2()
        self.mouse_down = False

        self.playing = True
        self.fps_avg = 0.0

        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

        self.setup_ui()

        self.init_sim()

    def setup_ui(self):
        screen = self.ui.screen
        window = sgl.ui.Window(screen, "Settings", size=sgl.float2(125, 100))

        self.fps_text = sgl.ui.Text(window, "FPS: 0")

        def start():
            self.playing = True
        sgl.ui.Button(window, "Start", callback=start)

        def stop():
            self.playing = False
        sgl.ui.Button(window, "Stop", callback=stop)

    def init_sim(self):
        self.read_idx = 0
        self.init_particle(
            position={
                'x': kf.RandFloatArg(0, self.window.width, 1),
                'y': kf.RandFloatArg(0, self.window.height, 1),
            },
            angle=kf.RandFloatArg(0, 2 * 3.14159, 1),
            particle=self.particles[self.read_idx])

        data = self.particles[self.read_idx].buffer.to_numpy().view("float32")
        print(data)

    def on_keyboard_event(self, event: sgl.KeyboardEvent):
        if self.ui.handle_keyboard_event(event):
            return

        if event.type == sgl.KeyboardEventType.key_press:
            if event.key == sgl.KeyCode.escape:
                self.window.close()
            elif event.key == sgl.KeyCode.f5:
                self.device.reload_all_programs()

    def on_mouse_event(self, event: sgl.MouseEvent):
        if self.ui.handle_mouse_event(event):
            return
        if event.type == sgl.MouseEventType.move:
            self.mouse_pos = event.pos
        elif event.type == sgl.MouseEventType.button_down:
            if event.button == sgl.MouseButton.left:
                self.mouse_down = True
        elif event.type == sgl.MouseEventType.button_up:
            if event.button == sgl.MouseButton.left:
                self.mouse_down = False

    def on_resize(self, width: int, height: int):
        self.framebuffers.clear()
        self.device.wait()
        self.swapchain.resize(width, height)
        self.create_framebuffers()

    def create_framebuffers(self):
        self.framebuffers = [
            self.device.create_framebuffer(render_targets=[image.get_rtv()])
            for image in self.swapchain.images
        ]

    def run(self):
        frame = 0
        time = 0.0
        timer = sgl.Timer()

        while not self.window.should_close():
            self.window.process_events()
            self.ui.process_events()

            elapsed = timer.elapsed_s()
            timer.reset()

            if self.playing:
                time += elapsed

            self.fps_avg = 0.95 * self.fps_avg + 0.05 * (1.0 / elapsed)
            self.fps_text.text = f"FPS: {self.fps_avg:.2f}"

            image_index = self.swapchain.acquire_next_image()
            if image_index < 0:
                continue

            image = self.swapchain.get_image(image_index)
            if (
                self.output_texture == None
                or self.output_texture.width != image.width
                or self.output_texture.height != image.height
            ):
                self.output_texture = self.device.create_texture(
                    format=sgl.Format.rgba16_float,
                    width=image.width,
                    height=image.height,
                    mip_count=1,
                    usage=sgl.ResourceUsage.shader_resource
                    | sgl.ResourceUsage.unordered_access,
                    debug_name="output_texture",
                )

            command_buffer = self.device.create_command_buffer()
            command_buffer.clear_texture(
                self.output_texture, sgl.float4(0.0, 0.0, 0.0, 1.0))
            command_buffer.submit()

            self.write_particle_pixels(
                particles=self.particles[self.read_idx],
                output=self.output_texture,
            )

            command_buffer = self.device.create_command_buffer()
            command_buffer.blit(image, self.output_texture)
            self.ui.new_frame(image.width, image.height)
            self.ui.render(self.framebuffers[image_index], command_buffer)
            command_buffer.set_texture_state(image, sgl.ResourceState.present)
            command_buffer.submit()

            del image

            self.swapchain.present()
            self.device.run_garbage_collection()
            frame += 1


if __name__ == "__main__":
    app = App()
    app.run()
