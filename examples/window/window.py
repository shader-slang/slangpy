# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent


class App:
    def __init__(self):
        super().__init__()
        self.window = spy.Window(width=1920, height=1280, title="Example", resizable=True)
        self.device = spy.Device(
            enable_debug_layers=True,
            compiler_options={"include_paths": [EXAMPLE_DIR]},
        )
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height)

        self.ui = spy.ui.Context(self.device)

        self.output_texture = None

        program = self.device.load_program("draw", ["compute_main"])
        self.kernel = self.device.create_compute_kernel(program)

        self.mouse_pos = spy.float2()
        self.mouse_down = False

        self.playing = True
        self.fps_avg = 0.0

        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

        self.setup_ui()

    def setup_ui(self):
        screen = self.ui.screen
        window = spy.ui.Window(screen, "Settings", size=spy.float2(500, 300))

        self.fps_text = spy.ui.Text(window, "FPS: 0")

        def start():
            self.playing = True

        spy.ui.Button(window, "Start", callback=start)

        def stop():
            self.playing = False

        spy.ui.Button(window, "Stop", callback=stop)

        self.noise_scale = spy.ui.SliderFloat(window, "Noise Scale", value=0.5, min=0, max=1)
        self.noise_amount = spy.ui.SliderFloat(window, "Noise Amount", value=0.5, min=0, max=1)
        self.mouse_radius = spy.ui.SliderFloat(window, "Radius", value=100, min=0, max=1000)

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if self.ui.handle_keyboard_event(event):
            return

        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.window.close()
            elif event.key == spy.KeyCode.f1:
                if self.output_texture:
                    spy.tev.show_async(self.output_texture)
            elif event.key == spy.KeyCode.f2:
                if self.output_texture:
                    bitmap = self.output_texture.to_bitmap()
                    bitmap.convert(
                        spy.Bitmap.PixelFormat.rgb,
                        spy.Bitmap.ComponentType.uint8,
                        srgb_gamma=True,
                    ).write_async("screenshot.png")

    def on_mouse_event(self, event: spy.MouseEvent):
        if self.ui.handle_mouse_event(event):
            return

        if event.type == spy.MouseEventType.move:
            self.mouse_pos = event.pos
        elif event.type == spy.MouseEventType.button_down:
            if event.button == spy.MouseButton.left:
                self.mouse_down = True
        elif event.type == spy.MouseEventType.button_up:
            if event.button == spy.MouseButton.left:
                self.mouse_down = False

    def on_resize(self, width: int, height: int):
        self.device.wait()
        if width > 0 and height > 0:
            self.surface.configure(width=width, height=height)
        else:
            self.surface.unconfigure()

    def run(self):
        frame = 0
        time = 0.0
        timer = spy.Timer()

        while not self.window.should_close():
            self.window.process_events()
            self.ui.process_events()

            elapsed = timer.elapsed_s()
            timer.reset()

            if self.playing:
                time += elapsed

            self.fps_avg = 0.95 * self.fps_avg + 0.05 * (1.0 / elapsed)
            self.fps_text.text = f"FPS: {self.fps_avg:.2f}"

            if not self.surface.config:
                continue
            surface_texture = self.surface.acquire_next_image()
            if not surface_texture:
                continue

            if (
                self.output_texture == None
                or self.output_texture.width != surface_texture.width
                or self.output_texture.height != surface_texture.height
            ):
                self.output_texture = self.device.create_texture(
                    format=spy.Format.rgba16_float,
                    width=surface_texture.width,
                    height=surface_texture.height,
                    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                    label="output_texture",
                )

            command_encoder = self.device.create_command_encoder()
            self.kernel.dispatch(
                thread_count=[self.output_texture.width, self.output_texture.height, 1],
                vars={
                    "g_output": self.output_texture,
                    "g_frame": frame,
                    "g_mouse_pos": self.mouse_pos,
                    "g_mouse_down": self.mouse_down,
                    "g_mouse_radius": self.mouse_radius.value,
                    "g_time": time,
                    "g_noise_scale": self.noise_scale.value,
                    "g_noise_amount": self.noise_amount.value,
                },
                command_encoder=command_encoder,
            )
            command_encoder.blit(surface_texture, self.output_texture)

            self.ui.new_frame(surface_texture.width, surface_texture.height)
            self.ui.render(surface_texture, command_encoder)

            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture

            self.surface.present()

            frame += 1


if __name__ == "__main__":
    app = App()
    app.run()
