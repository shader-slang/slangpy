# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Optional
import sgl
import slangpy
from pathlib import Path


class App:
    def __init__(self, title: str = "SDF Match Example", width: int = 1024, height: int = 1024, device_type: sgl.DeviceType = sgl.DeviceType.d3d12):
        super().__init__()

        # Create SGL window
        self._window = sgl.Window(
            width=width, height=height, title=title, resizable=True
        )

        # Create SlangPy device with local include path for shaders
        self._device = slangpy.create_device(device_type,
                                             include_paths=[Path(__file__).parent])

        # Setup swapchain
        self.surface = self._device.create_surface(self._window)
        self.surface.configure(width=self._window.width, height=self._window.height)

        # Will contain output texture
        self._output_texture: 'sgl.Texture' = self.device.create_texture(
            format=sgl.Format.rgba16_float,
            width=width,
            height=height,
            mip_count=1,
            usage=sgl.TextureUsage.shader_resource
            | sgl.TextureUsage.unordered_access,
            label="output_texture",
        )

        # Store mouse pos
        self._mouse_pos = sgl.float2()

        # Internal events
        self._window.on_keyboard_event = self._on_window_keyboard_event
        self._window.on_mouse_event = self._on_window_mouse_event
        self._window.on_resize = self._on_window_resize

        # Hookable events
        self.on_keyboard_event: Optional[Callable[[sgl.KeyboardEvent], None]] = None
        self.on_mouse_event: Optional[Callable[[sgl.MouseEvent], None]] = None

    @property
    def device(self) -> sgl.Device:
        return self._device

    @property
    def window(self) -> sgl.Window:
        return self._window

    @property
    def mouse_pos(self) -> sgl.float2:
        return self._mouse_pos

    @property
    def output(self) -> sgl.Texture:
        return self._output_texture

    def process_events(self):
        if self._window.should_close():
            return False
        self._window.process_events()
        return True

    def present(self):
        image = self.surface.acquire_next_image()
        if image is None:
            return

        if (
            self._output_texture == None
            or self._output_texture.width != image.width
            or self._output_texture.height != image.height
        ):
            self._output_texture = self.device.create_texture(
                format=sgl.Format.rgba16_float,
                width=image.width,
                height=image.height,
                mip_count=1,
                usage=sgl.TextureUsage.shader_resource
                | sgl.TextureUsage.unordered_access,
                label="output_texture",
            )

        command_encoder = self._device.create_command_encoder()
        command_encoder.blit(image, self._output_texture)
        command_encoder.set_texture_state(image, sgl.ResourceState.present)
        self._device.submit_command_buffer(command_encoder.finish())

        del image
        self.surface.present()
        self._device.run_garbage_collection()

    def _on_window_keyboard_event(self, event: sgl.KeyboardEvent):
        if event.type == sgl.KeyboardEventType.key_press:
            if event.key == sgl.KeyCode.escape:
                self._window.close()
                return
            elif event.key == sgl.KeyCode.f1:
                if self._output_texture:
                    sgl.tev.show_async(self._output_texture)
                return
            elif event.key == sgl.KeyCode.f2:
                if self._output_texture:
                    bitmap = self._output_texture.to_bitmap()
                    bitmap.convert(
                        sgl.Bitmap.PixelFormat.rgb,
                        sgl.Bitmap.ComponentType.uint8,
                        srgb_gamma=True,
                    ).write_async("screenshot.png")
                return
        if self.on_keyboard_event:
            self.on_keyboard_event(event)

    def _on_window_mouse_event(self, event: sgl.MouseEvent):
        if event.type == sgl.MouseEventType.move:
            self._mouse_pos = event.pos
        if self.on_mouse_event:
            self.on_mouse_event(event)

    def _on_window_resize(self, width: int, height: int):
        self._device.wait()
        self.surface.configure(width=width, height=height)
