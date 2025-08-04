# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
import slangpy as spy
import numpy as np
import numpy.typing as npt
from pathlib import Path
from dataclasses import dataclass
import struct

EXAMPLE_DIR = Path(__file__).parent

USE_RAYTRACING_PIPELINE = True


class Camera:
    def __init__(self):
        super().__init__()
        self.width = 100
        self.height = 100
        self.aspect_ratio = 1.0
        self.position = spy.float3(1, 1, 1)
        self.target = spy.float3(0, 0, 0)
        self.up = spy.float3(0, 1, 0)
        self.fov = 70.0
        self.recompute()

    def recompute(self):
        self.aspect_ratio = float(self.width) / float(self.height)

        self.fwd = spy.math.normalize(self.target - self.position)
        self.right = spy.math.normalize(spy.math.cross(self.fwd, self.up))
        self.up = spy.math.normalize(spy.math.cross(self.right, self.fwd))

        fov = spy.math.radians(self.fov)

        self.image_u = self.right * spy.math.tan(fov * 0.5) * self.aspect_ratio
        self.image_v = self.up * spy.math.tan(fov * 0.5)
        self.image_w = self.fwd

    def bind(self, cursor: spy.ShaderCursor):
        cursor["position"] = self.position
        cursor["image_u"] = self.image_u
        cursor["image_v"] = self.image_v
        cursor["image_w"] = self.image_w


class CameraController:
    MOVE_KEYS = {
        spy.KeyCode.a: spy.float3(-1, 0, 0),
        spy.KeyCode.d: spy.float3(1, 0, 0),
        spy.KeyCode.e: spy.float3(0, 1, 0),
        spy.KeyCode.q: spy.float3(0, -1, 0),
        spy.KeyCode.w: spy.float3(0, 0, 1),
        spy.KeyCode.s: spy.float3(0, 0, -1),
    }
    MOVE_SHIFT_FACTOR = 10.0

    def __init__(self, camera: Camera):
        super().__init__()
        self.camera = camera
        self.mouse_down = False
        self.mouse_pos = spy.float2()
        self.key_state = {k: False for k in CameraController.MOVE_KEYS.keys()}
        self.shift_down = False

        self.move_delta = spy.float3()
        self.rotate_delta = spy.float2()

        self.move_speed = 1.0
        self.rotate_speed = 0.002

    def update(self, dt: float):
        changed = False
        position = self.camera.position
        fwd = self.camera.fwd
        up = self.camera.up
        right = self.camera.right

        # Move
        if spy.math.length(self.move_delta) > 0:
            offset = right * self.move_delta.x
            offset += up * self.move_delta.y
            offset += fwd * self.move_delta.z
            factor = CameraController.MOVE_SHIFT_FACTOR if self.shift_down else 1.0
            offset *= self.move_speed * factor * dt
            position += offset
            changed = True

        # Rotate
        if spy.math.length(self.rotate_delta) > 0:
            yaw = spy.math.atan2(fwd.z, fwd.x)
            pitch = spy.math.asin(fwd.y)
            yaw += self.rotate_speed * self.rotate_delta.x
            pitch -= self.rotate_speed * self.rotate_delta.y
            fwd = spy.float3(
                spy.math.cos(yaw) * spy.math.cos(pitch),
                spy.math.sin(pitch),
                spy.math.sin(yaw) * spy.math.cos(pitch),
            )
            self.rotate_delta = spy.float2()
            changed = True

        if changed:
            self.camera.position = position
            self.camera.target = position + fwd
            self.camera.up = spy.float3(0, 1, 0)
            self.camera.recompute()

        return changed

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if event.is_key_press() or event.is_key_release():
            down = event.is_key_press()
            if event.key in CameraController.MOVE_KEYS:
                self.key_state[event.key] = down
            elif event.key == spy.KeyCode.left_shift:
                self.shift_down = down
        self.move_delta = spy.float3()
        for key, state in self.key_state.items():
            if state:
                self.move_delta += CameraController.MOVE_KEYS[key]

    def on_mouse_event(self, event: spy.MouseEvent):
        self.rotate_delta = spy.float2()
        if event.is_button_down() and event.button == spy.MouseButton.left:
            self.mouse_down = True
        if event.is_button_up() and event.button == spy.MouseButton.left:
            self.mouse_down = False
        if event.is_move():
            mouse_delta = event.pos - self.mouse_pos
            if self.mouse_down:
                self.rotate_delta = mouse_delta
            self.mouse_pos = event.pos


class Material:
    def __init__(self, base_color: "spy.float3param" = spy.float3(0.5)):
        super().__init__()
        self.base_color = base_color


class Mesh:
    def __init__(
        self,
        vertices: npt.NDArray[np.float32],  # type: ignore
        indices: npt.NDArray[np.uint32],  # type: ignore
    ):
        super().__init__()
        assert vertices.ndim == 2 and vertices.dtype == np.float32
        assert indices.ndim == 2 and indices.dtype == np.uint32
        self.vertices = vertices
        self.indices = indices

    @property
    def vertex_count(self):
        return self.vertices.shape[0]

    @property
    def triangle_count(self):
        return self.indices.shape[0]

    @property
    def index_count(self):
        return self.triangle_count * 3

    @classmethod
    def create_quad(cls, size: "spy.float2param" = spy.float2(1)):
        vertices = np.array(
            [
                # position, normal, uv
                [-0.5, 0, -0.5, 0, 1, 0, 0, 0],
                [+0.5, 0, -0.5, 0, 1, 0, 1, 0],
                [-0.5, 0, +0.5, 0, 1, 0, 0, 1],
                [+0.5, 0, +0.5, 0, 1, 0, 1, 1],
            ],
            dtype=np.float32,
        )
        vertices[:, (0, 2)] *= [size[0], size[1]]
        indices = np.array(
            [
                [2, 1, 0],
                [1, 2, 3],
            ],
            dtype=np.uint32,
        )
        return Mesh(vertices, indices)

    @classmethod
    def create_cube(cls, size: "spy.float3param" = spy.float3(1)):
        vertices = np.array(
            [
                # position, normal, uv
                # left
                [-0.5, -0.5, -0.5, 0, -1, 0, 0.0, 0.0],
                [-0.5, -0.5, +0.5, 0, -1, 0, 1.0, 0.0],
                [+0.5, -0.5, +0.5, 0, -1, 0, 1.0, 1.0],
                [+0.5, -0.5, -0.5, 0, -1, 0, 0.0, 1.0],
                # right
                [-0.5, +0.5, +0.5, 0, +1, 0, 0.0, 0.0],
                [-0.5, +0.5, -0.5, 0, +1, 0, 1.0, 0.0],
                [+0.5, +0.5, -0.5, 0, +1, 0, 1.0, 1.0],
                [+0.5, +0.5, +0.5, 0, +1, 0, 0.0, 1.0],
                # back
                [-0.5, +0.5, -0.5, 0, 0, -1, 0.0, 0.0],
                [-0.5, -0.5, -0.5, 0, 0, -1, 1.0, 0.0],
                [+0.5, -0.5, -0.5, 0, 0, -1, 1.0, 1.0],
                [+0.5, +0.5, -0.5, 0, 0, -1, 0.0, 1.0],
                # front
                [+0.5, +0.5, +0.5, 0, 0, +1, 0.0, 0.0],
                [+0.5, -0.5, +0.5, 0, 0, +1, 1.0, 0.0],
                [-0.5, -0.5, +0.5, 0, 0, +1, 1.0, 1.0],
                [-0.5, +0.5, +0.5, 0, 0, +1, 0.0, 1.0],
                # bottom
                [-0.5, +0.5, +0.5, -1, 0, 0, 0.0, 0.0],
                [-0.5, -0.5, +0.5, -1, 0, 0, 1.0, 0.0],
                [-0.5, -0.5, -0.5, -1, 0, 0, 1.0, 1.0],
                [-0.5, +0.5, -0.5, -1, 0, 0, 0.0, 1.0],
                # top
                [+0.5, +0.5, -0.5, +1, 0, 0, 0.0, 0.0],
                [+0.5, -0.5, -0.5, +1, 0, 0, 1.0, 0.0],
                [+0.5, -0.5, +0.5, +1, 0, 0, 1.0, 1.0],
                [+0.5, +0.5, +0.5, +1, 0, 0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        vertices[:, 0:3] *= [size[0], size[1], size[2]]

        indices = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 6, 5],
                [4, 7, 6],
                [8, 10, 9],
                [8, 11, 10],
                [12, 14, 13],
                [12, 15, 14],
                [16, 18, 17],
                [16, 19, 18],
                [20, 22, 21],
                [20, 23, 22],
            ],
            dtype=np.uint32,
        )

        return Mesh(vertices, indices)


class Transform:
    def __init__(self):
        super().__init__()
        self.translation = spy.float3(0)
        self.scaling = spy.float3(1)
        self.rotation = spy.float3(0)
        self.matrix = spy.float4x4.identity()

    def update_matrix(self):
        T = spy.math.matrix_from_translation(self.translation)
        S = spy.math.matrix_from_scaling(self.scaling)
        R = spy.math.matrix_from_rotation_xyz(self.rotation)
        self.matrix = spy.math.mul(spy.math.mul(T, R), S)


class Stage:
    def __init__(self):
        super().__init__()
        self.camera = Camera()
        self.materials = []
        self.meshes = []
        self.transforms = []
        self.instances = []

    def add_material(self, material: Material):
        material_id = len(self.materials)
        self.materials.append(material)
        return material_id

    def add_mesh(self, mesh: Mesh):
        mesh_id = len(self.meshes)
        self.meshes.append(mesh)
        return mesh_id

    def add_transform(self, transform: Transform):
        transform_id = len(self.transforms)
        self.transforms.append(transform)
        return transform_id

    def add_instance(self, mesh_id: int, material_id: int, transform_id: int):
        instance_id = len(self.instances)
        self.instances.append((mesh_id, material_id, transform_id))
        return instance_id

    @classmethod
    def demo(cls):
        stage = Stage()
        stage.camera.target = spy.float3(0, 1, 0)
        stage.camera.position = spy.float3(2, 1, 2)

        floor_material = stage.add_material(Material(base_color=spy.float3(0.5)))
        floor_mesh = stage.add_mesh(Mesh.create_quad([5, 5]))
        floor_transform = stage.add_transform(Transform())
        stage.add_instance(floor_mesh, floor_material, floor_transform)

        cube_materials = []
        for _ in range(10):
            cube_materials.append(
                stage.add_material(
                    Material(base_color=spy.float3(np.random.rand(3).astype(np.float32)))  # type: ignore (TYPINGTODO: need explicit np->float conversion)
                )
            )
        cube_mesh = stage.add_mesh(Mesh.create_cube([0.1, 0.1, 0.1]))

        for i in range(1000):
            transform = Transform()
            transform.translation = spy.float3((np.random.rand(3) * 2 - 1).astype(np.float32))  # type: ignore (TYPINGTODO: need explicit np->float conversion)
            transform.translation[1] += 1
            transform.scaling = spy.float3((np.random.rand(3) + 0.5).astype(np.float32))  # type: ignore (TYPINGTODO: need explicit np->float conversion)
            transform.rotation = spy.float3((np.random.rand(3) * 10).astype(np.float32))  # type: ignore (TYPINGTODO: need explicit np->float conversion)
            transform.update_matrix()
            cube_transform = stage.add_transform(transform)
            stage.add_instance(cube_mesh, cube_materials[i % len(cube_materials)], cube_transform)

        return stage


class Scene:
    @dataclass
    class MaterialDesc:
        base_color: spy.float3

        def pack(self):
            return struct.pack("fff", self.base_color[0], self.base_color[1], self.base_color[2])

    @dataclass
    class MeshDesc:
        vertex_count: int
        index_count: int
        vertex_offset: int
        index_offset: int

        def pack(self):
            return struct.pack(
                "IIII",
                self.vertex_count,
                self.index_count,
                self.vertex_offset,
                self.index_offset,
            )

    @dataclass
    class InstanceDesc:
        mesh_id: int
        material_id: int
        transform_id: int

        def pack(self):
            return struct.pack("III", self.mesh_id, self.material_id, self.transform_id)

    def __init__(self, device: spy.Device, stage: Stage):
        super().__init__()
        self.device = device

        self.camera = stage.camera

        # Prepare material descriptors
        self.material_descs = [Scene.MaterialDesc(base_color=m.base_color) for m in stage.materials]
        material_descs_data = np.frombuffer(
            b"".join(d.pack() for d in self.material_descs), dtype=np.uint8
        ).flatten()
        self.material_descs_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="material_descs_buffer",
            data=material_descs_data,
        )

        # Prepare mesh descriptors
        vertex_count = 0
        index_count = 0
        self.mesh_descs = []
        for mesh in stage.meshes:
            self.mesh_descs.append(
                Scene.MeshDesc(
                    vertex_count=mesh.vertex_count,
                    index_count=mesh.index_count,
                    vertex_offset=vertex_count,
                    index_offset=index_count,
                )
            )
            vertex_count += mesh.vertex_count
            index_count += mesh.index_count

        # Prepare instance descriptors
        self.instance_descs = []
        for mesh_id, material_id, transform_id in stage.instances:
            self.instance_descs.append(Scene.InstanceDesc(mesh_id, material_id, transform_id))

        # Create vertex and index buffers
        vertices = np.concatenate([mesh.vertices for mesh in stage.meshes], axis=0)
        indices = np.concatenate([mesh.indices for mesh in stage.meshes], axis=0)
        assert vertices.shape[0] == vertex_count
        assert indices.shape[0] == index_count // 3

        self.vertex_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="vertex_buffer",
            data=vertices,
        )

        self.index_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="index_buffer",
            data=indices,
        )

        mesh_descs_data = np.frombuffer(
            b"".join(d.pack() for d in self.mesh_descs), dtype=np.uint8
        ).flatten()
        self.mesh_descs_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="mesh_descs_buffer",
            data=mesh_descs_data,
        )

        instance_descs_data = np.frombuffer(
            b"".join(d.pack() for d in self.instance_descs), dtype=np.uint8
        ).flatten()
        self.instance_descs_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="instance_descs_buffer",
            data=instance_descs_data,
        )

        # Prepare transforms
        self.transforms = [t.matrix for t in stage.transforms]
        self.inverse_transpose_transforms = [
            spy.math.transpose(spy.math.inverse(t)) for t in self.transforms
        ]
        self.transform_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="transform_buffer",
            data=np.stack([t.to_numpy() for t in self.transforms]),
        )
        self.inverse_transpose_transforms_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="inverse_transpose_transforms_buffer",
            data=np.stack([t.to_numpy() for t in self.inverse_transpose_transforms]),
        )

        # Build BLASes
        self.blases = [self.build_blas(mesh_desc) for mesh_desc in self.mesh_descs]

        # Build TLAS
        self.tlas = self.build_tlas()

    def build_blas(self, mesh_desc: MeshDesc):
        build_input = spy.AccelerationStructureBuildInputTriangles(
            {
                "vertex_buffers": [
                    {
                        "buffer": self.vertex_buffer,
                        "offset": mesh_desc.vertex_offset * 32,
                    }
                ],
                "vertex_format": spy.Format.rgb32_float,
                "vertex_count": mesh_desc.vertex_count,
                "vertex_stride": 32,
                "index_buffer": {
                    "buffer": self.index_buffer,
                    "offset": mesh_desc.index_offset * 4,
                },
                "index_format": spy.IndexFormat.uint32,
                "index_count": mesh_desc.index_count,
                "flags": spy.AccelerationStructureGeometryFlags.opaque,
            }
        )

        build_desc = spy.AccelerationStructureBuildDesc({"inputs": [build_input]})

        sizes = self.device.get_acceleration_structure_sizes(build_desc)

        blas_scratch_buffer = self.device.create_buffer(
            size=sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
            label="blas_scratch_buffer",
        )

        blas = self.device.create_acceleration_structure(
            size=sizes.acceleration_structure_size,
            label="blas",
        )

        command_encoder = self.device.create_command_encoder()
        command_encoder.build_acceleration_structure(
            desc=build_desc, dst=blas, src=None, scratch_buffer=blas_scratch_buffer
        )
        self.device.submit_command_buffer(command_encoder.finish())

        return blas

    def build_tlas(self):
        instance_list = self.device.create_acceleration_structure_instance_list(
            size=len(self.instance_descs)
        )
        for instance_id, instance_desc in enumerate(self.instance_descs):
            instance_list.write(
                instance_id,
                {
                    "transform": spy.float3x4(self.transforms[instance_desc.transform_id]),
                    "instance_id": instance_id,
                    "instance_mask": 0xFF,
                    "instance_contribution_to_hit_group_index": 0,
                    "flags": spy.AccelerationStructureInstanceFlags.none,
                    "acceleration_structure": self.blases[instance_desc.mesh_id].handle,
                },
            )

        build_desc = spy.AccelerationStructureBuildDesc(
            {
                "inputs": [instance_list.build_input_instances()],
            }
        )

        sizes = self.device.get_acceleration_structure_sizes(build_desc)

        tlas_scratch_buffer = self.device.create_buffer(
            size=sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
            label="tlas_scratch_buffer",
        )

        tlas = self.device.create_acceleration_structure(
            size=sizes.acceleration_structure_size,
            label="tlas",
        )

        command_encoder = self.device.create_command_encoder()
        command_encoder.build_acceleration_structure(
            desc=build_desc, dst=tlas, src=None, scratch_buffer=tlas_scratch_buffer
        )
        self.device.submit_command_buffer(command_encoder.finish())

        return tlas

    def bind(self, cursor: spy.ShaderCursor):
        cursor["tlas"] = self.tlas
        cursor["material_descs"] = self.material_descs_buffer
        cursor["mesh_descs"] = self.mesh_descs_buffer
        cursor["instance_descs"] = self.instance_descs_buffer
        cursor["vertices"] = self.vertex_buffer
        cursor["indices"] = self.index_buffer
        cursor["transforms"] = self.transform_buffer
        cursor["inverse_transpose_transforms"] = self.inverse_transpose_transforms_buffer
        self.camera.bind(cursor["camera"])


class PathTracer:
    def __init__(self, device: spy.Device, scene: Scene):
        super().__init__()
        self.device = device
        self.scene = scene

        if USE_RAYTRACING_PIPELINE:
            self.program = self.device.load_program(
                "pathtracer.slang", ["rt_ray_gen", "rt_closest_hit", "rt_miss"]
            )
            self.rt_pipeline = self.device.create_ray_tracing_pipeline(
                program=self.program,
                hit_groups=[
                    {
                        "hit_group_name": "default",
                        "closest_hit_entry_point": "rt_closest_hit",
                    }
                ],
                max_recursion=6,
                max_ray_payload_size=128,
            )
            self.shader_table = self.device.create_shader_table(
                program=self.program,
                ray_gen_entry_points=["rt_ray_gen"],
                miss_entry_points=["rt_miss"],
                hit_group_names=["default"],
            )
        else:
            self.program = self.device.load_program("pathtracer.slang", ["compute_main"])
            self.pipeline = self.device.create_compute_pipeline(self.program)

    def execute(self, command_encoder: spy.CommandEncoder, output: spy.Texture, frame: int):
        w = output.width
        h = output.height

        self.scene.camera.width = w
        self.scene.camera.height = h
        self.scene.camera.recompute()

        if USE_RAYTRACING_PIPELINE:
            with command_encoder.begin_ray_tracing_pass() as pass_encoder:
                shader_object = pass_encoder.bind_pipeline(self.rt_pipeline, self.shader_table)
                cursor = spy.ShaderCursor(shader_object)
                cursor.g_output = output
                cursor.g_frame = frame
                self.scene.bind(cursor.g_scene)
                pass_encoder.dispatch_rays(0, [w, h, 1])
        else:
            with command_encoder.begin_compute_pass() as pass_encoder:
                shader_object = pass_encoder.bind_pipeline(self.pipeline)
                cursor = spy.ShaderCursor(shader_object)
                cursor.g_output = output
                cursor.g_frame = frame
                self.scene.bind(cursor.g_scene)
                pass_encoder.dispatch(thread_count=[w, h, 1])


class Accumulator:
    def __init__(self, device: spy.Device):
        super().__init__()
        self.device = device
        self.program = self.device.load_program("accumulator.slang", ["compute_main"])
        self.kernel = self.device.create_compute_kernel(self.program)
        self.accumulator: Optional[spy.Texture] = None

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        input: spy.Texture,
        output: spy.Texture,
        reset: bool = False,
    ):
        if (
            self.accumulator == None
            or self.accumulator.width != input.width
            or self.accumulator.height != input.height
        ):
            self.accumulator = self.device.create_texture(
                format=spy.Format.rgba32_float,
                width=input.width,
                height=input.height,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label="accumulator",
            )
        self.kernel.dispatch(
            thread_count=[input.width, input.height, 1],
            vars={
                "g_accumulator": {
                    "input": input,
                    "output": output,
                    "accumulator": self.accumulator,
                    "reset": reset,
                }
            },
            command_encoder=command_encoder,
        )


class ToneMapper:
    def __init__(self, device: spy.Device):
        super().__init__()
        self.device = device
        self.program = self.device.load_program("tone_mapper.slang", ["compute_main"])
        self.kernel = self.device.create_compute_kernel(self.program)

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        input: spy.Texture,
        output: spy.Texture,
    ):
        self.kernel.dispatch(
            thread_count=[input.width, input.height, 1],
            vars={
                "g_tone_mapper": {
                    "input": input,
                    "output": output,
                }
            },
            command_encoder=command_encoder,
        )


class App:
    def __init__(self):
        super().__init__()
        self.window = spy.Window(width=1920, height=1080, title="PathTracer", resizable=True)
        self.device = spy.Device(
            # type=spy.DeviceType.cuda,
            enable_debug_layers=False,
            compiler_options={
                "include_paths": [EXAMPLE_DIR],
                "defines": {"USE_RAYTRACING_PIPELINE": "1" if USE_RAYTRACING_PIPELINE else "0"},
            },
        )
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height, vsync=False)

        self.render_texture: spy.Texture = None  # type: ignore (will be set immediately)
        self.accum_texture: spy.Texture = None  # type: ignore (will be set immediately)
        self.output_texture: spy.Texture = None  # type: ignore (will be set immediately)

        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

        self.stage = Stage.demo()
        self.scene = Scene(self.device, self.stage)

        self.camera_controller = CameraController(self.stage.camera)

        self.path_tracer = PathTracer(self.device, self.scene)
        self.accumulator = Accumulator(self.device)
        self.tone_mapper = ToneMapper(self.device)

    def on_keyboard_event(self, event: spy.KeyboardEvent):
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

        self.camera_controller.on_keyboard_event(event)

    def on_mouse_event(self, event: spy.MouseEvent):
        self.camera_controller.on_mouse_event(event)

    def on_resize(self, width: int, height: int):
        self.device.wait()
        if width > 0 and height > 0:
            self.surface.configure(width=width, height=height, vsync=False)
        else:
            self.surface.unconfigure()

    def main_loop(self):
        frame = 0
        timer = spy.Timer()
        while not self.window.should_close():
            dt = timer.elapsed_s()
            timer.reset()

            self.window.process_events()

            if self.camera_controller.update(dt):
                frame = 0

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
                    format=spy.Format.rgba32_float,
                    width=surface_texture.width,
                    height=surface_texture.height,
                    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                    label="output_texture",
                )
                self.render_texture = self.device.create_texture(
                    format=spy.Format.rgba32_float,
                    width=surface_texture.width,
                    height=surface_texture.height,
                    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                    label="render_texture",
                )
                self.accum_texture = self.device.create_texture(
                    format=spy.Format.rgba32_float,
                    width=surface_texture.width,
                    height=surface_texture.height,
                    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                    label="accum_texture",
                )

            command_encoder = self.device.create_command_encoder()

            self.path_tracer.execute(command_encoder, self.render_texture, frame)
            self.accumulator.execute(
                command_encoder, self.render_texture, self.accum_texture, frame == 0
            )
            self.tone_mapper.execute(command_encoder, self.accum_texture, self.output_texture)

            command_encoder.blit(surface_texture, self.output_texture)
            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture

            self.surface.present()

            frame += 1

        self.device.wait()


app = App()
app.main_loop()
