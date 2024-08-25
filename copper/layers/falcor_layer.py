from __future__ import annotations

from .layer_base import (
    ComputeLayer,
    Program,
    ShaderCursor,
    Kernel,
    ReflectedType,
    BufferRef,
    Device,
    DeviceKind,
    TensorRef,
)
from .. import printing

from .. import tensor_layer
from ..types.base import Modifier, SlangName, ScalarKind, VoidType
from ..types.interfaces import SlangType
from ..types.arithmetic import ScalarType, ArrayType, VectorType
from ..types.struct import StructType, InterfaceType
from ..types.enum import EnumType
from ..types.resources import StructuredBufferType, RawBufferType
from ..types.func import SlangFunc, SlangFuncParam

import falcor
from falcor import ReflectionBasicType

import os
import math
import struct
import logging

from collections import OrderedDict
from pathlib import Path

from typing import Optional, Any

kind_to_falcor = {
    ScalarKind.Bool: falcor.DataType.int32,
    ScalarKind.Uint8: falcor.DataType.uint8,
    ScalarKind.Uint16: falcor.DataType.uint16,
    ScalarKind.Uint: falcor.DataType.uint32,
    ScalarKind.Uint64: falcor.DataType.uint64,
    ScalarKind.Int8: falcor.DataType.int8,
    ScalarKind.Int16: falcor.DataType.int16,
    ScalarKind.Int: falcor.DataType.int32,
    ScalarKind.Int64: falcor.DataType.int64,
    ScalarKind.Float16: falcor.DataType.float16,
    ScalarKind.Float: falcor.DataType.float32,
    ScalarKind.Float64: falcor.DataType.float64,
}
type_size = {
    ScalarKind.Bool: 4,
    ScalarKind.Uint8: 1,
    ScalarKind.Uint16: 2,
    ScalarKind.Uint: 4,
    ScalarKind.Uint64: 8,
    ScalarKind.Int8: 1,
    ScalarKind.Int16: 2,
    ScalarKind.Int: 4,
    ScalarKind.Int64: 8,
    ScalarKind.Float16: 2,
    ScalarKind.Float: 4,
    ScalarKind.Float64: 8,
}
scalar_types = {
    ReflectionBasicType.Type.Bool: ScalarKind.Bool,
    ReflectionBasicType.Type.Uint8: ScalarKind.Uint8,
    ReflectionBasicType.Type.Uint16: ScalarKind.Uint16,
    ReflectionBasicType.Type.Uint: ScalarKind.Uint,
    ReflectionBasicType.Type.Uint64: ScalarKind.Uint64,
    ReflectionBasicType.Type.Int8: ScalarKind.Int8,
    ReflectionBasicType.Type.Int16: ScalarKind.Int16,
    ReflectionBasicType.Type.Int: ScalarKind.Int,
    ReflectionBasicType.Type.Int64: ScalarKind.Int64,
    ReflectionBasicType.Type.Float16: ScalarKind.Float16,
    ReflectionBasicType.Type.Float: ScalarKind.Float,
    ReflectionBasicType.Type.Float64: ScalarKind.Float64,
}
vector_types = {
    ReflectionBasicType.Type.Bool2: (ReflectionBasicType.Type.Bool, 2),
    ReflectionBasicType.Type.Bool3: (ReflectionBasicType.Type.Bool, 3),
    ReflectionBasicType.Type.Bool4: (ReflectionBasicType.Type.Bool, 4),
    ReflectionBasicType.Type.Uint8_2: (ReflectionBasicType.Type.Uint8, 2),
    ReflectionBasicType.Type.Uint8_3: (ReflectionBasicType.Type.Uint8, 3),
    ReflectionBasicType.Type.Uint8_4: (ReflectionBasicType.Type.Uint8, 4),
    ReflectionBasicType.Type.Uint16_2: (ReflectionBasicType.Type.Uint16, 2),
    ReflectionBasicType.Type.Uint16_3: (ReflectionBasicType.Type.Uint16, 3),
    ReflectionBasicType.Type.Uint16_4: (ReflectionBasicType.Type.Uint16, 4),
    ReflectionBasicType.Type.Uint2: (ReflectionBasicType.Type.Uint, 2),
    ReflectionBasicType.Type.Uint3: (ReflectionBasicType.Type.Uint, 3),
    ReflectionBasicType.Type.Uint4: (ReflectionBasicType.Type.Uint, 4),
    ReflectionBasicType.Type.Uint64_2: (ReflectionBasicType.Type.Uint64, 2),
    ReflectionBasicType.Type.Uint64_3: (ReflectionBasicType.Type.Uint64, 3),
    ReflectionBasicType.Type.Uint64_4: (ReflectionBasicType.Type.Uint64, 4),
    ReflectionBasicType.Type.Int8_2: (ReflectionBasicType.Type.Int8, 2),
    ReflectionBasicType.Type.Int8_3: (ReflectionBasicType.Type.Int8, 3),
    ReflectionBasicType.Type.Int8_4: (ReflectionBasicType.Type.Int8, 4),
    ReflectionBasicType.Type.Int16_2: (ReflectionBasicType.Type.Int16, 2),
    ReflectionBasicType.Type.Int16_3: (ReflectionBasicType.Type.Int16, 3),
    ReflectionBasicType.Type.Int16_4: (ReflectionBasicType.Type.Int16, 4),
    ReflectionBasicType.Type.Int2: (ReflectionBasicType.Type.Int, 2),
    ReflectionBasicType.Type.Int3: (ReflectionBasicType.Type.Int, 3),
    ReflectionBasicType.Type.Int4: (ReflectionBasicType.Type.Int, 4),
    ReflectionBasicType.Type.Int64_2: (ReflectionBasicType.Type.Int64, 2),
    ReflectionBasicType.Type.Int64_3: (ReflectionBasicType.Type.Int64, 3),
    ReflectionBasicType.Type.Int64_4: (ReflectionBasicType.Type.Int64, 4),
    ReflectionBasicType.Type.Float16_2: (ReflectionBasicType.Type.Float16, 2),
    ReflectionBasicType.Type.Float16_3: (ReflectionBasicType.Type.Float16, 3),
    ReflectionBasicType.Type.Float16_4: (ReflectionBasicType.Type.Float16, 4),
    ReflectionBasicType.Type.Float2: (ReflectionBasicType.Type.Float, 2),
    ReflectionBasicType.Type.Float3: (ReflectionBasicType.Type.Float, 3),
    ReflectionBasicType.Type.Float4: (ReflectionBasicType.Type.Float, 4),
    ReflectionBasicType.Type.Float64_2: (ReflectionBasicType.Type.Float64, 2),
    ReflectionBasicType.Type.Float64_3: (ReflectionBasicType.Type.Float64, 3),
    ReflectionBasicType.Type.Float64_4: (ReflectionBasicType.Type.Float64, 4),
}


class FalcorLayer(ComputeLayer):
    def __init__(self, device: falcor.Device, print_buf_capacity: int = 16384):
        super().__init__()
        self.raw_device = device
        bind_flags = (
            falcor.ResourceBindFlags.UnorderedAccess
            | falcor.ResourceBindFlags.ShaderResource
        )
        self.print_counter = device.create_buffer(4, bind_flags)
        self.print_buffer = device.create_structured_buffer(
            8, print_buf_capacity, bind_flags
        )
        self.print_capacity = print_buf_capacity
        self.print_enabled = False

    def extend_program(
        self,
        prog: FalcorProgram,
        new_module_name: str,
        new_module_src: str,
        entry_point: Optional[str] = None,
    ) -> Program:
        raw_prog = prog.raw_prog

        base_path, imports = generate_all_imports(raw_prog)
        module_src = "\n".join(imports) + "\n\n" + new_module_src

        # wrapper_path = f'{base_path}/{name}.slang'
        ext = "slang" if entry_point is None else "cs.slang"
        wrapper_path = Path(new_module_name + "." + ext)
        module = falcor.ProgramDesc.ShaderModule(new_module_name).add_string(
            module_src, path=wrapper_path
        )

        desc = copy_desc_without_entrypoints(raw_prog.desc)
        desc.type_conformances = raw_prog.type_conformances
        desc.add_shader_modules([module])
        desc.compiler_flags |= falcor.SlangCompilerFlags.DumpIntermediates

        if entry_point is not None:
            desc.cs_entry(entry_point)

        return FalcorProgram(self.raw_device.create_program(desc, raw_prog.defines))

    def create_kernel(self, prog: FalcorProgram) -> Kernel:
        return FalcorKernel(self, falcor.ComputePass(self.raw_device, prog.raw_prog))

    def from_raw_program(self, raw_prog: Any) -> Optional[Program]:
        if not isinstance(raw_prog, falcor.Program):
            raise ValueError(
                f"Expected falcor.Program, received {type(raw_prog).__name__}"
            )

        return FalcorProgram(raw_prog)

    def device(self) -> Device:
        return FalcorDevice(self.raw_device)

    def enable_printing(self, value: bool = True):
        self.print_enabled = value


class FalcorShaderCursor(ShaderCursor):
    def __init__(
        self, kernel: FalcorKernel, var: falcor.ShaderVar, path: tuple[str | int, ...]
    ):
        super().__init__()
        self.kernel = kernel
        self.var = var
        self.path = path

    def path_str(self) -> str:
        out = str(self.path[0])
        for i in self.path[1:]:
            if isinstance(i, int):
                out += f"[{i}]"
            else:
                out += "." + str(i)
        return out

    def set_tensor_data(self, tensor: TensorRef):
        buf = tensor.buffer()
        assert isinstance(buf, FalcorBufferRef)

        layout = self["layout"]
        D = layout.var["shape"].type.as_array_type.element_count
        pad = D - len(tensor.get_shape())

        shape = (1,) * pad + tensor.get_shape()
        strides = (0,) * pad + tensor.get_strides()
        offset = tensor.get_offset()
        broadcast = any(d == 0 for d in tensor.get_strides())

        layout["shape"] = shape
        layout["strides"] = strides
        layout["offset"] = offset
        layout["is_broadcast"] = broadcast
        self.var["data"] = buf.raw_buffer

        logging.debug(
            f"    Setting tensor data {{shape: {shape}, strides: {strides}, "
            f"offset: {offset}, is_broadcast: {broadcast}, data: {buf.raw_buffer}}}"
        )

    def set_tensor(self, tensor: TensorRef, readable: bool, writeable: bool):
        tensor.check_valid()

        logging.debug(
            f"ShaderVar {self.path_str()} = tensor.{tensor.get_dtype().to_slang_type()}{list(tensor.get_shape())}"
        )

        # If this tensor is already backed by memory in the graphics pipeline, we can reuse that buffer directly
        buf = tensor.buffer()
        if isinstance(buf, FalcorBufferRef):
            logging.debug(f"    Tensor already in DX buffer {buf.raw_buffer}")
            self.set_tensor_data(tensor)
            return

        # This tensor is backed by memory elsewhere. We need to allocate a falcor buffer, copy input data in (if
        # the buffer is readable) and copy output data out (if the buffer is writeable). We create the buffer here
        # and map it to a tensor native in whichever framework, and do the copying right before/after the kernel
        # call.
        #
        # A tensor can be a 1:1 view, a smaller view (via e.g. slicing) or a larger view (via e.g. expanding)
        # of the underlying storage. We take the smaller of the actual storage size and the viewable size to be
        # the number of elements that need to exist in the buffer
        dtype = tensor.get_dtype()
        shape = tensor.get_shape()
        numel = math.prod(shape)
        real_numel = 1 + sum(
            (dim - 1) * stride
            for dim, stride in zip(tensor.get_shape(), tensor.get_strides())
        )
        required_count = min(numel, real_numel)
        element_size = type_size[dtype]
        required_size = required_count * element_size

        cache_key = self.path_str()
        buffer = self.kernel.buffer_cache.get(cache_key)
        if buffer is None:
            logging.debug(f"    Buffer cache lookup failed for key {cache_key}")
        else:
            logging.debug(f"    Found buffer {buffer} in cache for key {cache_key}")

        if buffer is None or buffer.raw_buffer.size < required_size:
            flags = (
                falcor.ResourceBindFlags.Shared
                | falcor.ResourceBindFlags.ShaderResource
                | falcor.ResourceBindFlags.UnorderedAccess
            )
            raw_buffer = self.kernel.layer.raw_device.create_buffer(
                required_size, flags
            )
            buffer = FalcorBufferRef(raw_buffer, cache_key)
            self.kernel.buffer_cache[cache_key] = buffer
            logging.debug(
                f"    Created new buffer of size {required_size} ({raw_buffer})"
            )

        strides = tensor.get_strides()
        if numel < real_numel:
            strides = [1]
            for dim in reversed(tensor.get_shape()[1:]):
                strides = [dim * strides[0]] + strides
            logging.debug(
                f"    Recomputing strides for shape {tensor.get_shape()}: {tensor.get_strides()} -> {strides}"
            )
        mapped_tensor = tensor_layer().import_tensor(
            buffer.raw_buffer.to_dlpack(
                kind_to_falcor[dtype], list(shape), list(strides), 0
            ),
            buffer,
        )

        self.set_tensor_data(mapped_tensor)
        if readable:
            src_tensor, dst_tensor = tensor, mapped_tensor
            if real_numel < numel:
                src_tensor = tensor.create_view(0, (real_numel,), (1,))
                dst_tensor = mapped_tensor.create_view(0, (real_numel,), (1,))

            self.kernel.add_input_tensor(src_tensor, dst_tensor)
        if writeable:
            self.kernel.add_output_tensor(mapped_tensor, tensor)

    def set(self, value: Any):
        # Falcor is picky when setting vector valued variables. Be nice here and convert e.g. [1, 2, 3] to falcor.float3
        expected = self.var.expected_type
        if expected in vector_types and isinstance(value, list):
            type, dim = vector_types[expected]
            if dim != len(value):
                raise ValueError(
                    f"Can't set vector variable: Expected a list with {dim} elements, received {len(value)}"
                )

            mapping = {
                ReflectionBasicType.Type.Bool: [
                    falcor.bool2,
                    falcor.bool3,
                    falcor.bool4,
                ],
                ReflectionBasicType.Type.Int: [
                    falcor.int2,
                    falcor.int3,
                    falcor.int4,
                ],
                ReflectionBasicType.Type.Uint: [
                    falcor.uint2,
                    falcor.uint3,
                    falcor.uint4,
                ],
                ReflectionBasicType.Type.Float: [
                    falcor.float2,
                    falcor.float3,
                    falcor.float4,
                ],
                ReflectionBasicType.Type.Float16: [
                    falcor.float16_t2,
                    falcor.float16_t3,
                    falcor.float16_t4,
                ],
            }
            if type in mapping and dim >= 2 and dim <= 4:
                value = mapping[type][dim](*value)
        elif isinstance(value, FalcorBufferRef):
            value = value.raw_buffer
        elif isinstance(value, list) or isinstance(value, tuple):
            for i, v in enumerate(value):
                self[i].set(v)
            return

        logging.debug(f"ShaderVar {self.path_str()} = {value}")

        self.var.set(value)  # type: ignore

    def __getitem__(self, idx: int | str) -> FalcorShaderCursor:
        return FalcorShaderCursor(self.kernel, self.var[idx], self.path + (idx,))


class FalcorProgram(Program):
    def __init__(self, raw_prog: falcor.Program):
        super().__init__()
        self.raw_prog = raw_prog
        self.refl = self.raw_prog.reflector

    def find_type(self, name: str) -> Optional[ReflectedType]:
        refl_type = self.refl.find_type(name)
        if refl_type is None:
            return None
        return FalcorReflectedType(refl_type, self.refl)

    def find_function(self, name: str) -> Optional[SlangFunc]:
        # TODO: Should do proper overload resolution at some point. Just grab first overload for now
        func = self.refl.find_function(name, 0)
        if func is None:
            return None
        return reflect_function(func, self.refl)


class FalcorReflectedType(ReflectedType):
    def __init__(
        self,
        refl_type: falcor.ReflectionType,
        reflector: falcor.ProgramReflection,
        mapped_type: Optional[SlangType] = None,
    ):
        super().__init__()
        self.refl_type = refl_type
        self.refl = reflector

        if mapped_type is None and refl_type is not None:
            mapped_type = reflect_type(refl_type, reflector)
        self.mapped_type = mapped_type

    def valid(self) -> bool:
        return self.refl_type is not None

    def type(self) -> Optional[SlangType]:
        return self.mapped_type

    def methods(self) -> list[SlangFunc]:
        if not isinstance(self.mapped_type, StructType):
            return []

        return [
            reflect_function(f, self.refl)
            for f in self.refl.get_methods(self.refl_type)
        ]

    def generic_args(self) -> list[SlangType | int]:
        if not isinstance(self.mapped_type, StructType):
            return []

        # This really should be solved by walking the DeclRef tree, but we're not there yet.
        # Instead we do string processing on a stringified slang type. Not great, but will
        # work for now until we have the correct solution.
        decl_string = self.mapped_type.name.specialized
        if not decl_string.endswith(">"):
            return []

        open_pos = decl_string.find("<")
        if open_pos == -1:  # ???
            return []

        arg_str = decl_string[open_pos + 1 : -1]

        args = []
        cur = ""
        level = 0
        for c in arg_str:
            if c in ["(", "<"]:
                level += 1
            elif c in [")", ">"]:
                level += 1
            elif c == "," and level == 0:
                args.append(cur)
                cur = ""
            else:
                cur += c
        if cur:
            args.append(cur)

        result = []
        for arg in args:
            resolved_arg = None
            try:
                resolved_arg = int(arg)
            except ValueError:
                # TODO: This could fail, as the generic argument could live in some other scope
                refl_type = self.refl.find_type(arg)
                assert refl_type is not None
                resolved_arg = reflect_type(refl_type, self.refl)

            result.append(resolved_arg)

        return result


class FalcorKernel(Kernel):
    def __init__(self, layer: FalcorLayer, cpass: falcor.ComputePass):
        super().__init__()
        self.layer = layer
        self.cpass = cpass
        self.pending_input_tensors: list[tuple[TensorRef, TensorRef]] = []
        self.pending_output_tensors: list[tuple[TensorRef, TensorRef]] = []
        self.buffer_cache: dict[str, FalcorBufferRef] = {}

    def add_input_tensor(self, src_tensor: TensorRef, dst_tensor: TensorRef):
        self.pending_input_tensors.append((src_tensor, dst_tensor))

    def add_output_tensor(self, src_tensor: TensorRef, dst_tensor: TensorRef):
        self.pending_output_tensors.append((src_tensor, dst_tensor))

    def rootvar(self) -> FalcorShaderCursor:
        return FalcorShaderCursor(self, self.cpass.root_var, ())

    def dispatch(self, *in_dims: int):
        if len(in_dims) > 3:
            raise ValueError("Too many dispatch dimensions; at most 3 are allowed")
        dims = list(in_dims) + [1] * (3 - len(in_dims))

        tensor_device = tensor_layer().device()
        if self.pending_input_tensors:
            for src_tensor, dst_tensor in self.pending_input_tensors:
                dst_tensor.copy_data(src_tensor)

                buf = dst_tensor.buffer()
                assert isinstance(buf, FalcorBufferRef)
                buf.bump_version()

            if tensor_device.kind == DeviceKind.Cuda:
                self.layer.raw_device.render_context.wait_for_cuda(
                    tensor_device.stream()
                )
            else:
                tensor_device.sync()

        if self.layer.print_enabled:
            self.cpass.root_var["gPrintData"]["printIndex"] = self.layer.print_counter
            self.cpass.root_var["gPrintData"]["printBuffer"] = self.layer.print_buffer
            self.cpass.root_var["gPrintData"][
                "printBufferCapacity"
            ] = self.layer.print_capacity
            self.layer.raw_device.render_context.clear_buffer(self.layer.print_counter)
        else:
            self.cpass.root_var["gPrintData"]["printBufferCapacity"] = 0  # TODO

        self.cpass.execute(dims[0], dims[1], dims[2])

        if self.layer.print_enabled:
            count = struct.unpack_from("<I", self.layer.print_counter.to_bytes())[0]
            buf = self.layer.print_buffer.to_bytes(
                min(count, self.layer.print_capacity) * 8
            )
            hashed_strings = {}
            for hs in self.cpass.program.reflector.hashed_strings:
                hashed_strings[hs.hash] = hs.string
            printing.print_slang_buffer(buf, hashed_strings)

        if tensor_device.kind == DeviceKind.Cuda:
            self.layer.raw_device.render_context.wait_for_falcor(tensor_device.stream())
        else:
            self.layer.raw_device.wait()

        if self.pending_output_tensors:
            for src_tensor, dst_tensor in self.pending_output_tensors:
                buf = src_tensor.buffer()
                assert isinstance(buf, FalcorBufferRef)
                buf.bump_version()

                if dst_tensor.buffer() is None:
                    dst_tensor.point_to(src_tensor)
                else:
                    dst_tensor.copy_data(src_tensor)

        self.pending_input_tensors = []
        self.pending_output_tensors = []


class FalcorDevice(Device):
    def __init__(self, device: falcor.Device):
        super().__init__()
        self.device = device

    def kind(self) -> DeviceKind:
        return DeviceKind.Cuda

    def idx(self) -> int:
        return self.device.desc.gpu

    def sync(self):
        self.device.wait()

    def stream(self):
        return 0


class FalcorBufferRef(BufferRef):
    def __init__(self, raw_buffer: falcor.Buffer, tag: str):
        super().__init__(tag)

        self.raw_buffer = raw_buffer

    def data_ptr(self):
        self.raw_buffer.cuda_ptr

    def device(self) -> Device:
        return FalcorDevice(self.raw_buffer.device)

    def bump_version(self):
        self.version += 1

    def __repr__(self):
        return f"FalcorBufferRef(version={self.version}, tag={self.tag}, raw_buffer={repr(self.raw_buffer)})"


def reflect_function(
    func: falcor.ReflectionFunction, reflector: falcor.ProgramReflection
) -> SlangFunc:
    func_modifiers = Modifier.Nothing
    if func.is_no_diff:
        func_modifiers |= Modifier.NoDiff
    if func.is_no_diff_this:
        func_modifiers |= Modifier.NoDiffThis
    if func.is_forward_differentiable:
        func_modifiers |= Modifier.ForwardDifferentiable
    if func.is_backward_differentiable:
        func_modifiers |= Modifier.BackwardDifferentiable
    if func.is_mutating:
        func_modifiers |= Modifier.Mutating

    return_type = reflect_type(func.return_type, reflector)
    params = [reflect_funcparam(arg, reflector) for arg in func.parameters]

    return SlangFunc(func.name, params, return_type, func_modifiers)


def reflect_funcparam(
    param: falcor.ReflectionParameter, reflector: falcor.ProgramReflection
) -> SlangFuncParam:
    modifiers = Modifier.Nothing
    if param.has_modifier(falcor.ReflectionParameter.Modifier.NoDiff):
        modifiers |= Modifier.NoDiff
    has_in = param.has_modifier(falcor.ReflectionParameter.Modifier.In)
    has_out = param.has_modifier(falcor.ReflectionParameter.Modifier.Out)
    has_inout = param.has_modifier(falcor.ReflectionParameter.Modifier.InOut)

    if has_inout or (has_in and has_out):
        modifiers |= Modifier.InOut
    elif has_out:
        modifiers |= Modifier.Out
    else:
        modifiers |= Modifier.In

    return SlangFuncParam(
        param.name, reflect_type(param.type, reflector), param.has_default, modifiers, 0
    )


def reflect_type(
    type: falcor.ReflectionType, reflector: falcor.ProgramReflection
) -> SlangType:
    assert isinstance(type, falcor.ReflectionType)

    if type.kind == falcor.ReflectionType.Kind.Interface:
        return reflect_interface(type.as_interface_type, reflector)
    elif type.kind == falcor.ReflectionType.Kind.Struct:
        return reflect_struct(type.as_struct_type, reflector)
    elif type.kind == falcor.ReflectionType.Kind.Enum:
        return reflect_enum(type.as_enum_type, reflector)
    elif type.kind == falcor.ReflectionType.Kind.Resource:
        return reflect_resource(type.as_resource_type, reflector)
    elif type.kind == falcor.ReflectionType.Kind.Array:
        return reflect_array(type.as_array_type, reflector)
    elif type.kind == falcor.ReflectionType.Kind.Basic:
        return reflect_basic(type.as_basic_type, reflector)

    raise ValueError(f"Unsupported type '{type.kind}' (in scope {type})")


def reflect_basic(
    type: ReflectionBasicType, reflector: falcor.ProgramReflection
) -> SlangType:
    kind = type.type

    if kind == ReflectionBasicType.Type.Void:
        return VoidType()
    elif kind in scalar_types:
        return ScalarType(scalar_types[kind])
    elif kind in vector_types:
        subtype, count = vector_types[kind]
        return VectorType(ScalarType(scalar_types[subtype]), count)
    else:
        raise ValueError(f"Unsupported basic type '{kind}'")


def reflect_array(
    type: falcor.ReflectionArrayType, reflector: falcor.ProgramReflection
) -> ArrayType:
    dtype = reflect_type(type.element_type, reflector)

    return ArrayType(dtype, type.element_count)


def reflect_resource(
    resource: falcor.ReflectionResourceType, reflector: falcor.ProgramReflection
):
    writeable = (
        resource.shader_access == falcor.ReflectionResourceType.ShaderAccess.ReadWrite
    )

    if resource.type == falcor.ReflectionResourceType.Type.StructuredBuffer:
        return StructuredBufferType(
            reflect_type(resource.struct_type, reflector), writeable
        )
    elif resource.type == falcor.ReflectionResourceType.Type.RawBuffer:
        return RawBufferType(writeable)
    elif resource.type == falcor.ReflectionResourceType.Type.Texture:
        # return TextureType(resource.dimensions, writeable)
        raise RuntimeError("Texture types are currently unsupported")
    else:
        raise ValueError(f"Unsupported resource type '{resource}'")


def reflect_interface(
    type: falcor.ReflectionInterfaceType, reflector: falcor.ProgramReflection
) -> InterfaceType:
    return InterfaceType()


def reflect_name(
    type: falcor.ReflectionStructType | falcor.ReflectionEnumType,
) -> SlangName:
    return SlangName(
        type.name,
        type.to_string(include_scopes=False),
        type.to_string(include_scopes=True),
    )


def reflect_enum(
    type: falcor.ReflectionEnumType, reflector: falcor.ProgramReflection
) -> EnumType:
    dtype = reflect_type(type.value_type, reflector)

    return EnumType(reflect_name(type), dtype, type.is_flags, type.cases)


def reflect_struct(
    type: falcor.ReflectionStructType, reflector: falcor.ProgramReflection
) -> StructType:
    name = reflect_name(type)

    members = OrderedDict()
    for i in range(len(type)):
        members[type[i].name] = reflect_type(type[i].type, reflector)

    result = StructType(name, members)

    differential = reflector.find_type(f"{name.declared}.Differential")
    if differential is None:
        result.differential = None
    elif differential.as_struct_type is type:
        result.differential = result
    else:
        refl_differential = reflect_type(differential, reflector)
        assert isinstance(refl_differential, StructType)
        result.differential = refl_differential

    if name.base in SlangType.opaque_types:
        return SlangType.opaque_types[type.name].from_reflection(
            name, FalcorReflectedType(type, reflector, result)
        )

    return result


def copy_desc_without_entrypoints(desc: falcor.ProgramDesc) -> falcor.ProgramDesc:
    result = falcor.ProgramDesc()

    result.shader_model = desc.shader_model
    result.compiler_flags = desc.compiler_flags
    result.compiler_arguments = desc.compiler_arguments

    for m in desc.modules:
        sources = []
        for s in m.sources:
            if s.type == falcor.ProgramDesc.ShaderSource.Type.String:
                sources.append(s)
        if sources:
            new_m = result.add_shader_module(m.name)
            for s in sources:
                new_m.add_string(s.string, s.path)

    return result


def generate_all_imports(prog: falcor.Program) -> tuple[str, list[str]]:
    paths = []
    for m in prog.desc.modules:
        for s in m.sources:
            if s.path:
                paths.append(os.path.normpath(s.path))

    if not paths:
        return "", []

    # base_path = os.path.commonpath(paths)
    base_path = os.path.split(paths[-1])[0]

    module_names = [
        p.removesuffix(".cs.slang")
        .removesuffix(".slang")
        .replace("-", "_")
        .replace(os.sep, ".")
        for p in paths
    ]
    imports = [f"__exported import {m};" for m in module_names]

    return base_path, imports
