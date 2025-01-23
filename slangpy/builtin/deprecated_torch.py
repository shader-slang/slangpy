# SPDX-License-Identifier: Apache-2.0
# type: ignore

from __future__ import annotations

try:
    import torch  # @IgnoreException
except ImportError:
    torch = None

import numpy as np

from slangpy.backend import TypeReflection, SlangModule, Buffer, ResourceUsage, DataType
from slangpy.core.native import NativeBoundVariableRuntime
from slangpy.bindings import ReturnContext, BoundVariableRuntime, CallContext, Shape, AccessType
from slangpy.bindings import PYTHON_TYPES, PYTHON_SIGNATURES
from slangpy.reflection import SlangType, ScalarType, SlangProgramLayout
from slangpy.reflection import reflectiontypes
from slangpy.builtin.tensor import TensorMarshall, ITensorType, is_nested_array
from slangpy.core.module import Module
from slangpy.core.function import Function
from slangpy.core.utils import shape_to_contiguous_strides
from slangpy.types import ValueRef, Tensor
from slangpy.types.tensor import innermost_type

from typing import Any, Optional, cast


TPath = tuple[str | int, ...]


class TorchState:
    def __init__(self):
        super().__init__()

        self._active_call = None

    def before_call(self, func: Function):
        assert self._active_call is None, "Attempted to activate a new torch call context while one is already active"

        self._active_call = TorchCallContext(func)

    def after_call(self, func: Function):
        assert self._active_call is not None
        assert self._active_call.func is func
        self._active_call = None

    def before_write_calldata(self, context: CallContext, unpacked_args: tuple[Any, ...], unpacked_kwargs: dict[str, Any]):
        self.active_call.pre_write_calldata(context, unpacked_args, unpacked_kwargs)

    def before_dispatch(self, vars: dict[str, Any]):
        self.active_call.pre_dispatch(vars)

    def after_dispatch(self, vars: dict[str, Any]):
        self.active_call.post_dispatch(vars)

    def after_read_calldata(self, context: CallContext, unpacked_args: tuple[Any, ...], unpacked_kwargs: dict[str, Any]):
        self.active_call.post_read_calldata(context, unpacked_args, unpacked_kwargs)

    @property
    def active_call(self) -> TorchCallContext:
        if self._active_call is None:
            raise RuntimeError("Failed to access current torch context because none is active. "
                               "The most likely reason is attempting to pass a torch.Tensor to a bare slangpy Function. "
                               "Passing torch tensors is only supported for functions retrieved from a TorchModule.")
        return self._active_call


_TORCH_STATE = TorchState()


class TorchModule(Module):
    def __init__(self, module: SlangModule | Module):
        if torch is None:
            raise RuntimeError("Torch support is not available because torch is not installed")

        if isinstance(module, Module):
            device_module = module.device_module
        else:
            device_module = module

        super().__init__(device_module)

        if not device_module.session.device.supports_cuda_interop:
            raise RuntimeError("Cuda interop must be enabled for torch support "
                               "create SGL device with Device..., enable_cuda_interop=True")

    def __getattr__(self, name: str):
        result = super().__getattr__(name)

        if isinstance(result, Function):
            result = result \
                ._internal_hook(before_call=_TORCH_STATE.before_call, after_call=_TORCH_STATE.after_call,
                                before_dispatch=_TORCH_STATE.before_dispatch, after_dispatch=_TORCH_STATE.after_dispatch,
                                before_write_call_data=_TORCH_STATE.before_write_calldata, after_read_call_data=_TORCH_STATE.after_read_calldata) \
                .return_type(torch.Tensor)

        return result


if torch is not None:
    def filter_tensors(element: Any):
        if torch is None:
            raise RuntimeError("Torch support is not available because torch is not installed")

        if isinstance(element, dict):
            return {k: filter_tensors(v) for k, v in element.items()}
        elif isinstance(element, (list, tuple)):
            return [filter_tensors(v) for v in element]
        elif isinstance(element, (torch.Tensor, WrappedTensor)):
            return None
        else:
            return element

    ST = TypeReflection.ScalarType
    if torch is not None:
        _torch_to_scalar_type = {
            torch.int8: ST.int8, torch.int32: ST.int16, torch.int32: ST.int32, torch.int64: ST.int64,
            torch.uint8: ST.uint8,
            torch.float16: ST.float16, torch.float32: ST.float32, torch.float64: ST.float64
        }
        _torch_to_data_type = {
            torch.int8: DataType.int8, torch.int32: DataType.int16, torch.int32: DataType.int32, torch.int64: DataType.int64,
            torch.uint8: DataType.uint8,
            torch.float16: DataType.float16, torch.float32: DataType.float32, torch.float64: DataType.float64
        }
        _scalar_type_to_torch = {
            y: x for x, y in _torch_to_scalar_type.items()
        }
    else:
        _torch_to_scalar_type = {}
        _torch_to_data_type = {}
        _scalar_type_to_torch = {}

    def _slang_dtype_to_torch(slang_dtype: SlangType) -> Optional['torch.dtype']:
        if isinstance(slang_dtype, ScalarType):
            return _scalar_type_to_torch.get(slang_dtype.slang_scalar_type)
        return None

    def _torch_dtype_to_slang(torch_dtype: 'torch.dtype', layout: SlangProgramLayout) -> Optional[SlangType]:
        scalar_type = _torch_to_scalar_type.get(torch_dtype)
        if scalar_type is None:
            return None
        return layout.find_type_by_name(reflectiontypes.scalar_names[scalar_type])

    class WrappedTensor:
        def __init__(self, primal: 'torch.Tensor'):
            super().__init__()

            self.primal = primal
            self.grad_in: Optional[WrappedTensor] = None
            self.grad_out: Optional[WrappedTensor] = None

        def collect_streams(self, streams: set, include_meta: bool):
            if self.primal.is_cuda or (self.primal.is_meta and include_meta):
                device = self.primal.device if self.primal.is_cuda else None
                stream = torch.cuda.current_stream(device).cuda_stream
                streams.add(stream)
            if self.grad_in:
                self.grad_in.collect_streams(streams, include_meta)
            if self.grad_out:
                self.grad_out.collect_streams(streams, include_meta)

    class TensorWithContext:
        def __init__(self, tensor: WrappedTensor, path: TPath, sgl_tensor: Tensor, buffer: Buffer, readable: bool, writable: bool):
            super().__init__()

            self.tensor = tensor
            self.path = path
            self.sgl_tensor = sgl_tensor
            self.buffer = buffer
            self.readable = readable
            self.writable = writable

    def wrap_tensor(data: 'torch.Tensor' | WrappedTensor) -> WrappedTensor:
        if isinstance(data, torch.Tensor):
            return WrappedTensor(data)
        return data

    class TorchCallContext:
        def __init__(self, func: Function) -> None:
            super().__init__()
            self.device = func.module.device
            self.func = func
            self.tensor_to_path: dict[int, TPath]
            self.known_tensors: list[Optional[TensorWithContext]]
            self.marshall_to_index: dict[TorchTensorMarshall, int]

        def pre_write_calldata(self, context: CallContext, unpacked_args: tuple[Any, ...], unpacked_kwargs: dict[str, Any]):
            self.tensor_to_path = {}
            self.known_tensors = []
            self.marshall_to_index = {}

            self.record_tensor_args(unpacked_args, ())
            self.record_tensor_args(unpacked_kwargs, ())

        def record_tensor_args(self, element: Any, path: TPath):
            if isinstance(element, dict):
                for k, v in element.items():
                    self.record_tensor_args(v, path + (k, ))
            elif isinstance(element, (list, tuple)):
                for i, v in enumerate(element):
                    self.record_tensor_args(v, path + (i, ))
            elif isinstance(element, torch.Tensor):
                self.tensor_to_path[id(element)] = path
            elif isinstance(element, WrappedTensor):
                self.tensor_to_path[id(element.primal)] = path
                if element.grad_in is not None:
                    self.record_tensor_args(element.grad_in,  path + ('grad_in', ))
                if element.grad_out is not None:
                    self.record_tensor_args(element.grad_out, path + ('grad_out', ))

        def pre_dispatch(self, vars: dict[str, Any]):
            input_streams = set()
            for t in self.known_tensors:
                assert t is not None
                if t.readable:
                    t.tensor.collect_streams(input_streams, False)

            for stream in input_streams:
                self.device.sync_to_cuda(stream)

        def post_dispatch(self, vars: dict[str, Any]):
            output_streams = set()
            for t in self.known_tensors:
                assert t is not None
                if t.writable:
                    t.tensor.collect_streams(output_streams, True)

            for stream in output_streams:
                self.device.sync_to_device(stream)

        def post_read_calldata(self, context: CallContext, unpacked_args: tuple[Any, ...], unpacked_kwargs: dict[str, Any]):
            path_to_context = {t.path: t for t in self.known_tensors if t is not None}

            saved_tensors: list[tuple[TPath, torch.Tensor]] = []
            diff_input_tensors: list[TPath] = []
            diff_output_tensors: list[TPath] = []
            torch_func_inputs: list[torch.Tensor] = []

            for path in self.tensor_to_path.values():
                assert path in path_to_context
                t = path_to_context[path]

                saved_tensors.append((path, t.tensor.primal))

                if t.readable and t.sgl_tensor.dtype.differentiable:
                    diff_input_tensors.append(path)
                    # TODO: Should we sometimes use the original tensor here? Unclear because of contiguous()
                    torch_func_inputs.append(t.tensor.primal)
                if t.writable and t.sgl_tensor.dtype.differentiable:
                    diff_output_tensors.append(path)

            filtered_args = filter_tensors(unpacked_args)
            filtered_kwargs = filter_tensors(unpacked_kwargs)

            KfFunction.apply(self.func, filtered_args, filtered_kwargs, saved_tensors,
                             diff_input_tensors, diff_output_tensors, *torch_func_inputs)

            self.tensor_to_path = {}
            self.known_tensors = []
            self.marshall_to_index = {}

        def convert_tensor_to_sgl(self, marshall: TorchTensorMarshall, data: WrappedTensor, readable: bool, writable: bool):
            tensor = data.primal
            assert tensor.is_cuda or tensor.is_cpu or tensor.is_meta, f"Unsupported torch device: {tensor.device}"

            assert id(tensor) in self.tensor_to_path, \
                f"Don't have tensor with id {id(tensor)} in paths {self.tensor_to_path}"
            path = self.tensor_to_path[id(tensor)]

            shape = tuple(tensor.shape)
            strides = tuple(tensor.stride())
            assert all(stride >= 0 for stride in strides)

            indexable_span = 1 + sum((dim - 1) * stride for dim,
                                     stride in zip(shape, strides))
            contiguous_size = tensor.numel()

            if contiguous_size < indexable_span:
                strides = shape_to_contiguous_strides(shape)
                # TODO: Write CUDA kernel for simultaneous contiguous + copy
                t = tensor.contiguous()
            else:
                t = tensor

            view_length = min(contiguous_size, indexable_span)
            view_size = view_length * t.element_size()
            offset = t.storage_offset() * t.element_size()

            usage = ResourceUsage.shared | ResourceUsage.shader_resource
            if writable:
                usage |= ResourceUsage.unordered_access
            buf = self.device.create_buffer(view_size, usage=usage)

            if readable:
                if t.is_cuda:
                    buffer_view = cast(torch.Tensor, buf.to_torch(
                        DataType.uint8, (view_size, ), (1, )))
                    buffer_view.untyped_storage().copy_(t.untyped_storage(), non_blocking=False)
                elif t.is_cpu:
                    flattened = np.lib.stride_tricks.as_strided(
                        t.numpy(), (view_length, ), (t.element_size(), ))
                    buf.from_numpy(flattened)
                else:
                    raise ValueError(
                        f"Don't know how to read input data from torch tensor with device {t.device}")

            sgl_tensor = Tensor(buf, marshall.element_type, shape, strides, offset)

            if marshall.d_in is not None and data.grad_in is not None:
                sgl_tensor.grad_in = self.convert_tensor_to_sgl(
                    marshall.d_in, data.grad_in, True, False)
            if marshall.d_out is not None and data.grad_out is not None:
                sgl_tensor.grad_out = self.convert_tensor_to_sgl(
                    marshall.d_out, data.grad_out, True, True)

            self.marshall_to_index[marshall] = len(self.known_tensors)
            self.known_tensors.append(TensorWithContext(
                data, path, sgl_tensor, buf, readable, writable))

            return sgl_tensor

        def read_calldata(self, marshall: TorchTensorMarshall):
            idx = self.marshall_to_index.get(marshall)
            assert idx is not None
            t = self.known_tensors[idx]
            assert t is not None

            if marshall.d_out is not None and t.sgl_tensor.grad_out is not None:
                self.read_calldata(marshall.d_out)

            if not t.writable:
                return

            data_type = _torch_to_data_type[marshall.torch_dtype]

            result = cast(torch.Tensor, t.buffer.to_torch(
                data_type, t.sgl_tensor.shape, t.sgl_tensor.strides))

            if t.tensor.primal.is_meta:
                t.tensor.primal = result
            else:
                t.tensor.primal.copy_(result)

        # TODO: Only works together with create_output. The problem is that torch.Tensor data is rewrapping on every call
        def read_output(self, marshall: TorchTensorMarshall, data: WrappedTensor):
            return data.primal

    class KfFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            slangpy_func: Function,
            filtered_args: list[Any],
            filtered_kwargs: dict[str, Any],
            saved_tensors: list[tuple[TPath, torch.Tensor]],
            diff_input_tensors: list[TPath],
            diff_output_tensors: list[TPath],
            *inputs: torch.Tensor,
        ):
            ctx.slangpy_func = slangpy_func
            ctx.filtered_args = filtered_args
            ctx.filtered_kwargs = filtered_kwargs
            ctx.saved_tensor_paths = tuple(p for p, _ in saved_tensors)
            ctx.diff_input_tensors = diff_input_tensors
            ctx.diff_output_tensors = diff_output_tensors
            ctx.set_materialize_grads(False)

            ctx.save_for_backward(*(t for _, t in saved_tensors))

            return tuple(t for p, t in saved_tensors if p in diff_output_tensors)

        @staticmethod
        def backward(ctx: Any, *output_grads: torch.Tensor):
            filtered_args = cast(list[Any], filter_tensors(ctx.filtered_args))
            filtered_kwargs = cast(dict[str, Any], filter_tensors(ctx.filtered_kwargs))

            slangpy_func: Function = ctx.slangpy_func

            diff_input_tensors = {p: i for i, p in enumerate(ctx.diff_input_tensors)}
            diff_output_tensors = {p: i for i, p in enumerate(ctx.diff_output_tensors)}

            tensor_offset = 6
            needs_input_grad = ctx.needs_input_grad[tensor_offset:]

            input_grads: list[Optional[WrappedTensor]] = [None] * len(ctx.diff_input_tensors)

            for path, t in zip(ctx.saved_tensor_paths, ctx.saved_tensors):
                arg = WrappedTensor(t)
                if path in diff_output_tensors:
                    arg.grad_in = wrap_tensor(output_grads[diff_output_tensors[path]])
                if path in diff_input_tensors and needs_input_grad[diff_input_tensors[path]]:
                    # arg.grad_out = wrap_tensor(torch.empty(t.shape, dtype=t.dtype, device=torch.device('meta')))
                    # arg.grad_out = wrap_tensor(torch.ones(t.shape, dtype=t.dtype, device=t.device))
                    arg.grad_out = wrap_tensor(torch.zeros(t.shape, dtype=t.dtype, device=t.device))
                    input_grads[diff_input_tensors[path]] = arg.grad_out

                target = filtered_args if isinstance(path[0], int) else filtered_kwargs
                for step in path[:-1]:
                    target = target[step]
                target[path[-1]] = arg

            assert all(t is not None or not needs_grad for t,
                       needs_grad in zip(input_grads, needs_input_grad))

            slangpy_func.bwds(*filtered_args, **filtered_kwargs)

            result: list[Optional[torch.Tensor]] = [None] * tensor_offset
            for t, needs_grad in zip(input_grads, needs_input_grad):
                if needs_grad:
                    assert t is not None
                    result.append(t.primal)
                else:
                    result.append(None)

            return tuple(result)

    # TODO: Handle slang types that aren't bare float / etc.

    class TorchTensorMarshall(TensorMarshall):
        def __init__(self, layout: SlangProgramLayout, slang_dtype: SlangType, dims: int, d_in: Optional[TorchTensorMarshall], d_out: Optional[TorchTensorMarshall]):
            dtype = innermost_type(slang_dtype)
            if not is_nested_array(slang_dtype) or not isinstance(dtype, ScalarType) or len(slang_dtype.shape) > 2:
                raise ValueError(
                    f"Torch tensors do not support data type {slang_dtype.full_name}")

            torch_dtype = _slang_dtype_to_torch(dtype)
            if torch_dtype is None:
                raise ValueError(
                    f"Element type {slang_dtype.full_name} incompatible with torch")

            full_dims = dims + len(slang_dtype.shape)

            super().__init__(layout, dtype, full_dims, True, d_in, d_out)
            self.d_in: Optional[TorchTensorMarshall]
            self.d_out: Optional[TorchTensorMarshall]

            self.torch_dtype = torch_dtype
            self.slang_dtype = slang_dtype

        def create_calldata(self, context: CallContext, binding: BoundVariableRuntime, data: Tensor | torch.Tensor | WrappedTensor) -> Any:
            if isinstance(data, Tensor):
                return super().create_calldata(context, binding, data)

            tensor = wrap_tensor(data)

            readable = not tensor.primal.is_meta
            if isinstance(binding.vector_type, ITensorType):
                writable = binding.vector_type.writable
            else:
                writable = binding.access[0] in (AccessType.write, AccessType.readwrite) or \
                    binding.access[1] in (AccessType.read, AccessType.readwrite)

            sgl_tensor = _TORCH_STATE.active_call.convert_tensor_to_sgl(
                self, tensor, readable, writable)

            result = super().create_calldata(context, binding, sgl_tensor)

            return result

        def read_calldata(self, context: CallContext, binding: NativeBoundVariableRuntime, data: Any, result: Any) -> None:
            _TORCH_STATE.active_call.read_calldata(self)

        def create_output(self, context: CallContext, binding: BoundVariableRuntime) -> Any:
            slang_shape = context.call_shape.as_tuple() + self.slang_dtype.shape.as_tuple()

            return WrappedTensor(torch.empty(slang_shape, dtype=self.torch_dtype, device=torch.device('meta')))

        def get_shape(self, value: torch.Tensor | WrappedTensor | None = None) -> Shape:
            if isinstance(value, torch.Tensor):
                return Shape(value.shape)
            elif isinstance(value, WrappedTensor):
                return Shape(value.primal.shape)
            else:
                return super().get_shape()

        def read_output(self, context: CallContext, binding: BoundVariableRuntime, data: torch.Tensor | WrappedTensor) -> Any:
            return _TORCH_STATE.active_call.read_output(self, wrap_tensor(data))

    def create_tensor_marshall(layout: SlangProgramLayout, value: Any):
        if isinstance(value, torch.Tensor):
            dtype = _torch_dtype_to_slang(value.dtype, layout)
            if dtype is None:
                raise ValueError(f"Unsupported torch dtype {value.dtype}")
            marshall = TorchTensorMarshall(layout, dtype, len(value.shape), None, None)
        elif isinstance(value, ReturnContext):
            if value.bind_context.call_dimensionality == 0 and False:
                return tr.get_or_create_type(layout, ValueRef, value)
            else:
                marshall = TorchTensorMarshall(layout, value.slang_type,
                                               value.bind_context.call_dimensionality, None, None)
        elif isinstance(value, WrappedTensor):
            assert value.primal is not None
            dtype = _torch_dtype_to_slang(value.primal.dtype, layout)
            if dtype is None:
                raise ValueError(f"Unsupported torch dtype {value.primal.dtype}")

            d_in = create_tensor_marshall(
                layout, value.grad_in) if value.grad_in is not None else None
            d_out = create_tensor_marshall(
                layout, value.grad_out) if value.grad_out is not None else None

            marshall = TorchTensorMarshall(layout, dtype, len(value.primal.shape), d_in, d_out)
        else:
            raise ValueError(f"Type {type(value)} is unsupported for torch.Tensor marshall")

        return marshall

    def hash_tensor(value: Any) -> str:
        if isinstance(value, torch.Tensor):
            return f"torch.Tensor[{value.dtype},{value.ndim},{value.device}]"
        elif isinstance(value, WrappedTensor):
            sig = f"TorchTensorWithGrad[hash_tensor(value.primal)"
            if value.grad_in is not None:
                sig += hash_tensor(value.grad_in)
            sig += ","
            if value.grad_out is not None:
                sig += hash_tensor(value.grad_out)
            sig += "]"

            return sig
        else:
            raise ValueError(f"Unexpected type {type(value).__name__} for tensor hashing")

    PYTHON_TYPES[torch.Tensor] = create_tensor_marshall
    PYTHON_TYPES[torch.nn.Parameter] = create_tensor_marshall
    PYTHON_TYPES[WrappedTensor] = create_tensor_marshall
    PYTHON_SIGNATURES[torch.Tensor] = hash_tensor
    PYTHON_SIGNATURES[torch.nn.Parameter] = hash_tensor
    PYTHON_SIGNATURES[WrappedTensor] = hash_tensor
