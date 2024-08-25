# TODO: DiffTensor is non-differentiable, but still needs gradients assigned
# TODO: Tensor gradients should use accum instead of set

from __future__ import annotations

from .layer_base import TensorLayer, TensorRef, BufferRef, DeviceKind, Device

from ..types.base import ScalarKind
from ..types.diffpair import DifferentialPairType, DifferentialPairTranslator
from ..types.tensor import TensorType
from ..types.helpers import CollectionTranslator

from ..variable import Variable, AssignmentCursor

import torch
from torch.utils.dlpack import from_dlpack

from typing import Optional, Any, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from ..invokables import InvokableSlangFunc

kind_to_dtype = {
    ScalarKind.Bool: torch.bool,
    ScalarKind.Uint8: torch.uint8,
    ScalarKind.Uint16: torch.int16,
    ScalarKind.Uint: torch.int32,
    ScalarKind.Uint64: torch.int64,
    ScalarKind.Int8: torch.int8,
    ScalarKind.Int16: torch.int16,
    ScalarKind.Int: torch.int32,
    ScalarKind.Int64: torch.int64,
    ScalarKind.Float16: torch.float16,
    ScalarKind.Float: torch.float32,
    ScalarKind.Float64: torch.float64,
}
dtype_to_kind = {v: k for k, v in kind_to_dtype.items()}


def is_tensor_like(t: Any):
    return torch.overrides.is_tensor_like(t)


def wrap_kernel_call(func: InvokableSlangFunc, root_var: Variable, values: list[Any]):
    # torch doesn't look inside data structures in arguments, and we first need to extract all
    # tensors into bare arguments so that torch properly tracks them
    in_tensors = [
        value.raw_tensor
        for value, var in zip(values, root_var.nodes())
        if var.is_in and isinstance(value, TorchTensorRef)
    ]

    return KfFunction.apply(func, root_var, values, *in_tensors)


def meta_tensor(dtype: ScalarKind, shape: tuple[int, ...]) -> TensorRef:
    t = torch.empty(shape, dtype=kind_to_dtype[dtype], device=torch.device("meta"))
    return TorchTensorRef(t, None)


class TorchLayer(TensorLayer):
    def __init__(self, device: torch.device = torch.device("cuda")):
        super().__init__()
        self.torch_device = device

    def is_tensor(self, t: Any) -> bool:
        return is_tensor_like(t)

    def wrap_tensor(self, t: torch.Tensor | TensorRef) -> TensorRef:
        if isinstance(t, TorchTensorRef):
            return t
        elif isinstance(t, torch.Tensor):
            return TorchTensorRef(t)
        raise ValueError("Argument has to be a torch tensor or a TensorRef")

    def import_tensor(self, dlpack_ndarray: Any, buffer: BufferRef) -> TorchTensorRef:
        if isinstance(dlpack_ndarray, torch.Tensor):
            raw_tensor = dlpack_ndarray
        else:
            raw_tensor = from_dlpack(dlpack_ndarray)

        return TorchTensorRef(raw_tensor, buffer)

    def meta_tensor(self, dtype: ScalarKind, shape: tuple[int, ...]) -> TensorRef:
        return meta_tensor(dtype, shape)

    def empty_ref(self) -> TensorRef:
        return TorchTensorRef()

    def wrap_kernel_call(
        self, func: InvokableSlangFunc, root_var: Variable, values: Any
    ):
        wrap_kernel_call(func, root_var, values)

    def device(self) -> Device:
        return TorchDevice(self.torch_device)


class TensorMethodWrapper(type):
    def __new__(cls, name: str, bases: tuple[type, ...], dct: dict[str, Any]):
        exclusions = {"__new__"}

        def wrap_call(f: Any):
            return lambda *args, **kwargs: TorchTensorRef.__torch_function__(
                f, (TorchTensorRef,), args, kwargs
            )

        for src in torch.Tensor.mro():
            if src is not object:
                for k, v in src.__dict__.items():
                    if k in exclusions:
                        continue
                    if hasattr(v, "__call__"):
                        dct[k] = wrap_call(v)
                    elif hasattr(v, "__get__") and k[0] != "_":
                        getter = wrap_call(v.__get__) if hasattr(v, "__get__") else None
                        setter = wrap_call(v.__set__) if hasattr(v, "__set__") else None
                        deleter = (
                            wrap_call(v.__delete__)
                            if hasattr(v, "__delete__")
                            else None
                        )
                        dct[k] = property(fget=getter, fset=setter, fdel=deleter)

        return super().__new__(cls, name, bases, dct)


class TorchTensorRef(TensorRef, metaclass=TensorMethodWrapper):
    def __init__(
        self,
        raw_tensor: Optional[torch.Tensor] = None,
        buffer: Optional[BufferRef] = None,
    ):
        self.raw_tensor = raw_tensor
        if buffer is None and raw_tensor is not None and not raw_tensor.is_meta:
            buffer = TorchBuffer(raw_tensor.untyped_storage())
        self.backing_buffer = buffer
        super().__init__()

    def point_to(self, other: TensorRef):
        assert isinstance(other, TorchTensorRef)
        self.raw_tensor = other.raw_tensor
        self.backing_buffer = other.backing_buffer
        self.version = (
            self.backing_buffer.version if self.backing_buffer is not None else 0
        )

    def is_empty(self):
        return self.raw_tensor is None

    def unwrap(self):
        return self.raw_tensor

    def buffer(self) -> Optional[BufferRef]:
        return self.backing_buffer

    def copy_data(self, src: TensorRef):
        assert self.buffer() is not None
        assert isinstance(src, TorchTensorRef)
        assert self.raw_tensor is not None
        assert src.raw_tensor is not None
        self.raw_tensor.copy_(src.raw_tensor)

    def get_dtype(self) -> ScalarKind:
        assert self.raw_tensor is not None
        return dtype_to_kind[self.raw_tensor.dtype]

    def get_shape(self) -> tuple[int, ...]:
        assert self.raw_tensor is not None
        return tuple(self.raw_tensor.shape)

    def get_strides(self) -> tuple[int, ...]:
        assert self.raw_tensor is not None
        return tuple(self.raw_tensor.stride())

    def get_offset(self) -> int:
        assert self.raw_tensor is not None
        if self.backing_buffer is None:
            return 0
        else:
            return self.raw_tensor.storage_offset()

    def create_view(
        self, offset: int, shape: tuple[int, ...], strides: tuple[int, ...]
    ) -> TensorRef:
        assert self.raw_tensor is not None
        offset += self.raw_tensor.storage_offset()
        t = torch.as_strided(self.raw_tensor, shape, strides, storage_offset=offset)
        return TorchTensorRef(t, self.backing_buffer)

    def make_contiguous(self) -> TensorRef:
        assert self.raw_tensor is not None
        if self.raw_tensor.is_contiguous():
            return self
        else:
            return TorchTensorRef(self.raw_tensor.contiguous(), None)

    def __repr__(self):
        return f"TorchTensorRef(version={self.version}, buffer={repr(self.backing_buffer)}, tensor={repr(self.raw_tensor)})"

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ):
        if len(types) != 1 or types[0] is not TorchTensorRef:
            raise NotImplementedError(f"Don't support types {types}")

        if kwargs is None:
            kwargs = {}

        ptr_to_buf = {}

        def check_tensor(t: Any):
            if isinstance(t, TorchTensorRef):
                t.check_valid()
                if t.backing_buffer is not None:
                    assert t.raw_tensor is not None
                    ptr_to_buf[t.raw_tensor.untyped_storage().data_ptr()] = (
                        t.backing_buffer
                    )

        def wrap(t: Any):
            if isinstance(t, torch.Tensor):
                ptr = t.untyped_storage().data_ptr()
                if ptr in ptr_to_buf:
                    return TorchTensorRef(t, ptr_to_buf[ptr])
                else:
                    return TorchTensorRef(t)
            return t

        def unwrap(v: Any):
            if isinstance(v, TorchTensorRef):
                if v.is_empty():
                    raise RuntimeError("Attempting to access empty TorchTensorRef")
                return v.unwrap()
            elif isinstance(v, list) or isinstance(v, tuple):
                return type(v)(unwrap(t) for t in v)
            elif isinstance(v, dict):
                return {k: unwrap(t) for k, t in v.items()}
            else:
                return v

        for t in args:
            check_tensor(t)
        for t in kwargs.values():
            check_tensor(t)

        result = torch.Tensor.__torch_function__(func, (), unwrap(args), unwrap(kwargs))

        if isinstance(result, list) or isinstance(result, tuple):
            return type(result)(wrap(t) for t in result)
        elif isinstance(result, torch.Tensor):
            return wrap(result)
        else:
            return result


class TorchBuffer(BufferRef):
    def __init__(self, storage: torch.UntypedStorage):
        super().__init__()
        self.storage = storage
        self.version = 0

    def data_ptr(self):
        return self.storage.data_ptr()

    def device(self):
        return TorchDevice(self.storage.device)


class TorchDevice(Device):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        if self.device.type == "cuda":
            self.device_kind = DeviceKind.Cuda
        elif self.device.type == "cpu":
            self.device_kind = DeviceKind.CPU
        else:
            self.device_kind = DeviceKind.Other

    def kind(self) -> DeviceKind:
        return self.device_kind

    def idx(self) -> int:
        return self.device.index

    def stream(self):
        if self.device_kind == DeviceKind.Cuda:
            return torch.cuda.current_stream(self.device).cuda_stream
        return None

    def sync(self):
        if self.device_kind == DeviceKind.Cuda:
            torch.cuda.synchronize(self.device)
        elif self.device_kind == DeviceKind.CPU:
            pass
        elif self.device_kind == DeviceKind.Other:
            raise RuntimeError(f"Don't know how to sync for torch device {self.device}")


class KfFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        func: InvokableSlangFunc,
        root_var: Variable,
        values: list[Any],
        *tensors: torch.Tensor,
    ):
        func.specialize(root_var).call(values)

        # We need to save the inputs for later use in the backwards pass. However, we have to
        # explicitly separate tensor values from regular values and pass them to save_for_backward
        # instead of saving them on the context directly. This is to support torch saved
        # tensor hooks, better torch garbage collecting and error checking for inplace operations
        primal_values = values[:]
        input_needs_grads = []
        saved_tensors = []
        output_tensors = []
        nodiff_output_tensors = []
        for i, var in enumerate(root_var.nodes()):
            if not isinstance(primal_values[i], TorchTensorRef):
                continue
            tensor = primal_values[i].raw_tensor
            primal_values[i] = TorchTensorRef(None, primal_values[i].buffer())

            saved_tensors.append(tensor)

            if var.is_in:
                input_needs_grads.append(tensor.requires_grad)
            if var.is_out:
                output_tensors.append(tensor)
                if not var.is_differentiable:
                    nodiff_output_tensors.append(tensor)

        ctx.primal = func
        ctx.primal_vars = root_var
        ctx.primal_values = primal_values
        ctx.input_needs_grads = input_needs_grads
        ctx.mark_non_differentiable(*nodiff_output_tensors)
        ctx.save_for_backward(*saved_tensors)

        return tuple(output_tensors)

    @staticmethod
    # @torch.autograd.once_differentiable
    def backward(ctx: Any, *output_grad_tuple: torch.Tensor):
        primal: InvokableSlangFunc = ctx.primal
        primal_vars: Variable = ctx.primal_vars
        primal_values = ctx.primal_values[:]
        primal_params = primal.func_params

        # Reassemble input values
        saved_tensors = list(ctx.saved_tensors)
        requested_grads = list(ctx.input_needs_grads)
        requested_grad_vars = set()
        primal_tensors = {}
        for i, var in enumerate(primal_vars.nodes()):
            if isinstance(primal_values[i], TorchTensorRef):
                primal_values[i].raw_tensor = saved_tensors.pop(0)
                primal_tensors[var] = primal_values[i]

                if var.is_in:
                    if requested_grads.pop(0):
                        requested_grad_vars.add(var)

        # Extract the root level values, i.e. value directly assigned to each parameter rather than any children
        primal_root_values = []
        idx = 1
        for var in primal_vars.children.values():
            primal_root_values.append(primal_values[idx])
            idx += sum(1 for _ in var.nodes())

        # Now differentiate the function signature to begin value reshuffling
        bwd_diff = primal.differentiate()

        # Now begins the big shuffle of assigning primary values onto parameters of the differential function
        bwd_vars = AssignmentCursor.from_function(bwd_diff)
        output_grads = list(output_grad_tuple)
        input_grads = {}

        def is_tensor(var: Variable):
            return isinstance(var.type, TensorType) or isinstance(
                var.translator, TensorType
            )

        def contains_tensor(var: Variable):
            return any(is_tensor(v) for v in var.leaves())

        def differentiate_values(
            primal_var: Variable, bwd_var: AssignmentCursor, was_out: bool, is_out: bool
        ):
            if not contains_tensor(primal_var):
                bwd_var.prune()
                return

            if is_tensor(primal_var):
                requests_grads = primal_var in requested_grad_vars
                if not was_out and not requests_grads:
                    # This was a pure input that grads were not requested for, i.e. it consumes no gradients
                    # and returns no gradients. Prune it from the differential variable tree.
                    bwd_var.prune()
                    return

                # This variable consumes output gradients, produces input gradients, or both. In any case, it will be
                # assigned a tensor to hold those gradients.
                # Let's figure out what that tensor should be:
                if was_out:
                    # This variable was an output, and it will have externally provided gradients of those outputs.
                    # Consume those now.
                    bwd_val = output_grads.pop(0)
                else:
                    # Otherwise, this was an input that returns gradients; fill in an empty tensor to hold those gradients
                    assert primal_var in primal_tensors
                    primal_tensor = primal_tensors[primal_var]
                    bwd_val = meta_tensor(
                        primal_tensor.get_dtype(), primal_tensor.get_shape()
                    )
                # Assign the tensor
                bwd_var.set(bwd_val)
                # If this variable will return gradients, record that tensor for later
                # Store bwd_var.value as the result of .set() rather than bwd_val directly;
                # .set() might create views of the tensor that break the tensor ref
                if is_out and requests_grads:
                    input_grads[primal_var] = bwd_var.value

            # Recursively differentiate the child variables
            for k in primal_var.children.keys():
                if not primal_var[k].is_differentiable:
                    continue

                differentiate_values(primal_var[k], bwd_var[k], was_out, is_out)

            if bwd_var.children:
                # We might only partially assign the differentiated value. Make sure the whole thing is zero-initialized
                assert bwd_var.translator is not None
                bwd_var.set_translator(
                    CollectionTranslator(primal_var.type.to_slang_type() + "::dzero()")
                )

            if primal_var.children and not bwd_var.children:
                # All child variables were pruned during differentiation (i.e. they had no gradients
                # or no gradients were requested for them). Prune the parent too.
                bwd_var.prune()

        primal_param_map = {p.name: i for i, p in enumerate(primal_params)}
        for bwd_p in bwd_diff.func_params:
            assert bwd_p.name in primal_param_map
            primal_idx = primal_param_map[bwd_p.name]
            primal_p = primal_params[primal_idx]
            primal_value = primal_root_values[primal_idx]
            primal_var = primal_vars[primal_p.name]

            if primal_p.derivative_degree == bwd_p.derivative_degree:
                # Primal parameter was retained unchanged
                # -> Copy over its values
                bwd_vars[bwd_p.name] = primal_value
            else:
                # This parameter was retained and differentiated.

                # There's no scenario where a bwd_diff function ends
                # up with a pure out parameter
                assert bwd_p.is_in()

                bwd_var = bwd_vars[bwd_p.name]
                if bwd_p.is_out():
                    # This is an inout DifferentialPair parameter, and receives the primal parameters.
                    # It will return gradients, and may receive gradients if it was previously inout
                    assert isinstance(bwd_vars[bwd_p.name].type, DifferentialPairType)

                    bwd_var.create_sub_assignment("p", True, False, False)
                    bwd_var["p"] = primal_value
                    bwd_var.create_sub_assignment("d", primal_p.is_out(), True, True)
                    bwd_var = bwd_var["d"]

                differentiate_values(
                    primal_var, bwd_var, primal_p.is_out(), bwd_p.is_out()
                )

                if bwd_p.is_out():
                    bwd_var = bwd_vars[bwd_p.name]
                    have_differential = "d" in bwd_var
                    if not have_differential:
                        bwd_var.is_out = False
                    assert isinstance(bwd_var.type, DifferentialPairType)
                    type = cast(DifferentialPairType, bwd_var.type)
                    translator = DifferentialPairTranslator(
                        type.primal, type.differential, have_differential
                    )
                    bwd_var.set_translator(translator)

        vars, values = bwd_vars.finalize()
        wrap_kernel_call(bwd_diff, vars, values)

        # Have to return None grads for the `func`, `root_var` and `values` parameters
        return_grads = [None, None, None]
        for i, var in enumerate(primal_vars.nodes()):
            if isinstance(primal_values[i], TorchTensorRef):
                if var in requested_grad_vars:
                    if var not in input_grads:
                        var.error("Requested gradients for non-differentiable input")
                    return_grads.append(
                        input_grads[var].raw_tensor
                    )  # TODO: torch, why?
                else:
                    return_grads.append(None)
        return tuple(return_grads)
