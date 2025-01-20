# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Optional, Union
import torch

from slangpy.core.native import AccessType
from slangpy.torchintegration.wrappedtensor import WrappedTensor
from slangpy.core.function import Function, IThis
import slangpy.reflection as kfr
from slangpy.backend import (FunctionReflection, TypeConformance, Device)

if TYPE_CHECKING:
    from slangpy.core.module import Module
    from slangpy.core.struct import Struct
    from slangpy.torchintegration.torchstruct import TorchStruct


def check_cuda_enabled(device: Device):
    if not device.supports_cuda_interop:
        raise RuntimeError("Cuda interop must be enabled for torch support "
                           "create SGL device with Device..., enable_cuda_interop=True")


def unpack_arg(arg: Any, tensors: list[torch.Tensor]) -> Any:
    if hasattr(arg, "get_this"):
        arg = arg.get_this()
    if isinstance(arg, dict):
        arg = {k: unpack_arg(v, tensors) for k, v in arg.items()}
    if isinstance(arg, (list, tuple)):
        arg = [unpack_arg(v, tensors) for v in arg]
    if isinstance(arg, torch.Tensor):
        id = len(tensors)
        tensor = arg.contiguous()
        tensors.append(tensor)
        arg = WrappedTensor(id=id, primal=tensor)
    return arg


def populate_tensor_refs(arg: Any, tensors: tuple[torch.Tensor, ...]) -> Any:
    if isinstance(arg, dict):
        arg = {k: populate_tensor_refs(v, tensors) for k, v in arg.items()}
    if isinstance(arg, (list, tuple)):
        arg = [populate_tensor_refs(v, tensors) for v in arg]
    if isinstance(arg, WrappedTensor) and arg.id >= 0:
        arg.primal = tensors[arg.id]
        if arg.grad_in is not None:
            arg.grad_in = populate_tensor_refs(arg.grad_in, tensors)
        if arg.grad_out is not None:
            arg.grad_out = populate_tensor_refs(arg.grad_out, tensors)


def clear_tensor_refs(arg: Any) -> Any:
    if isinstance(arg, dict):
        arg = {k: clear_tensor_refs(v) for k, v in arg.items()}
    if isinstance(arg, (list, tuple)):
        arg = [clear_tensor_refs(v) for v in arg]
    if isinstance(arg, WrappedTensor) and arg.id >= 0:
        arg.primal = None
        if arg.grad_in is not None:
            arg.grad_in = clear_tensor_refs(arg.grad_in)
        if arg.grad_out is not None:
            arg.grad_out = clear_tensor_refs(arg.grad_out)
    return arg


def gather_and_clear_primal_tensors(arg: Any,
                                    primal_in_tensors: list[torch.Tensor],
                                    primal_out_tensors: list[torch.Tensor]) -> Any:

    if isinstance(arg, dict):
        arg = {k: gather_and_clear_primal_tensors(
            v, primal_in_tensors, primal_out_tensors) for k, v in arg.items()}
    if isinstance(arg, (list, tuple)):
        arg = [gather_and_clear_primal_tensors(
            v, primal_in_tensors, primal_out_tensors) for v in arg]
    if isinstance(arg, WrappedTensor) and arg.id >= 0:
        if arg.last_access_type[0] in (AccessType.read, AccessType.readwrite):
            primal_in_tensors.append(arg.primal)
        if arg.last_access_type[0] in (AccessType.write, AccessType.readwrite):
            primal_out_tensors.append(arg.primal)
    return arg


def assign_primal_and_grad_tensors(arg: Any, all_tensors: list[torch.Tensor], grad_in_tensors: list[torch.Tensor], grad_out_tensors: list[torch.Tensor]) -> Any:
    if isinstance(arg, dict):
        arg = {k: assign_primal_and_grad_tensors(
            v, all_tensors, grad_in_tensors, grad_out_tensors) for k, v in arg.items()}
    if isinstance(arg, (list, tuple)):
        arg = [assign_primal_and_grad_tensors(
            v, all_tensors, grad_in_tensors, grad_out_tensors) for v in arg]
    if isinstance(arg, WrappedTensor) and arg.id >= 0:
        arg.primal = all_tensors[arg.id]
        if arg.last_access_type[0] in (AccessType.read, AccessType.readwrite):
            arg.grad_out = WrappedTensor(primal=torch.zeros_like(arg.primal))
            grad_out_tensors.append(arg.grad_out.primal)
        if arg.last_access_type[0] in (AccessType.write, AccessType.readwrite):
            arg.grad_in = WrappedTensor(primal=grad_in_tensors.pop(0).contiguous())
    return arg


def alloc_gradients(arg: Any, tensors: list[Optional[torch.Tensor]]) -> Any:
    if isinstance(arg, dict):
        arg = {k: alloc_gradients(v, tensors) for k, v in arg.items()}
    if isinstance(arg, (list, tuple)):
        arg = [alloc_gradients(v, tensors) for v in arg]
    if isinstance(arg, WrappedTensor):
        if arg.primal is not None and arg.primal.requires_grad:
            grad = torch.zeros_like(arg.primal)
            arg.grad_out = WrappedTensor(grad)
            tensors.append(grad)
        else:
            tensors.append(None)
    return arg


class TorchAutoGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        spy_function: Function,
        unpacked_args: tuple[Any, ...],
        unpacked_kwargs: dict[str, Any],
        primal_in_tensors: tuple[torch.Tensor],
        primal_out_tensors: tuple[torch.Tensor],
        all_tensors: tuple[torch.Tensor],
        *tensors: torch.Tensor
    ):
        # Store inputs
        ctx.spy_function = spy_function
        ctx.unpacked_args = unpacked_args
        ctx.unpacked_kwargs = unpacked_kwargs

        # Save inputs and outputs for backwards pass then return result
        ctx.save_for_backward(*all_tensors)
        return primal_out_tensors

    @staticmethod
    def backward(ctx: Any, *args: torch.Tensor):
        # Load parameters from context
        spy_function: Function = ctx.spy_function
        unpacked_args: tuple[Any, ...] = ctx.unpacked_args
        unpacked_kwargs: dict[str, Any] = ctx.unpacked_kwargs
        result_out_provided = "_result" in unpacked_kwargs

        all_tensors = list(ctx.saved_tensors)
        grad_in_tensors: list[torch.Tensor] = list(args)
        grad_out_tensors: list[torch.Tensor] = []
        assign_primal_and_grad_tensors((unpacked_args, unpacked_kwargs),
                                       all_tensors, grad_in_tensors, grad_out_tensors)

        # Check for a final tensor from the args, which would be the return value if there was one
        # This is only necessary if user did not supply an _result argument (if they did, the
        # assign_primal_and_grad_tensors function will have already set it up correctly).
        if not result_out_provided and len(grad_in_tensors) > 0:
            assert len(grad_in_tensors) == 1
            result_grad_tensor = args[0].contiguous()

            # Setup the result input tensor
            result = WrappedTensor(ctx.saved_tensors[-1])
            result.grad_in = WrappedTensor(result_grad_tensor)
        else:
            result_grad_tensor = None
            result = None

        # Gather streams from tensors (both saved tensors + args)
        streams: set[int] = set()
        for tensor in all_tensors:
            if tensor.is_cuda:
                streams.add(torch.cuda.current_stream(tensor.device).cuda_stream)
        for gout in grad_out_tensors:
            if gout.is_cuda:
                streams.add(torch.cuda.current_stream(gout.device).cuda_stream)
        for arg in args:
            if arg.is_cuda:
                streams.add(torch.cuda.current_stream(arg.device).cuda_stream)

        # Sync device with cuda
        for stream in streams:
            spy_function.module.device.sync_to_cuda(stream)

        # Run backwards pass
        if result is not None:
            spy_function.bwds(*unpacked_args, **unpacked_kwargs, _result=result)
        else:
            spy_function.bwds(*unpacked_args, **unpacked_kwargs)

        # Sync cuda with device
        for stream in streams:
            spy_function.module.device.sync_to_device(stream)

        # Clear the tensors after passing to the function
        clear_tensor_refs((unpacked_args, unpacked_kwargs))

        # Return the gradients
        res = (None, None, None, None, None, None) + tuple(grad_out_tensors)
        return res


class TorchFunction(torch.nn.Module):

    def __init__(self, function: Function):
        super().__init__()
        check_cuda_enabled(function.module.device)
        self.function = function.return_type(WrappedTensor)

    def forward(self, *args: Any, **kwargs: Any):
        # Build 'unpacked' args (that handle IThis)
        all_tensors: list[torch.Tensor] = []
        unpacked_args = tuple([unpack_arg(x, all_tensors) for x in args])
        unpacked_kwargs = {k: unpack_arg(v, all_tensors) for k, v in kwargs.items()}
        result_out_provided = "_result" in unpacked_kwargs

        # Gather streams from tensors
        streams: set[int] = set()
        for tensor in all_tensors:
            if tensor.is_cuda:
                streams.add(torch.cuda.current_stream(tensor.device).cuda_stream)

        # Sync device with cuda
        for stream in streams:
            self.function.module.device.sync_to_cuda(stream)

        # Get the result
        result = self.function(*unpacked_args, **unpacked_kwargs)

        if isinstance(result, WrappedTensor):
            assert result.primal is not None
            result = result.primal
            if result.is_cuda:
                streams.add(torch.cuda.current_stream(result.device).cuda_stream)

        # Sync cuda with device
        for stream in streams:
            self.function.module.device.sync_to_device(stream)

        # Gather up all tensors in the arguments into in/out lists and clear references so
        # garbage collect works properly
        primal_in_tensors: list[torch.Tensor] = []
        primal_out_tensors: list[torch.Tensor] = []
        gather_and_clear_primal_tensors(
            (unpacked_args, unpacked_kwargs), primal_in_tensors, primal_out_tensors)

        # If result is a tensor, add it to the list of all and result tensors
        if not result_out_provided and isinstance(result, torch.Tensor):
            all_tensors.append(result)
            primal_out_tensors.append(result)

        # Call the dummy auto-grad apply function, which critically takes the primal input list
        # as arguments and returns the primal output list as results
        TorchAutoGradFunction.apply(
            self.function, unpacked_args, unpacked_kwargs,
            tuple(primal_in_tensors),
            tuple(primal_out_tensors),
            tuple(all_tensors),
            *primal_in_tensors)

        # Return the single result
        return result

    def attach(self, module: 'Module', func: Union[str, kfr.SlangFunction, list[FunctionReflection]], struct: Optional['Struct'] = None, options: dict[str, Any] = {}) -> None:
        """
        Links a function to its parent module or struct. Typically only called internally by SlangPy.
        """
        self.function.attach(module, func, struct, options)

    def bind(self, this: IThis):
        """
        Bind a `this` object to the function. Typically
        this is called automatically when calling a function on a struct.
        """
        return TorchFunction(self.function.bind(this))

    def map(self, *args: Any, **kwargs: Any):
        """
        Apply dimension or type mapping to all or some of the arguments.

        myfunc.map((1,)(0,))(arg1, arg2) # Map arg1 to dimension 1, arg2 to dimension 0

        myfunc.map(module.Foo, module.Bar)(arg1, arg2) # Cast arg1 to Foo, arg2 to Bar
        """
        return TorchFunction(self.function.map(*args, **kwargs))

    def set(self, *args: Any, **kwargs: Any):
        """
        Specify additional uniform values that should be set whenever the function's kernel
        is dispatched. Useful for setting constants or other values that are not passed as arguments.
        """
        return TorchFunction(self.function.set(*args, **kwargs))

    def constants(self, constants: dict[str, Any]):
        """
        Specify link time constants that should be set when the function is compiled. These are
        the most optimal way of specifying unchanging data, however note that changing a constant
        will result in the function being recompiled.
        """
        return TorchFunction(self.function.constants(constants))

    def type_conformances(self, type_conformances: list[TypeConformance]):
        """
        Specify Slang type conformances to use when compiling the function.
        """
        return TorchFunction(self.function.type_conformances(type_conformances))

    def return_type(self, return_type: Union[type, str]):
        """
        Explicitly specify the desired return type from the function.
        """
        return TorchFunction(self.function.return_type(return_type))

    @property
    def name(self):
        """
        Get the name of the function.
        """
        return self.function.name

    def as_func(self) -> 'TorchFunction':
        """
        Typing helper to cast the function to a function (i.e. a no-op)
        """
        return self

    def as_struct(self) -> 'TorchStruct':
        """
        Typing helper to detect attempting to treat a function as a struct.
        """
        raise ValueError("Cannot convert a function to a struct")
