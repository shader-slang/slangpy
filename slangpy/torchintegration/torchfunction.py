# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Optional, Union
import torch

from slangpy.torchintegration.wrappedtensor import WrappedTensor
from slangpy.core.function import Function, IThis
import slangpy.reflection as kfr
from slangpy.backend import (FunctionReflection, TypeConformance)

if TYPE_CHECKING:
    from slangpy.core.module import Module
    from slangpy.core.struct import Struct
    from slangpy.torchintegration.torchstruct import TorchStruct


def unpack_arg(arg: Any, tensors: list[torch.Tensor]) -> Any:
    if hasattr(arg, "get_this"):
        arg = arg.get_this()
    if isinstance(arg, dict):
        arg = {k: unpack_arg(v, tensors) for k, v in arg.items()}
    if isinstance(arg, (list, tuple)):
        arg = [unpack_arg(v, tensors) for v in arg]
    if isinstance(arg, torch.Tensor):
        tensors.append(arg)
        arg = WrappedTensor(arg)
    return arg


def alloc_gradients(arg: Any, tensors: list[torch.Tensor]) -> Any:
    if hasattr(arg, "get_this"):
        arg = arg.get_this()
    if isinstance(arg, dict):
        arg = {k: alloc_gradients(v, tensors) for k, v in arg.items()}
    if isinstance(arg, (list, tuple)):
        arg = [alloc_gradients(v, tensors) for v in arg]
    if isinstance(arg, WrappedTensor):
        grad = torch.zeros_like(arg.primal)
        arg.grad_out = WrappedTensor(grad)
        tensors.append(grad)
    return arg


class TorchAutoGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        spy_function: Function,
        unpacked_args: tuple[Any, ...],
        unpacked_kwargs: dict[str, Any],
        *tensors: torch.Tensor
    ):
        # Store inputs
        ctx.spy_function = spy_function
        ctx.unpacked_args = unpacked_args
        ctx.unpacked_kwargs = unpacked_kwargs

        # Sync device with cuda
        spy_function.module.device.sync_to_cuda()

        # Get the result (will be a slangpy wrapped tensor)
        wrapped_tensor = spy_function(*unpacked_args, **unpacked_kwargs)
        assert isinstance(wrapped_tensor, WrappedTensor)
        result = wrapped_tensor.primal

        # Sync cuda with device
        spy_function.module.device.sync_to_device()

        # Save inputs and outputs for backwards pass then return result
        ctx.save_for_backward(*tensors+(result,))
        return result

    @staticmethod
    def backward(ctx: Any, *args: torch.Tensor):
        # Load parameters from context
        spy_function: Function = ctx.spy_function
        unpacked_args: tuple[Any, ...] = ctx.unpacked_args
        unpacked_kwargs: dict[str, Any] = ctx.unpacked_kwargs

        # Get the color gradient tensor
        result_grad_tensor = args[0]

        # Setup the result input tensor
        result = WrappedTensor(ctx.saved_tensors[-1])
        result.grad_in = WrappedTensor(result_grad_tensor)

        # Alloc gradients and get list back. As alloc_gradients
        # runs the same process as unpack_arg, the gradients list
        # will match 1-to-1 the input tensors list.
        gradients: list[torch.Tensor] = []
        alloc_gradients((unpacked_args, unpacked_kwargs), gradients)

        # Sync device with cuda
        spy_function.module.device.sync_to_cuda()

        # Run backwards pass
        spy_function.bwds(*unpacked_args, **unpacked_kwargs, _result=result)

        # Sync cuda with device
        spy_function.module.device.sync_to_device()

        # Return the gradients
        res = (None, None, None) + tuple(gradients)
        return res


def find_tensors(element: Any, tensors: list[torch.Tensor]):
    if torch is None:
        raise RuntimeError("Torch support is not available because torch is not installed")

    if isinstance(element, dict):
        for k, v in element.items():
            find_tensors(v, tensors)
    elif isinstance(element, (list, tuple)):
        for v in element:
            find_tensors(v, tensors)
    elif isinstance(element, (torch.Tensor,)):
        tensors.append(element)


class TorchFunction(torch.nn.Module):

    def __init__(self, function: Function):
        super().__init__()
        self.function = function.return_type(WrappedTensor)

    def forward(self, *args: Any, **kwargs: Any):
        # Build 'unpacked' args (that handle IThis)
        tensors: list[torch.Tensor] = []
        unpacked_args = tuple([unpack_arg(x, tensors) for x in args])
        unpacked_kwargs = {k: unpack_arg(v, tensors) for k, v in kwargs.items()}

        find_tensors((unpacked_args, unpacked_kwargs), tensors)

        return TorchAutoGradFunction.apply(self.function, unpacked_args, unpacked_kwargs, *tensors)

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
