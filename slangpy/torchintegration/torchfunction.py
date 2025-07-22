# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Any, Optional, Union, cast
import torch

from slangpy.core.native import AccessType, unpack_refs_and_args, unpack_refs_and_kwargs
from slangpy.torchintegration.wrappedtensor import WrappedTensor
from slangpy.torchintegration.torchtensormarshall import TensorRef
from slangpy.core.function import Function, FunctionNode, IThis
from slangpy import TypeConformance, Device, DeviceType, NativeHandle

if TYPE_CHECKING:
    from slangpy.torchintegration.torchstruct import TorchStruct


def check_cuda_enabled(device: Device):
    if not device.supports_cuda_interop and device.info.type != DeviceType.cuda:
        raise RuntimeError(
            "Cuda interop must be enabled for torch support "
            "create SGL device with Device..., enable_cuda_interop=True"
        )


def populate_tensor_refs(args: list[TensorRef], tensors: tuple[torch.Tensor, ...]) -> Any:
    for arg in args:
        if arg.id >= 0:
            arg.tensor = tensors[arg.id]
        if arg.grad_in is not None and arg.grad_in.id >= 0:
            arg.grad_in.tensor = tensors[arg.grad_in.id]
        if arg.grad_out is not None and arg.grad_out.id >= 0:
            arg.grad_out.tensor = tensors[arg.grad_out.id]


def clear_tensor_refs(args: list[TensorRef]) -> Any:
    for arg in args:
        arg.tensor = None
        if arg.grad_in is not None:
            arg.grad_in.tensor = None
        if arg.grad_out is not None:
            arg.grad_out.tensor = None
    return arg


def gather_and_clear_primal_tensors(
    args: list[TensorRef],
    primal_in_tensors: list[torch.Tensor],
    primal_out_tensors: list[torch.Tensor],
) -> Any:
    for arg in args:
        if arg.last_access[0] in (AccessType.read, AccessType.readwrite):
            assert arg.tensor is not None
            primal_in_tensors.append(arg.tensor)
        if arg.last_access[0] in (AccessType.write, AccessType.readwrite):
            assert arg.tensor is not None
            primal_out_tensors.append(arg.tensor)


def assign_primal_and_grad_tensors(
    args: list[TensorRef],
    all_tensors: list[torch.Tensor],
    grad_in_tensors: list[torch.Tensor],
    grad_out_tensors: list[torch.Tensor],
) -> Any:
    for arg in args:
        if arg.id >= 0:
            arg.tensor = all_tensors[arg.id]
            if arg.last_access[0] in (AccessType.read, AccessType.readwrite):
                arg.grad_out = TensorRef(-1, torch.zeros_like(arg.tensor))
                grad_out_tensors.append(arg.grad_out.tensor)  # type: ignore
            if arg.last_access[0] in (AccessType.write, AccessType.readwrite):
                arg.grad_in = TensorRef(-1, grad_in_tensors.pop(0).contiguous())


def alloc_gradients(args: list[TensorRef], tensors: list[Optional[torch.Tensor]]) -> Any:
    for arg in args:
        if arg.tensor is not None and arg.tensor.requires_grad:
            grad = torch.zeros_like(arg.tensor)
            arg.grad_out = TensorRef(-1, grad)
            tensors.append(grad)
        else:
            tensors.append(None)


class TorchAutoGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        spy_function: Function,
        unpacked_args: tuple[Any, ...],
        unpacked_kwargs: dict[str, Any],
        tensor_refs: list[TensorRef],
        *tensors: torch.Tensor,
    ):
        print("fwds called")
        # Store data
        ctx.spy_function = spy_function
        ctx.unpacked_args = unpacked_args
        ctx.unpacked_kwargs = unpacked_kwargs
        ctx.tensor_refs = tensor_refs

        # Extract any tensors that were written to, and so should be treated as outputs
        primal_out_tensors = [
            cast(torch.Tensor, x.tensor)
            for x in tensor_refs
            if x.last_access[0] in (AccessType.write, AccessType.readwrite)
        ]

        # Mark all the outputs as dirty, so torch knows they may have changed
        # as a result of the forward pass
        ctx.mark_dirty(*primal_out_tensors)

        # Save all tensors.
        all_tensors = [x.tensor for x in tensor_refs if x.tensor is not None]
        ctx.save_for_backward(*all_tensors)

        # Clear all torch tensor references (PyTorch takes over at this point, and may
        # want to allocate new ones, so holding on to them can just cause excess memory usage)
        clear_tensor_refs(tensor_refs)

        # Return the outputs, so they get hooked into the torch auto-grad graph
        return tuple(primal_out_tensors)

    @staticmethod
    def backward(ctx: Any, *args: torch.Tensor):

        # Load parameters from context
        spy_function: FunctionNode = ctx.spy_function
        unpacked_args: tuple[Any, ...] = ctx.unpacked_args
        unpacked_kwargs: dict[str, Any] = ctx.unpacked_kwargs
        tensor_refs: list[TensorRef] = ctx.tensor_refs
        result_out_provided = "_result" in unpacked_kwargs
        all_tensors = list(ctx.saved_tensors)

        # Re-populate the primal tensor references and create/assign the gradient tensors
        grad_in_tensors: list[torch.Tensor] = list(args)
        grad_out_tensors: list[torch.Tensor] = []
        assign_primal_and_grad_tensors(
            tensor_refs,
            all_tensors,
            grad_in_tensors,
            grad_out_tensors,
        )

        # Get cuda stream and tell slangpy to use it
        cuda_stream_handle = NativeHandle.from_cuda_stream(torch.cuda.current_stream().cuda_stream)
        spy_function = spy_function.cuda_stream(cuda_stream_handle)

        # Check for a final tensor from the args, which would be the return value if there was one
        # This is only necessary if user did not supply an _result argument (if they did, the
        # assign_primal_and_grad_tensors function will have already set it up correctly).
        if not result_out_provided and len(grad_in_tensors) > 0:
            # Function returns a value but user didn't provide an _result argument.
            # Need to create a new TensorRef for the result, and pass it in using the _result argument.
            assert len(grad_in_tensors) == 1
            result_grad_tensor = grad_in_tensors[0].contiguous()
            result = TensorRef(-1, ctx.saved_tensors[-1])
            result.grad_in = TensorRef(-1, result_grad_tensor)
            spy_function.bwds(*unpacked_args, **unpacked_kwargs, _result=result)
        else:
            # Function either returns no value, or user provided an _result argument
            # so can just call it directly with the provided args.
            spy_function.bwds(*unpacked_args, **unpacked_kwargs)

        # Clear the tensors after passing to the function
        # Is this necessary? I have a feeling not doing so would break
        # calling bwds more than once.
        clear_tensor_refs(tensor_refs)

        # Return the gradients, with 4 'nones' to correspond to the first
        # 4 arguments of the forward function.
        res = (None, None, None, None) + tuple(grad_out_tensors)
        return res


class TorchFunction(torch.nn.Module):

    def __init__(self, function: FunctionNode):
        super().__init__()
        check_cuda_enabled(function.module.device)
        self.function: FunctionNode = function.return_type(TensorRef)

    def forward(self, *args: Any, **kwargs: Any):

        tensor_refs = []
        unpacked_args = unpack_refs_and_args(tensor_refs, *args)
        unpacked_kwargs = unpack_refs_and_kwargs(tensor_refs, **kwargs)
        result_out_provided = "_result" in unpacked_kwargs

        cuda_stream_handle = NativeHandle.from_cuda_stream(torch.cuda.current_stream().cuda_stream)

        # Call the function with the unpacked args and kwargs on the current CUDA stream
        result = self.function.cuda_stream(cuda_stream_handle).call(
            *unpacked_args, **unpacked_kwargs
        )

        # Read back torch tensor if result is TensorRef
        if isinstance(result, TensorRef):
            assert result.tensor is not None
            if not result_out_provided:
                tensor_refs.append(result)
            result = cast(torch.Tensor, result.tensor)

        # Extract all tensors that should be treated as inputs to the auto-grad function
        # i.e. ones that SlangPy marked as 'read' or 'readwrite' during the primal call.
        # These can then be passed as arguments to the auto-grad function so they get hooked
        # into the torch auto-grad graph.
        primal_in_tensors = [
            x.tensor
            for x in tensor_refs
            if x.last_access[0] in (AccessType.read, AccessType.readwrite)
        ]

        # Call the dummy auto-grad apply function, which critically takes the primal input list
        # as arguments and returns the primal output list as results
        TorchAutoGradFunction.apply(
            self.function,
            unpacked_args,
            unpacked_kwargs,
            tensor_refs,
            *primal_in_tensors,
        )

        # Return the single result
        return result

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

    def as_func(self) -> "TorchFunction":
        """
        Typing helper to cast the function to a function (i.e. a no-op)
        """
        return self

    def as_struct(self) -> "TorchStruct":
        """
        Typing helper to detect attempting to treat a function as a struct.
        """
        raise ValueError("Cannot convert a function to a struct")
