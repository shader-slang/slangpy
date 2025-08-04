# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Any, Optional, Union, cast
import torch

from slangpy.core.native import AccessType, unpack_refs_and_args, unpack_refs_and_kwargs
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


class TorchAutoGradHook(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        spy_function: Function,
        unpacked_args: tuple[Any, ...],
        unpacked_kwargs: dict[str, Any],
        tensor_refs: list[TensorRef],
        *tensors: torch.Tensor,
    ):
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

        # Extract read-write tensors (i.e. will be inputs+outputs that must be marked dirty)
        primal_inout_tensors = [
            cast(torch.Tensor, x.tensor)
            for x in tensor_refs
            if x.last_access[0] == AccessType.readwrite
        ]

        # Mark all the outputs as dirty, so torch knows they may have changed
        # as a result of the forward pass
        ctx.mark_dirty(*primal_inout_tensors)

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
