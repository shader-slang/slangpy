# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
from typing import TYPE_CHECKING, Any, Iterator, Optional

from slangpy import CommandBuffer, Device, DeviceType

if TYPE_CHECKING:
    import torch

#: Name of the attribute that ties retained command buffers to a ``torch.cuda.CUDAGraph``.
_GRAPH_ATTRIBUTE = "_slangpy_command_buffers"


class _RetainedCommandBuffers:
    """
    Command buffers that were recorded into a CUDA graph, kept alive for as long as this object.

    The graph's nodes point into the command buffers' memory. This object is stored on the graph,
    so it is released when the graph is. Before letting the command buffers go it waits for the
    device to be idle, because a launch of the graph may still be running.
    """

    def __init__(self, device: Device, command_buffers: list[CommandBuffer]):
        super().__init__()
        self.device = device
        self.command_buffers = command_buffers

    def __del__(self):
        if not self.command_buffers:
            return
        # Outside a recording, a launch of the graph may still be running, so wait for it. During
        # a recording that wait is not possible, and not needed: capture_torch_graph waited for
        # the device before the recording started, and nothing has run since.
        try:
            import torch

            if not torch.cuda.is_current_stream_capturing():
                self.device.wait_for_idle()
        except Exception:
            # At interpreter exit torch can no longer be imported and the device may already be
            # closed. Its memory is gone with it, so there is nothing left to wait for.
            pass
        self.command_buffers.clear()


@contextlib.contextmanager
def capture_torch_graph(
    device: Device,
    graph: Optional["torch.cuda.CUDAGraph"] = None,
    **graph_args: Any,
) -> Iterator["torch.cuda.CUDAGraph"]:
    """
    Record PyTorch operations and SlangPy calls into a CUDA graph.

    Use it in place of ``torch.cuda.graph``::

        with spy.capture_torch_graph(device) as graph:
            y = module.my_function(x)   # x is a torch tensor
            loss = y.sum()
        graph.replay()

    The command buffers SlangPy submits inside the block are recorded into the graph instead of
    being run. The graph refers to their memory every time it is replayed, so they are attached to
    the graph and released together with it.

    Only SlangPy calls that take a torch tensor are recorded: those run on PyTorch's current CUDA
    stream, which is the stream being captured. A call made for the first time inside the block
    has its kernel generated and compiled there, which costs time while recording only.

    The graph is recorded in CUDA's relaxed capture mode, so objects that own GPU memory,
    including earlier graphs, may be released inside the block. Pass ``capture_error_mode`` to
    choose another mode.

    :param device: The SlangPy device the calls run on. It must be a CUDA device that shares
        PyTorch's context (see ``create_torch_device``).
    :param graph: The graph to record into. A new ``torch.cuda.CUDAGraph`` if not given.
    :param graph_args: Further arguments for ``torch.cuda.graph``, such as ``pool`` or ``stream``.
        ``capture_error_mode`` defaults to ``"relaxed"``.
    :return: The graph, ready to ``replay()`` once the block exits.
    """
    import torch

    if device.desc.type != DeviceType.cuda:
        raise ValueError(
            f"capture_torch_graph needs a CUDA device, got {device.desc.type}. "
            "Calls on other device types do not run on a CUDA stream and cannot be recorded."
        )
    if graph is None:
        graph = torch.cuda.CUDAGraph()

    # Set relaxed capture mode, which allows the default context to free memory such as garbage
    # collected buffers whilst recording the graph.
    graph_args.setdefault("capture_error_mode", "relaxed")

    # If a prior graph is garbage collected while this one is being recorded, it may still be
    # in flight. We'd need to wait_for_idle to do so, but wait_for_idle is illegal if a
    # device is capturing! To work around, ensure work is finished before starting the new capture.
    device.wait_for_idle()

    # Begin retaining command buffers for the new graph. This ensures any resources they use
    # are kept alive.
    device.begin_retain_command_buffers()
    try:
        # Create graph + yield
        with torch.cuda.graph(graph, **graph_args):
            yield graph
    finally:
        # Acquire the retained buffers and associate them with the graph. If the graph already
        # holds retained buffers, keep those alive too: its earlier nodes still point into them.
        command_buffers = device.end_retain_command_buffers()
        retained = getattr(graph, _GRAPH_ATTRIBUTE, None)
        if isinstance(retained, _RetainedCommandBuffers) and retained.device is device:
            retained.command_buffers.extend(command_buffers)
        else:
            setattr(graph, _GRAPH_ATTRIBUTE, _RetainedCommandBuffers(device, command_buffers))
