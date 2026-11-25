# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc
import sys
import weakref

import pytest

import slangpy as spy
from slangpy import DeviceType, Module
from slangpy.testing import helpers

try:
    import torch
except ImportError:
    pytest.skip("Pytorch not installed", allow_module_level=True)

if sys.platform == "darwin":
    pytest.skip("PyTorch requires CUDA, that is not available on macOS", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

requires_cuda_device = pytest.mark.skipif(
    DeviceType.cuda not in helpers.DEFAULT_DEVICE_TYPES, reason="CUDA device type not enabled"
)


def load_test_module() -> Module:
    device = helpers.get_torch_device(DeviceType.cuda)
    return Module.load_from_file(device, "test_torchintegration.slang")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_retain_command_buffers(device_type: DeviceType) -> None:
    device = helpers.get_device(device_type)

    def submit() -> spy.CommandBuffer:
        command_buffer = device.create_command_encoder().finish()
        device.submit_command_buffer(command_buffer)
        return command_buffer

    assert not device.is_retaining_command_buffers()
    submit()

    device.begin_retain_command_buffers()
    assert device.is_retaining_command_buffers()
    first = submit()
    device.begin_retain_command_buffers()
    second = submit()
    inner = device.end_retain_command_buffers()
    assert device.is_retaining_command_buffers()
    third = submit()
    outer = device.end_retain_command_buffers()
    assert not device.is_retaining_command_buffers()

    # An inner scope returns its own submissions; the enclosing scope also keeps them, in the
    # order the scopes handed them over.
    assert inner == [second]
    assert outer == [first, second, third]

    with pytest.raises(Exception, match="without a matching"):
        device.end_retain_command_buffers()
    device.wait_for_idle()


@requires_cuda_device
def test_capture_matches_eager() -> None:
    module = load_test_module()
    device = module.device

    a = torch.arange(1024, dtype=torch.float32, device="cuda")
    b = torch.full((1024,), 2.0, device="cuda")

    def step() -> torch.Tensor:
        # Two SlangPy calls with Torch operations between and after them.
        summed = module.add(a, b)
        return module.add(summed * 3.0, a) + 1.0

    # Compile the kernels and warm caches outside the recording.
    step()

    with spy.capture_torch_graph(device) as graph:
        result = step()
    assert not device.is_retaining_command_buffers()

    # The recording itself runs nothing: replay, then again with new inputs written in place.
    for scale in (1.0, 5.0):
        a.copy_(torch.arange(1024, dtype=torch.float32, device="cuda") * scale)
        b.fill_(scale + 1.0)
        graph.replay()
        recorded = result.clone()
        torch.testing.assert_close(recorded, step())

    # Work submitted after the recording is tracked and retired as usual.
    torch.testing.assert_close(module.add(a, b), a + b)
    device.wait_for_idle()


@requires_cuda_device
def test_capture_while_earlier_work_is_in_flight() -> None:
    module = load_test_module()
    device = module.device
    a = torch.ones(4096, device="cuda")
    count = 200

    # Leave eager submissions in flight, then record more calls than the command buffer
    # pool holds. The pool runs dry during the recording, which makes the device look for finished
    # command buffers (querying streams, freeing memory) while the stream is being captured.
    for _ in range(count):
        module.add(a, a)

    with spy.capture_torch_graph(device) as graph:
        result = a
        for _ in range(count):
            result = module.add(result, a)

    assert len(getattr(graph, "_slangpy_command_buffers").command_buffers) == count
    for _ in range(3):
        graph.replay()
    torch.testing.assert_close(result, torch.full((4096,), float(count + 1), device="cuda"))
    device.wait_for_idle()


@requires_cuda_device
def test_command_buffers_live_with_graph() -> None:
    module = load_test_module()
    device = module.device
    a = torch.ones(256, device="cuda")
    module.add(a, a)

    with spy.capture_torch_graph(device) as graph:
        result = module.add(module.add(a, a), a)

    retained = getattr(graph, "_slangpy_command_buffers")
    assert len(retained.command_buffers) == 2
    released = weakref.ref(retained)
    del retained

    graph.replay()
    torch.testing.assert_close(result, torch.full((256,), 3.0, device="cuda"))

    # Dropping the graph drops the command buffers recorded into it.
    del graph
    gc.collect()
    assert released() is None

    # The device carries on normally.
    torch.testing.assert_close(module.add(a, a), a * 2.0)
    device.wait_for_idle()


@requires_cuda_device
def test_replace_graph_while_capturing() -> None:
    module = load_test_module()
    device = module.device
    a = torch.ones(64, device="cuda")
    module.add(a, a)

    # Each recording drops the previous graph, and the command buffers attached to it, after the
    # new recording has started.
    for scale in (1.0, 2.0, 3.0):
        a.fill_(scale)
        with spy.capture_torch_graph(device) as graph:
            result = module.add(module.add(a, a), a)
        graph.replay()
        torch.testing.assert_close(result, torch.full((64,), 3.0 * scale, device="cuda"))
    device.wait_for_idle()


@requires_cuda_device
def test_capture_into_existing_graph_with_arguments() -> None:
    module = load_test_module()
    device = module.device
    a = torch.ones(64, device="cuda")
    module.add(a, a)

    graph = torch.cuda.CUDAGraph()
    with spy.capture_torch_graph(device, graph, capture_error_mode="global") as captured:
        assert captured is graph
        result = module.add(a, a)
    graph.replay()
    torch.testing.assert_close(result, a * 2.0)


@requires_cuda_device
def test_capture_survives_exception() -> None:
    module = load_test_module()
    device = module.device
    a = torch.ones(64, device="cuda")
    module.add(a, a)

    with pytest.raises(RuntimeError, match="stop here"):
        with spy.capture_torch_graph(device) as graph:
            module.add(a, a)
            raise RuntimeError("stop here")

    assert not device.is_retaining_command_buffers()
    # The graph is left as PyTorch left it, with what was recorded before the error kept alive.
    assert len(getattr(graph, "_slangpy_command_buffers").command_buffers) == 1

    # A later recording and eager calls still work. Rebinding the name drops the failed graph
    # while the new one is being recorded.
    with spy.capture_torch_graph(device) as graph:
        result = module.add(a, a)
    graph.replay()
    torch.testing.assert_close(result, a * 2.0)
    torch.testing.assert_close(module.add(a, a), a * 2.0)


def test_capture_needs_cuda_device() -> None:
    other = [
        t for t in helpers.DEFAULT_DEVICE_TYPES if t not in (DeviceType.cuda, DeviceType.metal)
    ]
    if not other:
        pytest.skip("No non-CUDA device type enabled")
    device = helpers.get_torch_device(other[0])
    with pytest.raises(ValueError, match="needs a CUDA device"):
        with spy.capture_torch_graph(device):
            pass
    assert not device.is_retaining_command_buffers()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
