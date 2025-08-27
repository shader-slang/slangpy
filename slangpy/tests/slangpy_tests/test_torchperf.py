# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from pathlib import Path
import time
import pytest
from slangpy import DeviceType, Device, Module, Tensor, Function
from slangpy.core.native import NativeCallDataCache, SignatureBuilder, TensorRef
import sys

sys.path.append(str(Path(__file__).parent))
import helpers

try:
    import torch
except ImportError:
    pytest.skip("Pytorch not installed", allow_module_level=True)

# Skip all tests in this file if running on MacOS
if sys.platform == "darwin":
    pytest.skip("PyTorch requires CUDA, that is not available on macOS", allow_module_level=True)

TEST_CODE = """
[Differentiable]
float square(float x) {
    return x * x;
}
"""

TEST_CODE_BINDFUL = """
[Differentiable]
void square_bindful(int call_id, StructuredBuffer<float> x, RWStructuredBuffer<float> x2) {
    auto val = x[call_id];
    x2[call_id] = val * val;
}
"""

TEST_CODE_BINDLESS = """
[Differentiable]
void square_bindless(int call_id, StructuredBuffer<float>.Handle x, RWStructuredBuffer<float>.Handle x2) {
    auto val = x[call_id];
    x2[call_id] = val * val;
}
"""

TEST_CODE_PTR = """
[Differentiable]
void square_ptr(int call_id, float* x, float* x2) {
    auto val = x[call_id];
    x2[call_id] = val * val;
}
"""

DEVICE_TYPES = helpers.DEFAULT_DEVICE_TYPES
# Metal does not support torch integration
if DeviceType.metal in DEVICE_TYPES:
    DEVICE_TYPES.remove(DeviceType.metal)

ROUNDS = 1
ITERATIONS = 5000
BUFFER_SIZE = 100000000


def run_slangpy_only_test(device: Device):
    func = helpers.create_function_from_module(device, "square", TEST_CODE)

    # Allocate buffer + initial run
    x2 = Tensor.empty(device, (BUFFER_SIZE,), dtype="float")
    x2res = Tensor.empty(device, (BUFFER_SIZE,), dtype="float")
    func(x2, _result=x2res)

    time.sleep(0.1)

    # Capture time
    av = 0
    for _ in range(ROUNDS):
        start = time.time()
        for _ in range(ITERATIONS):
            func(x2, _result=x2res)
        end = time.time()
        av += end - start
        device.wait_for_idle()

    # Return average
    return 1000 * av / ROUNDS


def run_torch_only_test():

    # Allocate random tensor + initial run
    x = torch.rand((BUFFER_SIZE,), dtype=torch.float32, device="cuda")
    x = x * x

    time.sleep(0.1)

    # Capture time
    av = 0
    for _ in range(ROUNDS):
        start = time.time()
        for _ in range(ITERATIONS):
            x = x * x
        end = time.time()
        av += end - start
        torch.cuda.synchronize()

    # Return average
    return 1000 * av / ROUNDS


def run_slangpy_and_torch_test(device: Device):
    func = helpers.create_function_from_module(device, "square", TEST_CODE)

    # Allocate random tensor + initial run
    x = torch.rand((BUFFER_SIZE,), dtype=torch.float32, device="cuda")
    x = func(x)

    time.sleep(0.1)

    # Capture time
    av = 0
    for _ in range(ROUNDS):
        start = time.time()
        for _ in range(ITERATIONS):
            x = func(x)
        end = time.time()
        av += end - start
        device.wait_for_idle()

    # Return average
    return 1000 * av / ROUNDS


@pytest.mark.parametrize("device_type", [DeviceType.cuda])
def test_add_values(device_type: DeviceType):

    device = helpers.get_torch_device(device_type)

    time.sleep(0.25)

    print("Running torch only test...")
    av2 = run_torch_only_test()
    print(f"Torch only: {av2:.4f}ms")
    time.sleep(0.25)

    print("Running slangpy only test...")
    av = run_slangpy_only_test(device)
    print(f"SlangPy only: {av:.4f}ms")
    time.sleep(0.25)

    print("Running slangpy and torch test...")
    av3 = run_slangpy_and_torch_test(device)
    print(f"SlangPy + Torch: {av3:.4f}ms")


if __name__ == "__main__":
    input("Press Enter to run the test...")
    test_add_values(DeviceType.cuda)
