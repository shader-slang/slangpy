# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys

from slangpy import DeviceType, Device, Module
from slangpy.core.native import NativeCallDataCache, SignatureBuilder, TensorRef
from slangpy.testing import helpers

try:
    import torch
except ImportError:
    pytest.skip("Pytorch not installed", allow_module_level=True)

# Skip all tests in this file if running on MacOS
if sys.platform == "darwin":
    pytest.skip("PyTorch requires CUDA, that is not available on macOS", allow_module_level=True)

TEST_CODE = """
import tensor;
[Differentiable]
float square(float x) {
    return x * x;
}
"""

DEVICE_TYPES = helpers.DEFAULT_DEVICE_TYPES
# Metal does not support torch integration
if DeviceType.metal in DEVICE_TYPES:
    DEVICE_TYPES.remove(DeviceType.metal)


def get_test_tensors(device: Device, N: int = 4):
    weights = torch.randn(
        (5, 8), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True
    )
    biases = torch.randn((5,), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True)
    x = torch.randn((8,), dtype=torch.float32, device=torch.device("cuda"), requires_grad=False)

    return weights, biases, x


def load_test_module(device_type: DeviceType):
    device = helpers.get_torch_device(device_type)
    return Module.load_from_file(device, "test_torchintegration.slang")


def compare_tensors(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape, f"Tensor shape {a.shape} does not match expected shape {b.shape}"
    err = torch.max(torch.abs(a - b)).item()
    assert err < 1e-4, f"Tensor deviates by {err} from reference"


@pytest.mark.parametrize(
    "pair",
    [
        (torch.empty((1,), dtype=torch.float32).cuda(), "D1,C2,B32,L1"),
        (torch.empty((1,), dtype=torch.float32, requires_grad=True).cuda(), "D1,C2,B32,L1"),
        (torch.empty((1,), dtype=torch.float16).cuda(), "D1,C2,B16,L1"),
        (torch.empty((1,), dtype=torch.int32).cuda(), "D1,C0,B32,L1"),
        (torch.empty((1,), dtype=torch.uint8).cuda(), "D1,C1,B8,L1"),
        (torch.empty((1, 1, 1), dtype=torch.uint8).cuda(), "D3,C1,B8,L1"),
    ],
)
def test_torch_signature(pair: tuple[torch.Tensor, str]):
    cd = NativeCallDataCache()
    sig = SignatureBuilder()
    cd.get_value_signature(sig, pair[0])
    assert sig.str == f"Tensor\n[torch,{pair[1]}]"

    ref = TensorRef(0, pair[0])
    sig = SignatureBuilder()
    cd.get_value_signature(sig, ref)
    assert sig.str.endswith(f"[torch,{pair[1]}]")


ADD_TESTS = [
    ("add", ()),
    ("add_vectors", (3,)),
    ("add_vectors_generic<4>", (4,)),
    ("add_arrays", (5,)),
]


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("extra_dims", [0, 1, 3])
@pytest.mark.parametrize(
    "func_and_shape", ADD_TESTS, ids=[f"{name}_{shape}" for name, shape in ADD_TESTS]
)
@pytest.mark.parametrize("result_mode", ["return", "pass", "out"])
def test_add_values(
    device_type: DeviceType,
    extra_dims: int,
    func_and_shape: tuple[str, tuple[int]],
    result_mode: str,
):

    module = load_test_module(device_type)

    func_name = func_and_shape[0]
    val_shape = func_and_shape[1]
    extra_shape = (5,) * extra_dims

    if len(extra_shape + val_shape) == 0:
        pytest.skip("No shape to test")

    a = torch.randn(
        extra_shape + val_shape,
        dtype=torch.float32,
        device=torch.device("cuda"),
        requires_grad=True,
    )
    b = torch.randn(
        extra_shape + val_shape,
        dtype=torch.float32,
        device=torch.device("cuda"),
        requires_grad=True,
    )

    if result_mode == "return":
        res = module[func_name](a, b)
    elif result_mode == "pass":
        res = torch.empty_like(a)
        module[func_name](a, b, _result=res)
    else:  # out
        res = torch.empty_like(a)
        if "<" in func_name:
            func_name = func_name.replace("<", "_out<")
        else:
            func_name += "_out"
        module[func_name](a, b, res)
    assert isinstance(res, torch.Tensor)

    test = a + b

    compare_tensors(a + b, res)

    # Not much to check for backwards pass of an 'add', but call it
    # so we at least catch any exceptions that fire.
    res.backward(torch.ones_like(res))


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("extra_dims", [0, 1, 3])
@pytest.mark.parametrize("func_and_shape", ADD_TESTS)
def test_add_values_fail(
    device_type: DeviceType, extra_dims: int, func_and_shape: tuple[str, tuple[int]]
):

    module = load_test_module(device_type)

    func_name = func_and_shape[0]
    val_shape = func_and_shape[1]
    if len(val_shape) == 0:
        pytest.skip("No shape to fail")

    extra_shape = (5,) * extra_dims

    val_shape = val_shape[0:-1] + (val_shape[-1] + 1,)

    a = torch.randn(extra_shape + val_shape, dtype=torch.float32, device=torch.device("cuda"))
    b = torch.randn(extra_shape + val_shape, dtype=torch.float32, device=torch.device("cuda"))

    with pytest.raises(ValueError, match="does not match expected shape"):
        res = module.add_vectors(a, b)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("extra_dims", [0, 1, 3])
def test_add_vectors_generic_explicit(device_type: DeviceType, extra_dims: int):
    pytest.skip("Crashes due to slang bug")

    module = load_test_module(device_type)

    extra_shape = (5,) * extra_dims

    a = torch.randn(extra_shape + (3,), dtype=torch.float32, device=torch.device("cuda"))
    b = torch.randn(extra_shape + (3,), dtype=torch.float32, device=torch.device("cuda"))

    # Can't currently infer generic vector from tensor shape, but explicit type map should work
    res = module.add_vectors_generic.map("float3", "float3")(a, b)
    assert isinstance(res, torch.Tensor)

    compare_tensors(a + b, res)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_polynomial(device_type: DeviceType):

    module = load_test_module(device_type)

    a = 2.0
    b = 4.0
    c = 1.0
    x = torch.randn((10,), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True)

    res = module.polynomial(a, b, c, x)
    assert isinstance(res, torch.Tensor)

    compare_tensors(a * x * x + b * x + c, res)

    res.backward(torch.ones_like(res))

    compare_tensors(2 * a * x + b, x.grad)  # type: ignore


# This test ensures that the PyTorch integration doesn't fail if re-using the
# same cached call data.
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_polynomial_multiple_calls(device_type: DeviceType):

    module = load_test_module(device_type)

    a = 2.0
    b = 4.0
    c = 1.0
    x = torch.randn((10,), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True)

    res = module.polynomial(a, b, c, x)
    assert isinstance(res, torch.Tensor)

    compare_tensors(a * x * x + b * x + c, res)

    res.backward(torch.ones_like(res))
    compare_tensors(2 * a * x + b, x.grad)  # type: ignore

    res2 = module.polynomial(a, b, c, x)
    assert isinstance(res2, torch.Tensor)

    x.grad.zero_()  # Reset gradients before the second call
    res2.backward(torch.ones_like(res2))
    compare_tensors(2 * a * x + b, x.grad)  # type: ignore


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_polynomial_outparam(device_type: DeviceType):

    module = load_test_module(device_type)

    a = 2.0
    b = 4.0
    c = 1.0
    x = torch.randn((10,), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True)
    res = torch.zeros_like(x)

    module.polynomial_out(a, b, c, x, res)

    compare_tensors(a * x * x + b * x + c, res)

    res.backward(torch.ones_like(res))

    compare_tensors(2 * a * x + b, x.grad)  # type: ignore


# Enable the vectors+arrays tests to reproduce compiler bugs
POLYNOMIAL_TESTS = [
    ("polynomial", ()),
    ("polynomial_vectors", (3,)),
    ("polynomial_arrays", (5,)),
]


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("extra_dims", [0, 1, 3])
@pytest.mark.parametrize(
    "func_and_shape",
    POLYNOMIAL_TESTS,
    ids=[f"{name}_{shape}" for name, shape in POLYNOMIAL_TESTS],
)
@pytest.mark.parametrize("result_mode", ["return", "pass", "out"])
def test_polynomials(
    device_type: DeviceType,
    extra_dims: int,
    func_and_shape: tuple[str, tuple[int]],
    result_mode: str,
):

    module = load_test_module(device_type)

    func_name = func_and_shape[0]
    val_shape = func_and_shape[1]
    extra_shape = (5,) * extra_dims

    if func_name == "polynomial_vectors":
        pytest.skip("Slang bug currently causing derivatives to return 0")

    if len(extra_shape + val_shape) == 0:
        pytest.skip("No shape to test")

    a = 2.0
    b = 4.0
    c = 1.0
    x = torch.randn(
        extra_shape + val_shape,
        dtype=torch.float32,
        device=torch.device("cuda"),
        requires_grad=True,
    )

    if result_mode == "return":
        res = module[func_name](a, b, c, x)
    elif result_mode == "pass":
        res = torch.empty_like(x)
        module[func_name](a, b, c, x, _result=res)
    else:  # out
        res = torch.empty_like(x)
        if "<" in func_name:
            func_name = func_name.replace("<", "_out<")
        else:
            func_name += "_out"
        module[func_name](a, b, c, x, res)
    assert isinstance(res, torch.Tensor)

    compare_tensors(a * x * x + b * x + c, res)

    res.backward(torch.ones_like(res))

    compare_tensors(2 * a * x + b, x.grad)  # type: ignore


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("extra_dims", [0, 1, 3])
def test_add_tensors(device_type: DeviceType, extra_dims: int):

    module = load_test_module(device_type)

    func_name = "add_tensors"
    val_shape = (8, 5)
    extra_shape = (5,) * extra_dims

    a = torch.randn(
        extra_shape + val_shape,
        dtype=torch.float32,
        device=torch.device("cuda"),
        requires_grad=True,
    )
    b = torch.randn(
        extra_shape + val_shape,
        dtype=torch.float32,
        device=torch.device("cuda"),
        requires_grad=True,
    )

    res = torch.empty_like(a)
    module[func_name](a, b, res)

    compare_tensors(a + b, res)

    # Should this work??
    # res.backward(torch.ones_like(res))


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_empty_tensor_null_data_ptr(device_type: DeviceType):
    """
    Test that tensors with null data pointers (e.g., zero-element tensors) are accepted.
    """
    module = load_test_module(device_type)

    # Create empty tensors - these have null data pointers
    input_tensor = torch.empty((0,), dtype=torch.float32, device=torch.device("cuda"))
    output_tensor = torch.empty((0,), dtype=torch.float32, device=torch.device("cuda"))

    # This should not crash - empty tensors with null data_ptr should be accepted
    module.copy_tensor(input_tensor, output_tensor)

    # Verify tensors are still empty
    assert input_tensor.numel() == 0
    assert output_tensor.numel() == 0


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_struct_tensor_vec2_1d(device_type: DeviceType):
    """
    Test Tensor<Vec2> reinterpretation using as_struct_tensor.
    A torch.Tensor of shape [N, 2] should be reinterpreted as [N] Vec2 structs.
    """
    import slangpy as spy

    module = load_test_module(device_type)

    N = 10
    # Create tensor of shape [N, 2] where last dim represents Vec2 {x, y}
    input_tensor = torch.randn((N, 2), dtype=torch.float32, device=torch.device("cuda"))
    output_tensor = torch.zeros((N, 2), dtype=torch.float32, device=torch.device("cuda"))

    # Wrap as struct tensors using the explicit helper
    struct_input = spy.as_struct_tensor(input_tensor, module.Vec2.struct, dims=1)
    struct_output = spy.as_struct_tensor(output_tensor, module.Vec2.struct, dims=1)

    # Call the function - slangpy will vectorize over the first dimension
    module.double_vec2(struct_input, struct_output)

    # Verify the result: each Vec2 should be doubled
    expected = input_tensor * 2.0
    compare_tensors(expected, output_tensor)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_struct_tensor_vec2_2d(device_type: DeviceType):
    """
    Test Tensor<Vec2> reinterpretation with 2D tensor.
    A torch.Tensor of shape [M, N, 2] should be reinterpreted as [M, N] Vec2 structs.
    """
    import slangpy as spy

    module = load_test_module(device_type)

    M, N = 4, 5
    # Create tensor of shape [M, N, 2] where last dim represents Vec2 {x, y}
    input_tensor = torch.randn((M, N, 2), dtype=torch.float32, device=torch.device("cuda"))
    output_tensor = torch.zeros((M, N, 2), dtype=torch.float32, device=torch.device("cuda"))

    # Wrap as struct tensors (2D tensors of Vec2 structs)
    struct_input = spy.as_struct_tensor(input_tensor, module.Vec2.struct, dims=2)
    struct_output = spy.as_struct_tensor(output_tensor, module.Vec2.struct, dims=2)

    # Call the function - slangpy will vectorize over both dimensions
    module.double_vec2(struct_input, struct_output)

    # Verify the result: each Vec2 should be doubled
    expected = input_tensor * 2.0
    compare_tensors(expected, output_tensor)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_struct_tensor_sum_vec2(device_type: DeviceType):
    """
    Test reading from Tensor<Vec2> and returning a scalar per element.
    """
    import slangpy as spy

    module = load_test_module(device_type)

    N = 8
    # Create tensor of shape [N, 2] where last dim represents Vec2 {x, y}
    input_tensor = torch.randn((N, 2), dtype=torch.float32, device=torch.device("cuda"))

    # Wrap as struct tensor
    struct_input = spy.as_struct_tensor(input_tensor, module.Vec2.struct, dims=1)

    # Call the function that sums x + y for each Vec2
    result = module.sum_vec2(struct_input)

    # Expected: sum of x and y for each element
    expected = input_tensor[:, 0] + input_tensor[:, 1]

    # Convert native tensor result to numpy for comparison
    result_np = result.to_numpy()
    expected_np = expected.cpu().numpy()

    err = abs(result_np - expected_np).max()
    assert err < 1e-4, f"Result deviates by {err} from reference"


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_struct_tensor_particle(device_type: DeviceType):
    """
    Test Tensor<Particle> reinterpretation.
    Particle struct has 6 floats (position: float3, velocity: float3).
    A torch.Tensor of shape [N, 6] should be reinterpreted as [N] Particle structs.
    """
    import slangpy as spy

    module = load_test_module(device_type)

    N = 5
    dt = 0.1
    # Create tensor of shape [N, 6] where last dim represents Particle
    # [px, py, pz, vx, vy, vz]
    input_tensor = torch.randn((N, 6), dtype=torch.float32, device=torch.device("cuda"))
    output_tensor = torch.zeros((N, 6), dtype=torch.float32, device=torch.device("cuda"))

    # Wrap as struct tensors
    struct_input = spy.as_struct_tensor(input_tensor, module.Particle.struct, dims=1)
    struct_output = spy.as_struct_tensor(output_tensor, module.Particle.struct, dims=1)

    # Call the function that updates particle positions
    module.update_particle(struct_input, dt, struct_output)

    # Expected: position = position + velocity * dt, velocity unchanged
    expected = input_tensor.clone()
    expected[:, 0:3] = input_tensor[:, 0:3] + input_tensor[:, 3:6] * dt
    compare_tensors(expected, output_tensor)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_struct_tensor_wrong_size_fails(device_type: DeviceType):
    """
    Test that passing a tensor with wrong last dimension size raises an error.
    Vec2 expects 2 floats (8 bytes), so shape [N, 3] should fail.
    """
    import slangpy as spy

    module = load_test_module(device_type)

    N = 10
    # Create tensor with wrong last dim size (3 instead of 2)
    input_tensor = torch.randn((N, 3), dtype=torch.float32, device=torch.device("cuda"))

    # This should raise an error because [N, 3] doesn't match Vec2's 2 floats
    with pytest.raises(ValueError):
        spy.as_struct_tensor(input_tensor, module.Vec2.struct, dims=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
