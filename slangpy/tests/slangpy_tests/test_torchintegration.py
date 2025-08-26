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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
