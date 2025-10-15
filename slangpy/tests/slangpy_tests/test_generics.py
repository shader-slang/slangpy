# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

from slangpy import DeviceType, TypeReflection
from slangpy.types.buffer import NDBuffer
from slangpy.types.tensor import Tensor
from slangpy.testing import helpers


def do_generic_test(
    device_type: DeviceType,
    container_type: str,
    slang_type_name: str,
    generic_args: str,
    buffer_type_name: str,
):
    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    module = helpers.create_module(
        device,
        f"""
{slang_type_name} get{generic_args}({slang_type_name} input) {{
    return input;
}}
""",
    )

    shape = (1024,)
    buffertype = module.layout.find_type_by_name(buffer_type_name)

    if container_type == "buffer":
        buffer = NDBuffer(device, dtype=buffertype, shape=shape)
        if buffer.cursor().element_type_layout.kind == TypeReflection.Kind.vector:
            helpers.write_ndbuffer_from_numpy(
                buffer,
                np.random.random(int(buffer.storage.size / 4)).astype(np.float32),
            )
        else:
            buffer.copy_from_numpy(
                np.random.random(int(buffer.storage.size / 4)).astype(np.float32)
            )

        results = module.get(buffer)
        assert results.dtype == buffer.dtype
        assert np.all(buffer.to_numpy() == results.to_numpy())
    elif container_type == "tensor":
        tensor = Tensor.empty(device, dtype=buffertype, shape=shape)
        results = module.get(tensor, _result="tensor")
        assert results.dtype == tensor.dtype
        assert np.all(tensor.to_numpy() == results.to_numpy())


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("container_type", ["buffer", "tensor"])
@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_generic_vector(device_type: DeviceType, container_type: str, dim: int):
    do_generic_test(device_type, container_type, "vector<float,N>", "<let N: int>", f"float{dim}")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("container_type", ["buffer", "tensor"])
@pytest.mark.parametrize("dim", [2])
def test_generic_array(device_type: DeviceType, container_type: str, dim: int):
    do_generic_test(device_type, container_type, "float[N]", "<let N: int>", f"float[{dim}]")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("container_type", ["buffer", "tensor"])
@pytest.mark.parametrize("dim", [2])
def test_generic_2d_array(device_type: DeviceType, container_type: str, dim: int):
    do_generic_test(
        device_type,
        container_type,
        "float[N][N]",
        "<let N: int>",
        f"float[{dim}][{dim}]",
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("container_type", ["buffer", "tensor"])
@pytest.mark.parametrize("dim", [2])
def test_generic_matrix(device_type: DeviceType, container_type: str, dim: int):
    do_generic_test(
        device_type,
        container_type,
        "matrix<float,N,N>",
        "<let N: int>",
        f"float{dim}x{dim}",
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("explicit", [True, False])
def test_arithmetic_generic_arguments_bug(device_type: DeviceType, explicit: bool):
    # This test reproduces a Slang issue: https://github.com/shader-slang/slang/issues/6463
    # Attempting to resolve a generic with arithmetic arguments fails. When the test is
    # run with 'explicit' specialization, we generate and succesfully find the function.
    # When we rely on Slang to solve the generic, it currently fails.

    CODE = """
interface IModule<int N, int M>
{
    float[M] eval(float[N] x);
}

struct LinearLayer<int N, int M> : IModule<N, M>
{
    float[M] eval(float[N] x)
    {
        float[M] result;
        [ForceUnroll]
        for (int i = 0; i < M; ++i) result[i] = 0.f;
        return result;
    }
};

float trainMaterial<
    int numLatents,
    Encoder : IModule<12, numLatents>,
    Decoder : IModule<numLatents+6, 3> // <---- here!
>
(Encoder encoder, Decoder decoder)
{
    return 0.f;
}
"""

    device = helpers.get_device(device_type)
    module = helpers.create_module(device, CODE)

    class LinearLayer:
        def __init__(self, n: int = 0, m: int = 0):
            super().__init__()
            self.n = n
            self.m = m

        def get_this(self):
            return {"_type": f"LinearLayer<{self.n}, {self.m}> "}

    encoder = LinearLayer(12, 8)
    decoder = LinearLayer(8 + 6, 3)

    if explicit:
        func_name = f"trainMaterial<{encoder.m}, {encoder.get_this()['_type']}, {decoder.get_this()['_type']}>"
        func = module.find_function(func_name)
        assert func is not None
        func(encoder, decoder)
    else:
        pytest.skip("Fails due to https://github.com/shader-slang/slang/issues/6463")
        module.trainMaterial(encoder, decoder)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
