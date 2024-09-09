import pytest
from kernelfunctions.types import NDDifferentiableBuffer
from kernelfunctions.codegen import CodeGen
from kernelfunctions.tests import helpers
from kernelfunctions.tests.code_gen_test_helpers import dot_product
from kernelfunctions.utils import diffPair, floatDiffPair
from kernelfunctions.backend import DeviceType, float3, float1


def code(cg: tuple[CodeGen, CodeGen, CodeGen]):
    return cg[0].finish(trampoline=True).strip(), cg[1].finish(trampoline=True).strip(), cg[2].finish(trampoline=True).strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_scalar(device_type: DeviceType):

    # really simple test case emulating slang function that takes
    # 2 x float3 and returns a float.
    (prim, bwds, fwds) = code(dot_product(device_type, float3(), float3(), None))

    # primitive call should pass in 2 vectors and output a float
    assert prim == """
[Differentiable]
void _trampoline(out float _result, no_diff in vector<float,3> a, no_diff in vector<float,3> b)
{
    _result = dot_product(a, b);
}
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_diff_pairs(device_type: DeviceType):

    # use a scalar ref should get a structured buffer for output
    (prim, bwds, fwds) = code(dot_product(device_type,
                                          diffPair(p=float3(), d=float3()),
                                          diffPair(p=float3(), d=float3()),
                                          floatDiffPair()))

    # primitive call should pass in 2 vectors and output a float
    print(prim)
    assert prim == """
[Differentiable]
void _trampoline(out float _result, in vector<float,3> a, in vector<float,3> b)
{
    _result = dot_product(a, b);
}
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_buffer(device_type: DeviceType):

    device = helpers.get_device(device_type)
    buffer_0 = NDDifferentiableBuffer(
        device=device,
        element_type=float3,
        element_count=50,
        requires_grad=True
    )
    buffer_1 = NDDifferentiableBuffer(
        device=device,
        element_type=float3,
        element_count=50,
        requires_grad=False
    )
    buffer_2 = NDDifferentiableBuffer(
        device=device,
        element_type=float1,
        element_count=50,
        requires_grad=True
    )

    # use a scalar ref should get a structured buffer for output^
    (prim, bwds, fwds) = code(dot_product(device_type,
                                          buffer_0,
                                          buffer_1,
                                          buffer_2))

    # prim call should have
    # - 2 1D read-only tensor buffers of type float3
    # - 1 1D read-write tensor buffer of type float
    assert prim == """
[Differentiable]
void _trampoline(out float _result, in vector<float,3> a, no_diff in vector<float,3> b)
{
    _result = dot_product(a, b);
}
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_ND_buffer(device_type: DeviceType):

    device = helpers.get_device(device_type)
    buffer_0 = NDDifferentiableBuffer(
        device=device,
        element_type=float3,
        shape=(50, 10, 4),
        requires_grad=True
    )
    buffer_1 = float3(10.0, 10.0, 10.0)
    buffer_2 = floatDiffPair()

    # use a scalar ref should get a structured buffer for output
    (prim, bwds, fwds) = code(dot_product(device_type,
                                          buffer_0,
                                          buffer_1,
                                          buffer_2))

    # prim call should have
    # - 2 1D read-only tensor buffers of type float3
    # - 1 1D read-write tensor buffer of type float
    assert prim == """
[Differentiable]
void _trampoline(out float _result, in vector<float,3> a, no_diff in vector<float,3> b)
{
    _result = dot_product(a, b);
}
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_ND_buffer_input_transform(device_type: DeviceType):

    device = helpers.get_device(device_type)
    buffer_0 = NDDifferentiableBuffer(
        device=device,
        element_type=float3,
        shape=(50, 10, 4),
        requires_grad=True
    )
    buffer_1 = float3(10.0, 10.0, 10.0)
    buffer_2 = floatDiffPair()

    # use a scalar ref should get a structured buffer for output
    (prim, bwds, fwds) = code(dot_product(device_type,
                                          buffer_0,
                                          buffer_1,
                                          buffer_2,
                                          opts={
                                              "input_transform": {
                                                  'a': (1, 2, 0)
                                              }
                                          }
                                          ))

    # prim call should have
    # - 2 1D read-only tensor buffers of type float3
    # - 1 1D read-write tensor buffer of type float
    assert prim == """
[Differentiable]
void _trampoline(out float _result, in vector<float,3> a, no_diff in vector<float,3> b)
{
    _result = dot_product(a, b);
}
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_ND_buffer_output_transform(device_type: DeviceType):

    device = helpers.get_device(device_type)
    buffer_0 = NDDifferentiableBuffer(
        device=device,
        element_type=float3,
        shape=(10,),
        requires_grad=True
    )
    buffer_1 = NDDifferentiableBuffer(
        device=device,
        element_type=float3,
        shape=(5,),
        requires_grad=True
    )
    buffer_2 = NDDifferentiableBuffer(
        device=device,
        element_type=float1,
        shape=(10, 5),
        requires_grad=True
    )

    # use a scalar ref should get a structured buffer for output
    (prim, bwds, fwds) = code(dot_product(device_type,
                                          buffer_0,
                                          buffer_1,
                                          buffer_2,
                                          opts={
                                              "output_transform": {
                                                  'a': (0,),
                                                  'b': (1,)
                                              }
                                          }))

    # prim call should have
    # - 2 1D read-only tensor buffers of type float3, reading from different call ids
    # - 1 1D read-write tensor buffer of type float, writing using both call ids
    assert prim == """
[Differentiable]
void _trampoline(out float _result, in vector<float,3> a, in vector<float,3> b)
{
    _result = dot_product(a, b);
}
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_buffer_soa(device_type: DeviceType):

    device = helpers.get_device(device_type)
    buffer_0 = NDDifferentiableBuffer(
        device=device,
        element_type=float,
        element_count=50,
        requires_grad=True
    )
    buffer_1 = NDDifferentiableBuffer(
        device=device,
        element_type=float3,
        element_count=50,
        requires_grad=False
    )
    buffer_2 = NDDifferentiableBuffer(
        device=device,
        element_type=float1,
        element_count=50,
        requires_grad=True
    )

    # use a scalar ref should get a structured buffer for output
    (prim, bwds, fwds) = code(dot_product(device_type,
                                          {
                                              'x': buffer_0,
                                              'y': floatDiffPair(),
                                              'z': 2.0
                                          },
                                          buffer_1,
                                          buffer_2))

    # prim call should have
    # - 1D read only tensor buffer of floats for a.x
    # - floats for a.y and a.z
    # - 1D read only tensor buffer of float3s for b
    # - 1 1D read-write tensor buffer of type float
    assert prim == """
[Differentiable]
void _trampoline(out float _result, in vector<float,3> a, no_diff in vector<float,3> b)
{
    _result = dot_product(a, b);
}
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_from_buffer(device_type: DeviceType):

    device = helpers.get_device(device_type)
    buffer_0 = NDDifferentiableBuffer(
        device=device,
        element_type=float3,
        shape=(50, 10, 4),
        requires_grad=True
    )
    buffer_1 = NDDifferentiableBuffer(
        device=device,
        element_type=float3,
        shape=(50, 1, 4),
        requires_grad=True
    )
    buffer_2 = floatDiffPair()

    # use a scalar ref should get a structured buffer for output
    (prim, bwds, fwds) = code(dot_product(device_type,
                                          buffer_0,
                                          buffer_1,
                                          buffer_2))

    # prim call should have
    # - 2 1D read-only tensor buffers of type float3
    #   - a reads all call ids, b's middle component is always 0
    # - 1 1D read-write tensor buffer of type float
    assert prim == """
[Differentiable]
void _trampoline(out float _result, in vector<float,3> a, in vector<float,3> b)
{
    _result = dot_product(a, b);
}
""".strip()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
