import pytest
import sgl
from kernelfunctions.buffer import StructuredBuffer
from kernelfunctions.codegen import CodeGen
from kernelfunctions.tests import helpers
from kernelfunctions.tests.code_gen_test_helpers import dot_product
from kernelfunctions.utils import diffPair, floatDiffPair, floatRef


def code(cg: tuple[CodeGen, CodeGen, CodeGen]):
    return cg[0].finish(input_load_store=True).strip(), cg[1].finish(input_load_store=True).strip(), cg[2].finish(input_load_store=True).strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_scalar(device_type: sgl.DeviceType):

    # really simple test case emulating slang function that takes
    # 2 x float3 and returns a float.
    (prim, bwds, fwds) = code(dot_product(device_type, sgl.float3(), sgl.float3(), None))

    # primitive call should pass in 2 vectors and output a float
    # print(prim)
    assert prim == """
void load_a_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.a_primal;
}
void load_b_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.b_primal;
}
void store__result_primal(int[] call_id, in float val)
{
    call_data._result_primal[0] = val;
}
""".strip()

    # bwds call is a bit pointless - no arguments are differentiable,
    # we just end up loading the 2 primals
    # print(bwds)
    assert bwds == """
void load_a_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.a_primal;
}
void load_b_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.b_primal;
}
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_diff_pairs(device_type: sgl.DeviceType):

    # use a scalar ref should get a structured buffer for output
    (prim, bwds, fwds) = code(dot_product(device_type,
                                          diffPair(p=sgl.float3(), d=sgl.float3()),
                                          diffPair(p=sgl.float3(), d=sgl.float3()),
                                          floatDiffPair()))

    # primitive call should pass in 2 vectors and output a float
    assert prim == """
void load_a_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.a_primal;
}
void load_b_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.b_primal;
}
void store__result_primal(int[] call_id, in float val)
{
    call_data._result_primal[0] = val;
}
""".strip()

    # bwds call should now have:
    # - primal inputs and derivative outputs for a and b
    # - derivative input for result
    assert bwds == """
void load_a_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.a_primal;
}
void store_a_derivative(int[] call_id, in vector<float,3> val)
{
    call_data.a_derivative[0] = val;
}
void load_b_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.b_primal;
}
void store_b_derivative(int[] call_id, in vector<float,3> val)
{
    call_data.b_derivative[0] = val;
}
void load__result_derivative(int[] call_id, out float val)
{
    val = call_data._result_derivative;
}
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_buffer(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    buffer_0 = StructuredBuffer(
        device=device,
        element_type=sgl.float3,
        element_count=50,
        requires_grad=True
    )
    buffer_1 = StructuredBuffer(
        device=device,
        element_type=sgl.float3,
        element_count=50,
        requires_grad=False
    )
    buffer_2 = StructuredBuffer(
        device=device,
        element_type=sgl.float1,
        element_count=50,
        requires_grad=True
    )

    # use a scalar ref should get a structured buffer for output
    (prim, bwds, fwds) = code(dot_product(device_type,
                                          buffer_0,
                                          buffer_1,
                                          buffer_2))

    # prim call should have
    # - 2 1D read-only tensor buffers of type float3
    # - 1 1D read-write tensor buffer of type float
    assert prim == """
void load_a_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.a_primal[{call_id[0]}];
}
void load_b_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.b_primal[{call_id[0]}];
}
void store__result_primal(int[] call_id, in float val)
{
    call_data._result_primal[{call_id[0]}] = val;
}
""".strip()

    # bwds call should now have:
    # - read-only primal buffer for a
    # - read-write derivative buffer for a
    # - read-only primal buffer for b (it was not differentiable)
    # - read-only derivative buffer for result (it was only an output)
    assert bwds == """
void load_a_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.a_primal[{call_id[0]}];
}
void store_a_derivative(int[] call_id, in vector<float,3> val)
{
    call_data.a_derivative[{call_id[0]}] = val;
}
void load_b_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.b_primal[{call_id[0]}];
}
void load__result_derivative(int[] call_id, out float val)
{
    val = call_data._result_derivative[{call_id[0]}];
}
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_ND_buffer(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    buffer_0 = StructuredBuffer(
        device=device,
        element_type=sgl.float3,
        shape=(50, 10, 4),
        requires_grad=True
    )
    buffer_1 = sgl.float3(10.0, 10.0, 10.0)
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
void load_a_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.a_primal[{call_id[0],call_id[1],call_id[2]}];
}
void load_b_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.b_primal;
}
void store__result_primal(int[] call_id, in float val)
{
    call_data._result_primal[0] = val;
}
""".strip()

    # bwds call should now have:
    # - read-only primal buffer for a
    # - read-write derivative buffer for a
    # - read-only primal buffer for b (it was not differentiable)
    # - read-only derivative buffer for result (it was only an output)
    assert bwds == """
void load_a_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.a_primal[{call_id[0],call_id[1],call_id[2]}];
}
void store_a_derivative(int[] call_id, in vector<float,3> val)
{
    call_data.a_derivative[{call_id[0],call_id[1],call_id[2]}] = val;
}
void load_b_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.b_primal;
}
void load__result_derivative(int[] call_id, out float val)
{
    val = call_data._result_derivative;
}
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_ND_buffer_input_transform(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    buffer_0 = StructuredBuffer(
        device=device,
        element_type=sgl.float3,
        shape=(50, 10, 4),
        requires_grad=True
    )
    buffer_1 = sgl.float3(10.0, 10.0, 10.0)
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
    print(prim)
    assert prim == """
void load_a_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.a_primal[{call_id[1],call_id[2],call_id[0]}];
}
void load_b_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.b_primal;
}
void store__result_primal(int[] call_id, in float val)
{
    call_data._result_primal[0] = val;
}
""".strip()

    # bwds call should now have:
    # - read-only primal buffer for a
    # - read-write derivative buffer for a
    # - read-only primal buffer for b (it was not differentiable)
    # - read-only derivative buffer for result (it was only an output)
    assert bwds == """
void load_a_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.a_primal[{call_id[1],call_id[2],call_id[0]}];
}
void store_a_derivative(int[] call_id, in vector<float,3> val)
{
    call_data.a_derivative[{call_id[1],call_id[2],call_id[0]}] = val;
}
void load_b_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.b_primal;
}
void load__result_derivative(int[] call_id, out float val)
{
    val = call_data._result_derivative;
}
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_ND_buffer_output_transform(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    buffer_0 = StructuredBuffer(
        device=device,
        element_type=sgl.float3,
        shape=(10,),
        requires_grad=True
    )
    buffer_1 = StructuredBuffer(
        device=device,
        element_type=sgl.float3,
        shape=(5,),
        requires_grad=True
    )
    buffer_2 = StructuredBuffer(
        device=device,
        element_type=sgl.float1,
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
void load_a_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.a_primal[{call_id[0]}];
}
void load_b_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.b_primal[{call_id[1]}];
}
void store__result_primal(int[] call_id, in float val)
{
    call_data._result_primal[{call_id[0],call_id[1]}] = val;
}
""".strip()

    # bwds call should now have:
    # - primal reads and derivative stores for a and b from their correct call ids
    # - derivative write to both call ids for result
    print(bwds)
    assert bwds == """
void load_a_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.a_primal[{call_id[0]}];
}
void store_a_derivative(int[] call_id, in vector<float,3> val)
{
    call_data.a_derivative[{call_id[0]}] = val;
}
void load_b_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.b_primal[{call_id[1]}];
}
void store_b_derivative(int[] call_id, in vector<float,3> val)
{
    call_data.b_derivative[{call_id[1]}] = val;
}
void load__result_derivative(int[] call_id, out float val)
{
    val = call_data._result_derivative[{call_id[0],call_id[1]}];
}
""".strip()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
