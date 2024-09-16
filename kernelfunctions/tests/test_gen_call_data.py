import pytest
from kernelfunctions.types import NDDifferentiableBuffer
from kernelfunctions.core.codegen import CodeGen
from kernelfunctions.tests import helpers
from kernelfunctions.tests.code_gen_test_helpers import dot_product
from kernelfunctions.types import diffPair, floatDiffPair
from kernelfunctions.backend import DeviceType, float3, float1
from kernelfunctions.types.valueref import floatRef

pytest.skip(reason="Code gen changed - need to fix tests", allow_module_level=True)


def code(cg: tuple[CodeGen, CodeGen, CodeGen]):
    return cg[0].finish(call_data=True).strip(), cg[1].finish(call_data=True).strip(), cg[2].finish(call_data=True).strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_scalar(device_type: DeviceType):

    # really simple test case emulating slang function that takes
    # 2 x float3 and returns a float.
    (prim, bwds, fwds) = code(dot_product(device_type, float3(), float3(), None))

    # primitive call should pass in 2 vectors and output a float
    assert prim == """
struct CallData
{
    uint3 _thread_count;
    vector<float,3> a_primal;
    vector<float,3> b_primal;
    RWStructuredBuffer<float> _result_primal;
}
ParameterBlock<CallData> call_data;
""".strip()

    # bwds call is a bit pointless - no arguments are differentiable
    assert bwds == """
struct CallData
{
    uint3 _thread_count;
    vector<float,3> a_primal;
    vector<float,3> b_primal;
    float _result_derivative;
}
ParameterBlock<CallData> call_data;
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_scalar_ref(device_type: DeviceType):

    # use a scalar ref should get a structured buffer for output
    (prim, bwds, fwds) = code(dot_product(device_type, float3(), float3(), floatRef()))

    # primitive call should pass in 2 vectors and output a float
    assert prim == """
struct CallData
{
    uint3 _thread_count;
    vector<float,3> a_primal;
    vector<float,3> b_primal;
    RWStructuredBuffer<float> _result_primal;
}
ParameterBlock<CallData> call_data;
""".strip()

    # bwds call is a bit pointless - no arguments are differentiable
    assert bwds == """
struct CallData
{
    uint3 _thread_count;
    vector<float,3> a_primal;
    vector<float,3> b_primal;
}
ParameterBlock<CallData> call_data;
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_diff_pairs(device_type: DeviceType):

    # use a scalar ref should get a structured buffer for output
    (prim, bwds, fwds) = code(dot_product(device_type,
                                          diffPair(p=float3(), d=float3()),
                                          diffPair(p=float3(), d=float3()),
                                          floatDiffPair()))

    # primitive call should pass in 2 vectors and output a float
    assert prim == """
struct CallData
{
    uint3 _thread_count;
    vector<float,3> a_primal;
    vector<float,3> b_primal;
    RWStructuredBuffer<float> _result_primal;
}
ParameterBlock<CallData> call_data;
""".strip()

    # bwds call should now have:
    # - primal inputs and derivative outputs for a and b
    # - derivative input for result
    assert bwds == """
struct CallData
{
    uint3 _thread_count;
    vector<float,3> a_primal;
    RWStructuredBuffer<vector<float,3>> a_derivative;
    vector<float,3> b_primal;
    RWStructuredBuffer<vector<float,3>> b_derivative;
    float _result_derivative;
}
ParameterBlock<CallData> call_data;
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

    # use a scalar ref should get a structured buffer for output
    (prim, bwds, fwds) = code(dot_product(device_type,
                                          buffer_0,
                                          buffer_1,
                                          buffer_2))

    # prim call should have
    # - 2 1D read-only tensor buffers of type float3
    # - 1 1D read-write tensor buffer of type float
    assert prim == """
struct CallData
{
    int[1] _call_stride;
    int[1] _call_dim;
    uint3 _thread_count;
    TensorBuffer<vector<float,3>,1> a_primal;
    TensorBuffer<vector<float,3>,1> b_primal;
    RWTensorBuffer<float,1> _result_primal;
}
ParameterBlock<CallData> call_data;
""".strip()

    # bwds call should now have:
    # - read-only primal buffer for a
    # - read-write derivative buffer for a
    # - read-only primal buffer for b (it was not differentiable)
    # - read-only derivative buffer for result (it was only an output)
    assert bwds == """
struct CallData
{
    int[1] _call_stride;
    int[1] _call_dim;
    uint3 _thread_count;
    TensorBuffer<vector<float,3>,1> a_primal;
    RWTensorBuffer<vector<float,3>,1> a_derivative;
    TensorBuffer<vector<float,3>,1> b_primal;
    TensorBuffer<float,1> _result_derivative;
}
ParameterBlock<CallData> call_data;
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
struct CallData
{
    int[1] _call_stride;
    int[1] _call_dim;
    uint3 _thread_count;
    TensorBuffer<float,1> a__x_primal;
    float a__y_primal;
    float a__z_primal;
    TensorBuffer<vector<float,3>,1> b_primal;
    RWTensorBuffer<float,1> _result_primal;
}
ParameterBlock<CallData> call_data;
""".strip()

    # bwds call should now have:
    # - read-only primal buffer for a.x
    # - read-write derivative buffer for a.y
    # - floats a.y and a.z primals
    # - rw structured buffer to receive a.y derivative
    # - read-only primal buffer for b (it was not differentiable)
    # - read-only derivative buffer for result (it was only an output)
    assert bwds == """
struct CallData
{
    int[1] _call_stride;
    int[1] _call_dim;
    uint3 _thread_count;
    TensorBuffer<float,1> a__x_primal;
    RWTensorBuffer<float,1> a__x_derivative;
    float a__y_primal;
    RWStructuredBuffer<float> a__y_derivative;
    float a__z_primal;
    TensorBuffer<vector<float,3>,1> b_primal;
    TensorBuffer<float,1> _result_derivative;
}
ParameterBlock<CallData> call_data;
""".strip()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
