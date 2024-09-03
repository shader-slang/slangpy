import re
from types import NoneType
from typing import Any, Optional
import pytest
import sgl
from kernelfunctions.buffer import StructuredBuffer
from kernelfunctions.callsignature import CallMode, SignatureNode, apply_signature, build_signature, calculate_and_apply_call_shape, generate_call_data_declarations, generate_call_data_struct, match_signature
from kernelfunctions.shapes import TLooseShape, build_indexer
import deepdiff

from kernelfunctions.tests import helpers
from kernelfunctions.typeregistry import BasePythonTypeMarshal, register_python_type
from kernelfunctions.utils import diffPair, floatDiffPair, floatRef

# Dummy class that fakes a buffer of a given shape for testing


class FakeBuffer:
    def __init__(self, shape: tuple[Optional[int], ...]):
        super().__init__()
        self.shape = shape


class FakeBufferMarshall(BasePythonTypeMarshal):
    def __init__(self):
        super().__init__(FakeBuffer)

    def get_shape(self, value: FakeBuffer) -> tuple[int | None, ...]:
        return value.shape

    def get_element_type(self, value: Any):
        return NoneType


register_python_type(FakeBuffer, FakeBufferMarshall(),
                     lambda stream, x: stream.write(x.element_type.__name + "\n"))


# First set of tests emulate the shape of the following slang function
# float test(float3 a, float3 b) { return dot(a,b); }
# Note that the return value is simply treated as a final 'out' parameter
def dot_product(device_type: sgl.DeviceType, a: Any, b: Any, result: Any) -> tuple[str, str, str]:
    device = helpers.get_device(device_type)

    function = helpers.create_function_from_module(
        device,
        "dot_product",
        "[Differentiable]\nfloat dot_product(float3 a, float3 b) { return dot(a,b);}",
    )

    sig = build_signature(a=a, b=b, _result=result)
    match = match_signature(
        sig, function.ast_functions[0].as_function(), CallMode.prim)
    assert match is not None
    apply_signature(match, function.ast_functions[0].as_function())
    call_shape = calculate_and_apply_call_shape(match)

    prim = generate_call_data_struct(tuple(call_shape), match, CallMode.prim).strip()
    bwds = generate_call_data_struct(tuple(call_shape), match, CallMode.bwds).strip()
    fwds = generate_call_data_struct(tuple(call_shape), match, CallMode.fwds).strip()

    return prim, bwds, fwds


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_scalar(device_type: sgl.DeviceType):

    # really simple test case emulating slang function that takes
    # 2 x float3 and returns a float.
    (prim, bwds, fwds) = dot_product(device_type, sgl.float3(), sgl.float3(), None)

    # primitive call should pass in 2 vectors and output a float
    assert prim == """
struct CallData {
    vector<float,3> a_primal;
    vector<float,3> b_primal;
    RWTensorBuffer<float,1> _result_primal;
}
ParameterBlock<CallData> call_data;
""".strip()

    # bwds call is a bit pointless - no arguments are differentiable
    assert bwds == """
struct CallData {
    vector<float,3> a_primal;
    vector<float,3> b_primal;
}
ParameterBlock<CallData> call_data;
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_scalar_ref(device_type: sgl.DeviceType):

    # use a scalar ref should get a structured buffer for output
    (prim, bwds, fwds) = dot_product(device_type, sgl.float3(), sgl.float3(), floatRef())

    # primitive call should pass in 2 vectors and output a float
    assert prim == """
struct CallData {
    vector<float,3> a_primal;
    vector<float,3> b_primal;
    RWStructuredBuffer<float> _result_primal;
}
ParameterBlock<CallData> call_data;
""".strip()

    # bwds call is a bit pointless - no arguments are differentiable
    assert bwds == """
struct CallData {
    vector<float,3> a_primal;
    vector<float,3> b_primal;
}
ParameterBlock<CallData> call_data;
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_diff_pairs(device_type: sgl.DeviceType):

    # use a scalar ref should get a structured buffer for output
    (prim, bwds, fwds) = dot_product(device_type,
                                     diffPair(p=sgl.float3(), d=sgl.float3()),
                                     diffPair(p=sgl.float3(), d=sgl.float3()),
                                     floatDiffPair())

    # primitive call should pass in 2 vectors and output a float
    assert prim == """
struct CallData {
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
struct CallData {
    vector<float,3> a_primal;
    RWStructuredBuffer<vector<float,3>> a_derivative;
    vector<float,3> b_primal;
    RWStructuredBuffer<vector<float,3>> b_derivative;
    float _result_derivative;
}
ParameterBlock<CallData> call_data;
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
    (prim, bwds, fwds) = dot_product(device_type,
                                     buffer_0,
                                     buffer_1,
                                     buffer_2)

    # prim call should have
    # - 2 1D read-only tensor buffers of type float3
    # - 1 1D read-write tensor buffer of type float
    assert prim == """
struct CallData {
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
struct CallData {
    TensorBuffer<vector<float,3>,1> a_primal;
    RWTensorBuffer<vector<float,3>,1> a_derivative;
    TensorBuffer<vector<float,3>,1> b_primal;
    TensorBuffer<float,1> _result_derivative;
}
ParameterBlock<CallData> call_data;
""".strip()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_buffer_soa(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    buffer_0 = StructuredBuffer(
        device=device,
        element_type=float,
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
    (prim, bwds, fwds) = dot_product(device_type,
                                     {
                                         'x': buffer_0,
                                         'y': floatDiffPair(),
                                         'z': 2.0
                                     },
                                     buffer_1,
                                     buffer_2)

    # prim call should have
    # - 1D read only tensor buffer of floats for a.x
    # - floats for a.y and a.z
    # - 1D read only tensor buffer of float3s for b
    # - 1 1D read-write tensor buffer of type float
    assert prim == """
struct CallData {
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
struct CallData {
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
