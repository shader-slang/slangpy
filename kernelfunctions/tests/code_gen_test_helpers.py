# Dummy class that fakes a buffer of a given shape for testing


from types import NoneType
from typing import Any, Optional

import sgl

from kernelfunctions.callsignature import apply_signature, build_signature, calculate_and_apply_call_shape, generate_code, match_signature
from kernelfunctions.codegen import CodeGen
from kernelfunctions.signaturenode import CallMode
from kernelfunctions.tests import helpers
from kernelfunctions.typeregistry import BasePythonTypeMarshal, register_python_type


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


def dot_product(device_type: sgl.DeviceType, a: Any, b: Any, result: Any, opts: Optional[dict[str, Any]] = None):
    device = helpers.get_device(device_type)

    function = helpers.create_function_from_module(
        device,
        "dot_product",
        "[Differentiable]\nfloat dot_product(float3 a, float3 b) { return dot(a,b);}",
    )

    if opts is None:
        opts = {}
    input_transform = opts.get("input_transform", None)
    output_transform = opts.get("output_transform", None)

    sig = build_signature(a=a, b=b, _result=result)
    match = match_signature(
        sig, function.ast_functions[0].as_function(), CallMode.prim)
    assert match is not None
    apply_signature(match, function.ast_functions[0].as_function(
    ), input_transforms=input_transform, output_transforms=output_transform)
    call_shape = calculate_and_apply_call_shape(match)

    prim = CodeGen()
    bwds = CodeGen()
    fwds = CodeGen()
    generate_code(call_shape, match, CallMode.prim, prim)
    generate_code(call_shape, match, CallMode.bwds, bwds)
    generate_code(call_shape, match, CallMode.fwds, fwds)

    return prim, bwds, fwds
