

from typing import Any, Optional
import numpy as np

from kernelfunctions.bindings.valuereftype import numpy_to_slang_value, slang_value_to_numpy
from kernelfunctions.core import CodeGenBlock, BindContext, BaseType, BaseTypeImpl, BoundVariable, AccessType, PrimType, BoundVariableRuntime, CallContext

from kernelfunctions.core.reflection import SlangProgramLayout, SlangType
from kernelfunctions.types import DiffPair

from kernelfunctions.backend import Buffer, ResourceUsage
from kernelfunctions.typeregistry import PYTHON_TYPES, get_or_create_type

import kernelfunctions.core.reflection as kfr


def generate_differential_pair(name: str, primal_storage: str, deriv_storage: str, primal_target: str, deriv_target: Optional[str]):
    assert primal_storage
    assert deriv_storage
    assert primal_target
    if deriv_target is None:
        deriv_target = primal_target

    DIFF_PAIR_CODE = f"""
struct _t_{name}
{{
    {primal_storage} primal;
    {deriv_storage} derivative;
    void load_primal(IContext context, out {primal_target} value) {{ primal.load_primal(context, value); }}
    void store_primal(IContext context, in {primal_target} value) {{ primal.store_primal(context, value); }}
    void load_derivative(IContext context, out {deriv_target} value) {{ derivative.load_primal(context, value); }}
    void store_derivative(IContext context, in {deriv_target} value) {{ derivative.store_primal(context, value); }}
}}
"""
    return DIFF_PAIR_CODE


class DiffPairType(BaseTypeImpl):

    def __init__(self, layout: SlangProgramLayout, primal_type: BaseType, derivative_type: Optional[BaseType], needs_grad: bool):
        super().__init__(layout)
        slt = layout.find_type_by_name(
            f"DifferentialPair<{primal_type.slang_type.full_name}>")
        assert isinstance(slt, kfr.DifferentialPairType)
        self.slang_type: kfr.DifferentialPairType = slt
        self.needs_grad = needs_grad
        self.primal = primal_type

        # A pure diff pair should be being passed to either a diff pair, or its primal. In both
        # cases treating its shape as that of its primal is valid for the dispatch.
        assert self.primal.concrete_shape.valid
        self.concrete_shape = self.primal.concrete_shape

    # Values don't store a derivative - they're just a value

    @property
    def has_derivative(self) -> bool:
        return self.needs_grad and self.slang_type.differentiable

    # Refs can be written to!
    @property
    def is_writable(self) -> bool:
        return True

    # A diff pair going to a diff pair is just a default cast. Otherwise
    # attempt to cast to the primal.
    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        if bound_type.name == "DifferentialPair":
            return bound_type
        else:
            return self.primal.resolve_type(context, bound_type)

    # A diff pair going to a diff pair has dimensionality of 0, otherwise use the
    # resolve function for the primal
    def resolve_dimensionality(self, context: BindContext, vector_target_type: 'kfr.SlangType'):
        if vector_target_type.name == "DifferentialPair":
            return 0
        else:
            return self.primal.resolve_dimensionality(context, vector_target_type)

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name

        prim_el = self.slang_type.primal.full_name
        deriv_el = self.slang_type.primal.derivative.full_name

        if access[0] == AccessType.none:
            primal_storage = f'NoneType'
        elif access[0] == AccessType.read:
            primal_storage = f"ValueType<{prim_el}>"
        else:
            primal_storage = f"RWValueRef<{prim_el}>"

        if access[1] == AccessType.none:
            deriv_storage = f'NoneType'
        elif access[1] == AccessType.read:
            deriv_storage = f"ValueType<{deriv_el}>"
        else:
            deriv_storage = f"RWValueRef<{deriv_el}>"

        assert binding.vector_type is not None
        primal_target = binding.vector_type.full_name
        deriv_target = binding.vector_type.full_name + ".Differential"

        cgb.append_code_indented(generate_differential_pair(name, primal_storage,
                                                            deriv_storage, primal_target, deriv_target))

    def get_type(self, prim: PrimType):
        return self.slang_type.primal if prim == PrimType.primal else self.slang_type.primal.derivative

    # Call data just returns the primal
    def create_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: DiffPair) -> Any:
        access = binding.access
        res = {}

        for prim in PrimType:
            prim_name = prim.name
            prim_access = access[prim.value]
            prim_data = data.get(prim)
            prim_type = self.get_type(prim)
            if prim_access in [AccessType.write, AccessType.readwrite]:
                assert prim_type is not None
                npdata = slang_value_to_numpy(prim_type, prim_data).view(dtype=np.uint8)
                res[prim_name] = {
                    'value': context.device.create_buffer(
                        element_count=1,
                        struct_size=npdata.size,
                        data=npdata,
                        usage=ResourceUsage.shader_resource | ResourceUsage.unordered_access)}
            elif prim_access == AccessType.read:
                res[prim_name] = {'value': prim_data}

        return res

    # Read back from call data does nothing
    def read_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: DiffPair, result: Any) -> None:
        access = binding.access
        for prim in PrimType:
            prim_name = prim.name
            prim_access = access[prim.value]
            prim_type = self.get_type(prim)
            if prim_access in [AccessType.write, AccessType.readwrite]:
                assert isinstance(result[prim_name]['value'], Buffer)
                assert prim_type is not None
                val = numpy_to_slang_value(
                    prim_type, result[prim_name]['value'].to_numpy())
                data.set(prim, val)

    def create_output(self, context: CallContext, binding: BoundVariableRuntime) -> Any:
        return DiffPair(None, None)

    def read_output(self, context: CallContext, binding: BoundVariableRuntime, data: DiffPair) -> Any:
        return data


def create_vr_type_for_value(layout: SlangProgramLayout, value: Any):
    assert isinstance(value, DiffPair)
    return DiffPairType(layout, get_or_create_type(layout, type(value.primal)), get_or_create_type(layout, type(value.grad)), value.needs_grad)


PYTHON_TYPES[DiffPair] = create_vr_type_for_value
