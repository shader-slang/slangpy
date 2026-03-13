# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import TYPE_CHECKING, Any

from slangpy.bindings.codegen import CodeGen, CodeGenBlock
from slangpy.core.native import AccessType

if TYPE_CHECKING:
    from slangpy.bindings.boundvariable import BoundVariable
    from slangpy.bindings.marshall import BindContext
    from slangpy.core.function import FunctionBuildInfo

#: Type names longer than this threshold get a ``typealias _t_{name}`` alias
#: to keep the generated ``CallData`` struct readable. Shorter names are
#: inlined directly.
MAX_INLINE_TYPE_LEN = 60


class KernelGenException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _is_slangpy_vector(type: Any) -> bool:
    return (
        hasattr(type, "element_type")
        and hasattr(type, "shape")
        and len(type.shape) == 1
        and type.shape[0] <= 4
    )


def generate_constants(build_info: "FunctionBuildInfo", cg: CodeGen) -> None:
    if build_info.constants is not None:
        for k, v in build_info.constants.items():
            if isinstance(v, bool):
                cg.constants.append_statement(
                    f"export static const bool {k} = {'true' if v else 'false'}"
                )
            elif isinstance(v, (int, float)):
                cg.constants.append_statement(f"export static const {type(v).__name__} {k} = {v}")
            elif _is_slangpy_vector(v):
                # Cheeky logic to take, eg, {0,0,0} -> float3(0,0,0)
                tn = type(v).__name__
                txt = f"{tn}({str(v)[1:-1]})"
                cg.constants.append_statement(f"export static const {tn} {k} = {txt}")
            else:
                raise KernelGenException(
                    f"Constant value '{k}' must be an int, float or bool, not {type(v).__name__}"
                )


def gen_calldata_type_name(binding: "BoundVariable", cgb: CodeGenBlock, type_name: str) -> None:
    """Record the Slang type name for this variable's CallData field.

    If the type name exceeds ``MAX_INLINE_TYPE_LEN``, a
    ``typealias _t_{name}`` is emitted and the alias is stored.
    Otherwise the raw type name is stored directly.

    :param binding: The bound variable to update.
    :param cgb: The code-gen block to write the type alias to (if needed).
    :param type_name: The resolved Slang type name.
    """
    if len(type_name) > MAX_INLINE_TYPE_LEN:
        alias = f"_t_{binding.variable_name}"
        cgb.type_alias(alias, type_name)
        binding.calldata_type_name = alias
    else:
        binding.calldata_type_name = type_name


def gen_call_data_code(
    binding: "BoundVariable", cg: CodeGen, context: "BindContext", depth: int = 0
) -> None:
    if binding.children is not None:
        cgb = cg.call_data_structs

        if binding.direct_bind:
            # Direct-bind: use raw type name directly
            assert binding.vector_type is not None
            gen_calldata_type_name(binding, cgb, binding.vector_type.full_name)
        else:
            struct_name = f"_t_{binding.variable_name}"
            cgb.begin_struct(struct_name)

            for field, variable in binding.children.items():
                gen_call_data_code(variable, cg, context, depth + 1)

            for var in binding.children.values():
                assert (
                    var.calldata_type_name is not None
                ), f"calldata_type_name not set for '{var.variable_name}'"
                cgb.declare(var.calldata_type_name, var.variable_name)

            assert binding.vector_type is not None
            context_decl = f"ContextND<{binding.call_dimensionality}> context"
            value_decl = f"{binding.vector_type.full_name} value"
            prefix = "[Differentiable]" if binding.access[1] != AccessType.none else ""

            if binding.access[0] in (AccessType.read, AccessType.readwrite):
                cgb.empty_line()
                cgb.append_line(f"{prefix} void __slangpy_load({context_decl}, out {value_decl})")
                cgb.begin_block()
                for field, var in binding.children.items():
                    gen_load = getattr(var.python, "gen_trampoline_load", None)
                    if gen_load is not None and gen_load(
                        cgb, var, var.variable_name, f"value.{field}"
                    ):
                        continue
                    cgb.append_statement(
                        f"{var.variable_name}.__slangpy_load(context.map(_m_{var.variable_name}),value.{field})"
                    )
                cgb.end_block()

            if binding.access[0] in (AccessType.write, AccessType.readwrite):
                cgb.empty_line()
                cgb.append_line(f"{prefix} void __slangpy_store({context_decl}, in {value_decl})")
                cgb.begin_block()
                for field, var in binding.children.items():
                    gen_store = getattr(var.python, "gen_trampoline_store", None)
                    if gen_store is not None and gen_store(
                        cgb, var, var.variable_name, f"value.{field}"
                    ):
                        continue
                    cgb.append_statement(
                        f"{var.variable_name}.__slangpy_store(context.map(_m_{var.variable_name}),value.{field})"
                    )
                cgb.end_block()

            cgb.end_struct()
            binding.calldata_type_name = struct_name

    else:
        # Generate call data
        binding.python.gen_calldata(cg.call_data_structs, context, binding)

    # Skip mapping constants for direct-bind variables (they bypass __slangpy_load/store)
    if not binding.direct_bind:
        if len(binding.vector_mapping) > 0:
            cg.call_data_structs.append_statement(
                f"static const int[] _m_{binding.variable_name} = {{ {','.join([str(x) for x in binding.vector_mapping.as_tuple()])} }}"
            )
        else:
            cg.call_data_structs.append_statement(
                f"static const int _m_{binding.variable_name} = 0"
            )

    if depth == 0:
        assert (
            binding.calldata_type_name is not None
        ), f"calldata_type_name not set for '{binding.variable_name}'"
        if binding.create_param_block:
            cg.add_parameter_block(binding.calldata_type_name, "_param_" + binding.variable_name)
        elif cg.skip_call_data:
            cg.entry_point_params.append(
                f"uniform {binding.calldata_type_name} {binding.variable_name}"
            )
        else:
            cg.call_data.declare(binding.calldata_type_name, binding.variable_name)
