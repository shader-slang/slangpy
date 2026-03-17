# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import TYPE_CHECKING, Any
import warnings

from slangpy.bindings.codegen import CodeGen, CodeGenBlock
from slangpy.core.native import AccessType, CallMode

if TYPE_CHECKING:
    from slangpy.bindings.boundvariable import BoundVariable
    from slangpy.bindings.marshall import BindContext
    from slangpy.core.function import FunctionBuildInfo
    from slangpy.bindings.boundvariable import BoundVariable, BoundCall

#: Type names longer than this threshold get a ``typealias _t_{name}`` alias
#: to keep the generated ``CallData`` struct readable. Shorter names are
#: inlined directly.
MAX_INLINE_TYPE_LEN = 60


class KernelGenException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# Access-tuple helpers
# ---------------------------------------------------------------------------
# BoundVariable.access is a (primal, derivative) tuple of AccessType.
# These predicates give readable names to the index lookups.


def _is_readable(b: "BoundVariable") -> bool:
    """True when the primal value is read (read or readwrite)."""
    return b.access[0] in (AccessType.read, AccessType.readwrite)


def _is_writable(b: "BoundVariable") -> bool:
    """True when the primal value is written (write or readwrite)."""
    return b.access[0] in (AccessType.write, AccessType.readwrite)


def _is_differentiable(b: "BoundVariable") -> bool:
    """True when the derivative access is anything other than none."""
    return b.access[1] != AccessType.none


def _grad_is_readable(b: "BoundVariable") -> bool:
    """True when the derivative/gradient is read back."""
    return b.access[1] == AccessType.read


# ---------------------------------------------------------------------------
# Shared trampoline dispatch helper
# ---------------------------------------------------------------------------


def _try_custom_gen(
    var: "BoundVariable",
    method: str,
    cgb: CodeGenBlock,
    data_name: str,
    value_expr: str,
) -> bool:
    """Try calling a marshall-specific ``gen_trampoline_load`` or ``gen_trampoline_store``.

    Returns True if the marshall handled code generation, False otherwise.
    """
    fn = getattr(var.python, method, None)
    return fn is not None and fn(cgb, var, data_name, value_expr)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _is_slangpy_vector(type: Any) -> bool:
    return (
        hasattr(type, "element_type")
        and hasattr(type, "shape")
        and len(type.shape) == 1
        and type.shape[0] <= 4
    )


# ---------------------------------------------------------------------------
# Call-data code generation
# ---------------------------------------------------------------------------


def _emit_user_constants(build_info: "FunctionBuildInfo", cg: CodeGen) -> None:
    """Emit user-provided ``build_info.constants`` as exported Slang constants,
    by appending them to the ``CodeGen.constants`` block.

    Example emitted declarations::

        export static const bool enable_flag = true;
        export static const int iterations = 32;
        export static const float threshold = 0.5;
        export static const float3 tint = float3(1,0,0);
    """
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


#: Compatibility alias for legacy imports.
generate_constants = _emit_user_constants


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


def _emit_field_load(
    cgb: CodeGenBlock,
    var: "BoundVariable",
    field: str,
) -> None:
    """Emit a single field's ``__slangpy_load`` call inside a composite struct.
    Will either use a marshall-specific load that implements direct binding to a uniform value,
    or emit a call to the field's ``__slangpy_load`` method.

    Example emitted code for field ``a``::
        a.__slangpy_load(context.map(_m_a), value.a); // use field load method
        value.a = a; // direct-bind load (no __slangpy_load method)
    """
    if _try_custom_gen(var, "gen_trampoline_load", cgb, var.variable_name, f"value.{field}"):
        return
    cgb.append_statement(
        f"{var.variable_name}.__slangpy_load(context.map(_m_{var.variable_name}),value.{field})"
    )


def _emit_field_store(
    cgb: CodeGenBlock,
    var: "BoundVariable",
    field: str,
) -> None:
    """Emit a single field's ``__slangpy_store`` call inside a composite struct.

    Example emitted code for field ``a``::
        a.__slangpy_store(context.map(_m_a), value.a);

    :param cgb: The code-gen block to write the store call to.
    :param var: The bound variable representing the field to store.
    :param field: The name of the field being stored (used for generating the value reference and error messages).
    """
    if _try_custom_gen(var, "gen_trampoline_store", cgb, var.variable_name, f"value.{field}"):
        return
    cgb.append_statement(
        f"{var.variable_name}.__slangpy_store(context.map(_m_{var.variable_name}),value.{field})"
    )


def _emit_composite_load_func(
    cgb: CodeGenBlock,
    binding: "BoundVariable",
) -> None:
    """Emit the ``__slangpy_load`` method for a composite call-data struct. This
    may include calls to __slangpy_load, or delegate to a marshall-specific
    load that implements direct binding to a uniform values;

    Example: for a struct with fields ``a`` and ``b``::
        void __slangpy_load(ContextND<2> context, out Foo value) {
            // load via marshall
            a.__slangpy_load(context.map(_m_a), value.a);

            // or direct-bind load
            value.b = this.b;
        }

    :param cgb: The code-gen block to write the load function to.
    :param binding: The bound variable representing the composite struct.
    """
    assert binding.children is not None
    assert binding.vector_type is not None
    context_decl = f"ContextND<{binding.call_dimensionality}> context"
    value_decl = f"{binding.vector_type.full_name} value"
    prefix = "[Differentiable]" if _is_differentiable(binding) else ""
    cgb.empty_line()
    cgb.append_line(f"{prefix} void __slangpy_load({context_decl}, out {value_decl})")
    cgb.begin_block()
    for field, var in binding.children.items():
        _emit_field_load(cgb, var, field)
    cgb.end_block()


def _emit_composite_store_func(
    cgb: CodeGenBlock,
    binding: "BoundVariable",
) -> None:
    """Emit the ``__slangpy_store`` method for a composite call-data struct.

    Example: for a struct with fields ``a`` and ``b``::
        void __slangpy_store(ContextND<2> context, in Foo value) {
            a.__slangpy_store(context.map(_m_a), value.a);
            b.__slangpy_store(context.map(_m_b), value.b);
        }

    :param cgb: The code-gen block to write the store function to.
    :param binding: The bound variable representing the composite struct.
    """
    assert binding.children is not None
    assert binding.vector_type is not None
    context_decl = f"ContextND<{binding.call_dimensionality}> context"
    value_decl = f"{binding.vector_type.full_name} value"
    prefix = "[Differentiable]" if _is_differentiable(binding) else ""
    cgb.empty_line()
    cgb.append_line(f"{prefix} void __slangpy_store({context_decl}, in {value_decl})")
    cgb.begin_block()
    for field, var in binding.children.items():
        _emit_field_store(cgb, var, field)
    cgb.end_block()


def _emit_type_and_struct(
    binding: "BoundVariable", cg: CodeGen, context: "BindContext", depth: int
) -> None:
    """Emit the type declaration for a single binding.

    For composite (struct/dict) variables, emits a ``_t_{name}`` struct with
    ``__slangpy_load`` / ``__slangpy_store`` methods and recurses into children.
    For leaf variables, delegates to the marshall's ``gen_calldata``.

    Example composite output::

        struct _t_foo {
            ChildType a;
            ChildType b;
            void __slangpy_load(ContextND<2> context, out Foo value) { ... }
            void __slangpy_store(ContextND<2> context, in Foo value) { ... }
        }

    Example leaf output::

        typealias _t_foo = int;
    """
    if binding.children is not None:
        cgb = cg.call_data_structs

        if binding.direct_bind:
            assert binding.vector_type is not None
            gen_calldata_type_name(binding, cgb, binding.vector_type.full_name)
        else:
            struct_name = f"_t_{binding.variable_name}"
            cgb.begin_struct(struct_name)

            # Recurse into children (emitted as separate structs, not nested).
            for field, variable in binding.children.items():
                _emit_call_data_code(variable, cg, context, depth + 1)

            # Member variables
            for var in binding.children.values():
                assert (
                    var.calldata_type_name is not None
                ), f"calldata_type_name not set for '{var.variable_name}'"
                cgb.declare(var.calldata_type_name, var.variable_name)

            # Load/store methods
            if _is_readable(binding):
                _emit_composite_load_func(cgb, binding)
            if _is_writable(binding):
                _emit_composite_store_func(cgb, binding)

            cgb.end_struct()
            binding.calldata_type_name = struct_name
    else:
        binding.python.gen_calldata(cg.call_data_structs, context, binding)

        if binding.calldata_type_name is None:
            warnings.warn(
                f"Marshall '{type(binding.python).__name__}' did not set "
                f"calldata_type_name for '{binding.variable_name}' in gen_calldata(). "
                f"Ensure gen_calldata calls binding.gen_calldata_type_name(). "
                f"Defaulting to python marshall slang type name '{binding.python.slang_type.full_name}'.",
            )
            binding.calldata_type_name = binding.python.slang_type.full_name


def _emit_mapping_constants(binding: "BoundVariable", cg: CodeGen) -> None:
    """Emit the vectorization mapping constant for a single binding.

    Example output::

        static const int[] _m_foo = {0,1,2};   // vectorized
        static const int _m_foo = 0;            // scalar
    """
    if binding.direct_bind:
        return
    if len(binding.vector_mapping) > 0:
        cg.call_data_structs.append_statement(
            f"static const int[] _m_{binding.variable_name}"
            f" = {{ {','.join([str(x) for x in binding.vector_mapping.as_tuple()])} }}"
        )
    else:
        cg.call_data_structs.append_statement(f"static const int _m_{binding.variable_name} = 0")


def _emit_root_declaration(binding: "BoundVariable", cg: CodeGen) -> None:
    """At depth 0, declare the variable in the appropriate destination.

    Chooses between a parameter block, an entry-point uniform, or a
    CallData struct field depending on the binding configuration.
    """
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


def _emit_call_data_code(
    binding: "BoundVariable", cg: CodeGen, context: "BindContext", depth: int = 0
) -> None:
    """Emit Slang call-data type declarations, mapping constants, and
    root-level variable declarations for a single binding.

    Orchestrates three sub-steps:
    1. Type/struct declaration (``_emit_type_and_struct``)
    2. Mapping constants (``_emit_mapping_constants``)
    3. Root declaration at depth 0 (``_emit_root_declaration``)

    :param binding: The bound variable to emit code for.
    :param cg: The active CodeGen object.
    :param context: The bind context for the current call.
    :param depth: Recursion depth (0 = root, >0 = struct field).
    """
    _emit_type_and_struct(binding, cg, context, depth)
    _emit_mapping_constants(binding, cg)
    if depth == 0:
        _emit_root_declaration(binding, cg)


#: Compatibility alias for legacy imports.
gen_call_data_code = _emit_call_data_code


# ---------------------------------------------------------------------------
# generate_code sub-functions
# ---------------------------------------------------------------------------


def _validate_and_compute_group_shape(
    build_info: "FunctionBuildInfo",
    call_data_len: int,
) -> tuple[int, list[int], list[int]]:
    """Validate ``call_group_shape`` and compute the flat group size and strides.

    Returns ``(call_group_size, call_group_strides, call_group_shape_vector)``.
    When no call_group_shape is set, returns ``(1, [], [])``.
    """
    call_group_size = 1
    call_group_strides: list[int] = []
    call_group_shape_vector: list[int] = []

    call_group_shape = build_info.call_group_shape
    if call_group_shape is not None:
        call_group_shape_vector = call_group_shape.as_list()

        if len(call_group_shape_vector) > call_data_len:
            raise KernelGenException(
                f"call_group_shape dimensionality ({len(call_group_shape_vector)}) must be <= "
                f"call_shape dimensionality ({call_data_len}). "
                f"call_group_shape cannot have more dimensions than call_shape."
            )
        elif len(call_group_shape_vector) < call_data_len:
            missing_dims = call_data_len - len(call_group_shape_vector)
            call_group_shape_vector = [1] * missing_dims + call_group_shape_vector

        for i, dim in enumerate(call_group_shape_vector):
            if dim < 1:
                raise KernelGenException(
                    f"call_group_shape[{i}] = {dim} is invalid. "
                    f"All call_group_shape elements must be >= 1."
                )

        for dim in call_group_shape_vector[::-1]:
            call_group_strides.append(call_group_size)
            call_group_size *= dim
        call_group_strides.reverse()

        if call_group_size > 1024:
            raise KernelGenException(
                f"call_group_size ({call_group_size}) exceeds the typical 1024 maximum "
                f"enforced by most APIs. Consider reducing your call_group_shape dimensions."
            )

    return call_group_size, call_group_strides, call_group_shape_vector


def _emit_link_time_constants(
    cg: CodeGen,
    build_info: "FunctionBuildInfo",
    call_data_len: int,
    call_group_size: int,
    call_group_strides: list[int],
    call_group_shape_vector: list[int],
) -> None:
    """Emit link-time constant declarations, including user defined ones
    and any of the required call group shape constants.

    Emits Slang code like::

        // User constants from build_info.constants (if present)
        export static const int user_const = 7;

        export static const int call_data_len = 2;
        export static const int call_group_size = 1;
        export static const int[call_data_len] call_group_strides = {};
        export static const int[call_data_len] call_group_shape_vector = {};
    """
    _emit_user_constants(build_info, cg)
    cg.constants.append_statement(f"export static const int call_data_len = {call_data_len}")
    cg.constants.append_statement(f"export static const int call_group_size = {call_group_size}")

    cg.constants.append_line(f"export static const int[call_data_len] call_group_strides = {{")
    cg.constants.inc_indent()
    if call_group_size != 1:
        for i in range(call_data_len):
            cg.constants.append_line(f"{call_group_strides[i]},")
    cg.constants.dec_indent()
    cg.constants.append_statement("}")

    cg.constants.append_line(f"export static const int[call_data_len] call_group_shape_vector = {{")
    cg.constants.inc_indent()
    if call_group_size != 1:
        for i in range(call_data_len):
            cg.constants.append_line(f"{call_group_shape_vector[i]},")
    cg.constants.dec_indent()
    cg.constants.append_statement("}")


def _emit_shape_and_metadata_params(
    cg: CodeGen,
    call_data_len: int,
    use_entrypoint_args: bool,
) -> None:
    """Emit shape arrays and ``_thread_count``.

    Fast path (entry-point params)::

        uniform int[N] _grid_stride
        uniform int[N] _grid_dim
        uniform int[N] _call_dim
        uniform uint3 _thread_count

    Fallback (CallData struct fields)::

        int[N] _grid_stride;
        int[N] _grid_dim;
        int[N] _call_dim;
        uint3 _thread_count;
    """
    if call_data_len > 0:
        if use_entrypoint_args:
            cg.entry_point_params.append(f"uniform int[{call_data_len}] _grid_stride")
            cg.entry_point_params.append(f"uniform int[{call_data_len}] _grid_dim")
            cg.entry_point_params.append(f"uniform int[{call_data_len}] _call_dim")
        else:
            cg.call_data.append_statement(f"int[{call_data_len}] _grid_stride")
            cg.call_data.append_statement(f"int[{call_data_len}] _grid_dim")
            cg.call_data.append_statement(f"int[{call_data_len}] _call_dim")

    if use_entrypoint_args:
        cg.entry_point_params.append("uniform uint3 _thread_count")
    else:
        cg.call_data.append_statement("uint3 _thread_count")


def _emit_call_data_definitions(
    cg: CodeGen,
    context: "BindContext",
    signature: "BoundCall",
) -> None:
    """Emit per-variable call-data structs and type aliases for all signature nodes."""
    for node in signature.values():
        _emit_call_data_code(node, cg, context)


def _data_name(x: "BoundVariable", use_entrypoint_args: bool) -> str:
    """Return the Slang name used to access a variable's call data in the trampoline.

    - ``_param_{name}`` for param-block variables (both paths).
    - ``__in_{name}`` in the fast (entry-point-args) path.
    - ``call_data.{name}`` in the fallback path.
    """
    if x.create_param_block:
        return f"_param_{x.variable_name}"
    elif use_entrypoint_args:
        return f"__in_{x.variable_name}"
    else:
        return f"call_data.{x.variable_name}"


def _emit_trampoline_loads(
    cgb: CodeGenBlock,
    root_params: list["BoundVariable"],
    use_entrypoint_args: bool,
) -> None:
    """Emit ``__slangpy_load`` calls for each readable trampoline parameter.

    For each parameter, either delegates to a marshall-specific
    ``gen_trampoline_load`` or emits a standard load call::

        __in_x.__slangpy_load(__slangpy_context__.map(_m_x), x); // slangpy load
        x = __in_x; // direct-bind load (no __slangpy_load method)
    """
    for x in root_params:
        data_name = _data_name(x, use_entrypoint_args)
        if _try_custom_gen(x, "gen_trampoline_load", cgb, data_name, x.variable_name):
            continue
        if _is_readable(x):
            cgb.append_statement(
                f"{data_name}.__slangpy_load(__slangpy_context__.map(_m_{x.variable_name}), {x.variable_name})"
            )


def _emit_trampoline_stores(
    cgb: CodeGenBlock,
    root_params: list["BoundVariable"],
    use_entrypoint_args: bool,
) -> None:
    """Emit ``__slangpy_store`` calls for each writable trampoline parameter.

    For each parameter that is written or whose gradient is read, either
    delegates to a marshall-specific ``gen_trampoline_store`` or emits a
    standard store call::

        __in_x.__slangpy_store(__slangpy_context__.map(_m_x), x);
    """
    from slangpy.bindings.boundvariable import BoundVariableException

    for x in root_params:
        if _is_writable(x) or _grad_is_readable(x):
            data_name = _data_name(x, use_entrypoint_args)
            if _try_custom_gen(x, "gen_trampoline_store", cgb, data_name, x.variable_name):
                continue
            if not x.python.is_writable:
                raise BoundVariableException(f"Cannot read back value for non-writable type", x)
            cgb.append_statement(
                f"{data_name}.__slangpy_store(__slangpy_context__.map(_m_{x.variable_name}), {x.variable_name})"
            )


def _emit_trampoline(
    cg: CodeGen,
    context: "BindContext",
    build_info: "FunctionBuildInfo",
    root_params: list["BoundVariable"],
    use_entrypoint_args: bool,
) -> None:
    """Emit the ``_trampoline`` helper function.

    Fast path signature::

        [Differentiable]
        void _trampoline(Context __slangpy_context__,
                         no_diff MyType __in_param0, ...)

    Fallback signature::

        [Differentiable]
        void _trampoline(Context __slangpy_context__)
    """
    if context.call_mode != CallMode.prim:
        cg.trampoline.append_line("[Differentiable]")

    if use_entrypoint_args:
        trampoline_params = ["Context __slangpy_context__"]
        for x in root_params:
            if x.create_param_block:
                continue
            assert x.calldata_type_name is not None
            trampoline_params.append(f"no_diff {x.calldata_type_name} __in_{x.variable_name}")
        cg.trampoline.append_line(f"void _trampoline({', '.join(trampoline_params)})")
    else:
        cg.trampoline.append_line("void _trampoline(Context __slangpy_context__)")
    cg.trampoline.begin_block()

    # Declare local variables for each parameter
    for x in root_params:
        assert x.vector_type is not None
        cg.trampoline.declare(x.vector_type.full_name, x.variable_name)

    # Load inputs from call data
    _emit_trampoline_loads(cg.trampoline, root_params, use_entrypoint_args)

    # Emit the 'result=' bit if function has a return value.
    cg.trampoline.append_indent()
    if any(x.variable_name == "_result" for x in root_params):
        cg.trampoline.append_code("_result = ")

    # Generate the function call prefix, with some special casing for constructors
    # and type method calls.
    func_name = build_info.name
    if func_name == "$init":
        results = [x for x in root_params if x.variable_name == "_result"]
        assert len(results) == 1
        assert results[0].vector_type is not None
        func_name = results[0].vector_type.full_name
    elif len(root_params) > 0 and root_params[0].variable_name == "_this":
        func_name = f"_this.{func_name}"

    # Emit the function call itself, passing in parameters other than _result and _this.
    normal_params = [
        x for x in root_params if x.variable_name != "_result" and x.variable_name != "_this"
    ]
    cg.trampoline.append_code(
        f"{func_name}(" + ", ".join(x.variable_name for x in normal_params) + ");\n"
    )

    # Store outputs back to call data
    _emit_trampoline_stores(cg.trampoline, root_params, use_entrypoint_args)

    cg.trampoline.end_block()
    cg.trampoline.append_line("")


def _emit_entry_point_signature(
    cg: CodeGen,
    build_info: "FunctionBuildInfo",
    call_data_len: int,
    call_group_size: int,
    use_entrypoint_args: bool,
) -> None:
    """Emit the ``[shader(...)]`` attribute line and entry-point function signature.

    Compute fast path::

        [shader("compute")]
        [numthreads(32, 1, 1)]
        void compute_main(int3 flat_call_thread_id: SV_DispatchThreadID,
                          int3 flat_call_group_id: SV_GroupID,
                          int flat_call_group_thread_id: SV_GroupIndex,
                          uniform int[N] _grid_stride, ...)

    Ray-tracing entry point::

        [shader("raygen")]
        void raygen_main()
    """
    from slangpy.core.function import PipelineType

    if build_info.pipeline_type == PipelineType.compute:
        cg.kernel.append_line('[shader("compute")]')
        if call_group_size != 1:
            cg.kernel.append_line(f"[numthreads({call_group_size}, 1, 1)]")
        else:
            cg.kernel.append_line("[numthreads(32, 1, 1)]")
        if use_entrypoint_args:
            sig_parts = ["int3 flat_call_thread_id: SV_DispatchThreadID"]
            if call_data_len > 0:
                sig_parts.append("int3 flat_call_group_id: SV_GroupID")
                sig_parts.append("int flat_call_group_thread_id: SV_GroupIndex")
            sig_parts.extend(cg.entry_point_params)
            cg.kernel.append_line(f"void compute_main({', '.join(sig_parts)})")
        else:
            sig_parts = [
                "int3 flat_call_thread_id: SV_DispatchThreadID",
                "int3 flat_call_group_id: SV_GroupID",
                "int flat_call_group_thread_id: SV_GroupIndex",
            ]
            cg.kernel.append_line(f"void compute_main({', '.join(sig_parts)})")
    elif build_info.pipeline_type == PipelineType.ray_tracing:
        cg.kernel.append_line('[shader("raygen")]')
        if use_entrypoint_args:
            sig_parts = list(cg.entry_point_params)
            cg.kernel.append_line(f"void raygen_main({', '.join(sig_parts)})")
        else:
            cg.kernel.append_line("void raygen_main()")
    else:
        raise RuntimeError(f"Unknown pipeline type: {build_info.pipeline_type}")


def _emit_kernel_body(
    cg: CodeGen,
    context: "BindContext",
    build_info: "FunctionBuildInfo",
    root_params: list["BoundVariable"],
    call_data_len: int,
    use_entrypoint_args: bool,
) -> None:
    """Emit the body of the compute/raygen entry-point function.

    Emits the bounds check, ``init_thread_local_call_shape_info``, Context
    construction, and the trampoline call::

        if (any(flat_call_thread_id >= _thread_count)) return;
        if (!init_thread_local_call_shape_info(...)) return;
        Context __slangpy_context__ = {flat_call_thread_id, ...};
        _trampoline(__slangpy_context__, ...);
    """
    from slangpy.core.function import PipelineType

    # For RTP, read thread ID using DispatchRaysIndex() instead of SV_DispatchThreadID
    if build_info.pipeline_type == PipelineType.ray_tracing:
        cg.kernel.append_statement("int3 flat_call_thread_id = DispatchRaysIndex();")

    # Bail out if out of bounds.
    if use_entrypoint_args:
        cg.kernel.append_statement("if (any(flat_call_thread_id >= _thread_count)) return")
    else:
        cg.kernel.append_statement(
            "if (any(flat_call_thread_id >= call_data._thread_count)) return"
        )

    # Call to init_thread_local_call_shape_info that unpacks the thread id into
    # a coordinate in the call shape, and stores the call shape info in thread-local storage
    context_args = "flat_call_thread_id"
    if call_data_len > 0:
        gp = "" if use_entrypoint_args else "call_data."
        if build_info.pipeline_type == PipelineType.compute:
            thread_arg = "flat_call_group_thread_id"
            group_arg = "flat_call_group_id"
        elif build_info.pipeline_type == PipelineType.ray_tracing:
            thread_arg = "0"
            group_arg = "uint3(0)"
        else:
            raise RuntimeError(f"Unknown pipeline type: {build_info.pipeline_type}")
        cg.kernel.append_statement(
            f"if (!init_thread_local_call_shape_info("
            f"{thread_arg}, {group_arg}, flat_call_thread_id, "
            f"{gp}_grid_stride, {gp}_grid_dim, {gp}_call_dim)) return"
        )
        context_args += ", CallShapeInfo::get_call_id().shape"

    # Define the core context.
    cg.kernel.append_statement(f"Context __slangpy_context__ = {{{context_args}}}")

    # Emit the trampoline call, passing the context and any entry-point args (if using them).
    fn = "_trampoline"
    if context.call_mode == CallMode.bwds:
        fn = f"bwd_diff({fn})"
    if use_entrypoint_args:
        trampoline_args = ["__slangpy_context__"]
        for x in root_params:
            if x.create_param_block:
                continue
            trampoline_args.append(x.variable_name)
        cg.kernel.append_statement(f"{fn}({', '.join(trampoline_args)})")
    else:
        cg.kernel.append_statement(f"{fn}(__slangpy_context__)")


def generate_code(
    context: "BindContext",
    build_info: "FunctionBuildInfo",
    signature: "BoundCall",
    cg: CodeGen,
) -> None:
    """Generate Slang kernel code for the given function call signature.

    Orchestrates all sub-steps: constants, shape params, call-data structs,
    trampoline, entry-point signature, and kernel body.
    """
    use_entrypoint_args = context.use_entrypoint_args
    cg.add_import("slangpy")
    call_data_len = context.call_dimensionality

    call_group_size, call_group_strides, call_group_shape_vector = (
        _validate_and_compute_group_shape(build_info, call_data_len)
    )

    cg.add_import(build_info.module.name)
    if use_entrypoint_args:
        cg.skip_call_data = True

    _emit_link_time_constants(
        cg, build_info, call_data_len, call_group_size, call_group_strides, call_group_shape_vector
    )
    _emit_shape_and_metadata_params(cg, call_data_len, use_entrypoint_args)
    _emit_call_data_definitions(cg, context, signature)

    root_params = sorted(signature.values(), key=lambda x: x.param_index)

    _emit_trampoline(cg, context, build_info, root_params, use_entrypoint_args)
    _emit_entry_point_signature(cg, build_info, call_data_len, call_group_size, use_entrypoint_args)
    cg.kernel.begin_block()
    _emit_kernel_body(cg, context, build_info, root_params, call_data_len, use_entrypoint_args)
    cg.kernel.end_block()
