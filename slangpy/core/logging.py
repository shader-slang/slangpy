# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

from slangpy.backend import FunctionReflection, ModifierID, VariableReflection

if TYPE_CHECKING:
    from slangpy.bindings.marshall import Marshall
    from slangpy.bindings.boundvariable import BoundCall, BoundVariable
    from slangpy.bindings.boundvariableruntime import (BoundCallRuntime,
                                                       BoundVariableRuntime)


class TableColumn:
    def __init__(self, name: str, width: int, id: Union[str, Callable[[Any], str]]):
        super().__init__()
        self.name = name
        if isinstance(id, str):
            self.id = lambda x: x[id] if isinstance(x, dict) else str(getattr(x, id))
        else:
            self.id = id
        self.width = width


def generate_table(columns: list[TableColumn], data: list[Any], children_id: Optional[Callable[[Any], Optional[list[Any]]]], highlight: Optional[Any] = None, filter: Optional[dict[str, bool]] = None):

    if filter is not None:
        columns = [c for c in columns if c.name in filter and filter[c.name]]

    column_names = [c.name for c in columns]
    column_widths = [c.width for c in columns]

    # Calculate the width of the table
    table_width = sum(column_widths) + (len(column_names) - 1) * 3

    # Generate the header
    header = " | ".join([f"{c.name:<{c.width}}" for c in columns])
    header_line = "-" * table_width

    # Generate the rows
    if children_id is None:
        children_id = lambda x: None
    rows = _generate_table_recurse(data, columns, 0, children_id, highlight)

    # Generate the table
    table = "\n".join([header, header_line] + rows)
    return table


def _fmt(value: Any, width: int) -> str:
    value = str(value)
    if len(value) > width:
        return value[:width-3] + "..."
    else:
        return value + " "*(width-len(value))


def _generate_table_recurse(data: list[Any], columns: list[TableColumn], depth: int, children_id: Callable[[Any], Optional[list[Any]]], highlight: Optional[Any]):
    rows = []
    for row in data:

        cols = []

        cols.append(" "*depth*2 + _fmt(columns[0].id(row), columns[0].width-depth*2))

        for i in range(1, len(columns)):
            cols.append(_fmt(columns[i].id(row), columns[i].width))

        row_str = " | ".join(cols)

        if row == highlight:
            row_str = f"\033[1;32;40m{row_str}\033[0m"

        rows.append(row_str)
        children = children_id(row)
        if children is not None:
            rows += _generate_table_recurse(children, columns,
                                            depth + 1, children_id, highlight)
    return rows


def _pyarg_name(value: Any) -> str:
    if value == '':
        return '<posarg>'
    return value


def _type_name(value: Optional['Marshall']) -> str:
    if value is None:
        return ""

    st = getattr(value, 'slang_type', None)
    if st is not None:
        return st.full_name

    nm = getattr(value, 'full_name', None)
    if nm is not None:
        return nm

    nm = getattr(value, 'name', None)
    if nm is not None:
        return nm

    return str(value)


def _type_shape(value: Optional['Marshall']) -> str:
    if value is None:
        return ""
    return str(value.slang_type.shape)


def bound_variables_table(data: list['BoundVariable'], highlight: Optional['BoundVariable'] = None, filter: Optional[dict[str, bool]] = None):
    columns = [
        TableColumn("Name", 20, lambda x: _pyarg_name(x.name)),
        TableColumn("Index", 10, "param_index"),
        TableColumn("PyType", 30, lambda x: _type_name(x.python)),
        TableColumn("SlType", 30, lambda x: _type_name(x.slang_type)),
        TableColumn("VType", 30, lambda x: _type_name(x.vector_type)),
        TableColumn("Shape", 20, lambda x: _type_shape(x.python)),
        TableColumn("Call Dim", 10, lambda x: x.call_dimensionality),
        TableColumn("VMap", 20, lambda x: x.vector_mapping),
        TableColumn("Vector", 30, lambda x: x.vector_mapping if x.vector_mapping.valid else _type_name(
            x.vector_type))
    ]

    if filter is None:
        filter = {c.name: True for c in columns}
        filter['Vector'] = False

    table = generate_table(columns, data, lambda x: x.children.values()
                           if x.children is not None else None, highlight, filter)
    return table


def bound_call_table(data: 'BoundCall', highlight: Optional['BoundVariable'] = None, filter: Optional[dict[str, bool]] = None):
    return bound_variables_table(data.args+list(data.kwargs.values()), highlight, filter)


def bound_runtime_variables_table(data: list['BoundVariableRuntime'], highlight: Optional['BoundVariableRuntime'] = None, filter: Optional[dict[str, bool]] = None):
    columns = [
        TableColumn("Name", 20, lambda x: _pyarg_name(x._source_for_exceptions.name)),
        TableColumn("Index", 10, "param_index"),
        TableColumn("PyType", 30, lambda x: _type_name(x._source_for_exceptions.python)),
        TableColumn("SlType", 30, lambda x: _type_name(
            x._source_for_exceptions.slang_type)),
        TableColumn("VType", 30, lambda x: _type_name(
            x._source_for_exceptions.vector_type)),
        TableColumn("Shape", 20, lambda x: _type_shape(x._source_for_exceptions.python)),
        TableColumn("Call Dim", 10, lambda x: x._source_for_exceptions.call_dimensionality),
        TableColumn("VMap", 20, lambda x: x._source_for_exceptions.vector_mapping),
        TableColumn("Vector", 30, lambda x: x._source_for_exceptions.vector_mapping if x._source_for_exceptions.vector_mapping.valid else _type_name(
            x._source_for_exceptions.vector_type))
    ]

    if filter is None:
        filter = {c.name: True for c in columns}
        filter['Vector'] = False

    table = generate_table(columns, data, lambda x: x.children.values()
                           if x.children is not None else None, highlight, filter)
    return table


def bound_runtime_call_table(data: 'BoundCallRuntime', highlight: Optional['BoundVariableRuntime'] = None, filter: Optional[dict[str, bool]] = None):
    args = cast(list['BoundVariableRuntime'], data.args)
    kwargs = cast(dict[str, 'BoundVariableRuntime'], data.kwargs)
    return bound_runtime_variables_table(args+list(kwargs.values()), highlight, filter)


def function_reflection(slang_function: Optional[FunctionReflection]):
    if slang_function is None:
        return ""

    def get_modifiers(val: VariableReflection):
        mods: list[str] = []
        for m in ModifierID:
            if val.has_modifier(m):
                mods.append(m.name)
        return " ".join(mods)

    text: list[str] = []
    if slang_function.return_type is not None:
        text.append(f"{slang_function.return_type.full_name} ")
    else:
        text.append("void ")
    nm = slang_function.name
    if nm is None:
        nm = "<unknown>"
    text.append(nm)
    text.append("(")
    parms = [
        f"{get_modifiers(x)}{x.type.full_name} {x.name}" for x in slang_function.parameters]
    text.append(", ".join(parms))
    text.append(")")
    return "".join(text)


def mismatch_info(call: 'BoundCall', reflections: list[FunctionReflection]):
    text: list[str] = []

    text.append(f"Possible overloads:")
    if len(reflections) == 1 and reflections[0].is_overloaded:
        reflections = [x for x in reflections[0].overloads]
    for r in reflections:
        text.append(f"  {function_reflection(r)}")
    text.append("")
    text.append(f"Python arguments:")
    text.append(f"{bound_call_table(call)}")

    return "\n".join(text)


def bound_exception_info(call: 'BoundCall', concrete_reflection: FunctionReflection, variable: Optional['BoundVariable']):
    text: list[str] = []

    text.append(f"Selected overload:")
    text.append(f"{function_reflection(concrete_reflection)}")
    text.append("")
    text.append(f"Python arguments:")
    text.append(f"{bound_call_table(call,highlight=variable)}")

    return "\n".join(text)
