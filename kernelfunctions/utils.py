from typing import Union

from kernelfunctions.backend import ProgramLayout, TypeLayoutReflection, TypeReflection, DeclReflection


def find_type_layout_for_buffer(program_layout: ProgramLayout, slang_type: Union[str, TypeReflection, TypeLayoutReflection]):
    if isinstance(slang_type, str):
        slang_type_name = slang_type
    elif isinstance(slang_type, (TypeReflection, TypeLayoutReflection)):
        slang_type_name = slang_type.name
    buffer_type = program_layout.find_type_by_name(f"StructuredBuffer<{slang_type_name}>")
    buffer_layout = program_layout.get_type_layout(buffer_type)
    return buffer_layout.element_type_layout


def try_find_type_decl(root: DeclReflection, type_name: str):

    type_names = type_name.split("::")

    type_decl = root
    while len(type_names) > 0:
        type_name = type_names.pop(0)
        type_decl = type_decl.find_first_child_of_kind(
            DeclReflection.Kind.struct, type_name)
        if type_decl is None:
            return None

    return type_decl


def try_find_type_via_ast(root: DeclReflection, type_name: str):
    type_decl = try_find_type_decl(root, type_name)
    return type_decl.as_type() if type_decl is not None else None


def try_find_function_overloads_via_ast(root: DeclReflection, type_name: str, func_name: str):

    type_decl = try_find_type_decl(root, type_name)
    if type_decl is None:
        return (None, None)

    func_decls = type_decl.find_children_of_kind(DeclReflection.Kind.func, func_name)
    return (type_decl.as_type(), [x.as_function() for x in func_decls])
