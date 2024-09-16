from typing import Union
from sgl import ProgramLayout, TypeLayoutReflection, TypeReflection


def find_type_layout_for_buffer(program_layout: ProgramLayout, slang_type: Union[str, TypeReflection, TypeLayoutReflection]):
    if isinstance(slang_type, str):
        slang_type_name = slang_type
    elif isinstance(slang_type, (TypeReflection, TypeLayoutReflection)):
        slang_type_name = slang_type.name
    buffer_type = program_layout.find_type_by_name(f"StructuredBuffer<{slang_type_name}>")
    buffer_layout = program_layout.get_type_layout(buffer_type)
    return buffer_layout.element_type_layout
