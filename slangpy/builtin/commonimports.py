# pyright: reportUnusedImport=false

# Common requirements for internal bindings to save massive import pain
from slangpy.bindings import CodeGenBlock, BindContext, ReturnContext, BaseType, BaseTypeImpl, BoundVariable, PYTHON_TYPES, PYTHON_SIGNATURES, BoundVariableRuntime, get_or_create_type
from slangpy.reflection import SlangProgramLayout, SlangType, TYPE_OVERRIDES, is_matching_array_type
from slangpy.types import NDBuffer, NDDifferentiableBuffer
from slangpy.backend import ResourceUsage, Buffer, ResourceView, ResourceViewType, Texture, ResourceType, FormatType, get_format_info
from slangpy.core.native import Shape, CallContext, AccessType, TypeReflection
from slangpy.core.enums import PrimType, IOType

import slangpy.reflection as kfr
import slangpy.bindings.typeregistry as tr
