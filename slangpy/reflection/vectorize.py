# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy.reflection.reflectiontypes as rt


def scalar_to_scalar(marshall_type: rt.SlangType, target_type: rt.SlangType):
    """Attempt to match marshall scalar type to target scalar type, allowing for target being generic."""
    if not isinstance(marshall_type, rt.ScalarType):
        return None
    if isinstance(target_type, rt.ScalarType):
        if marshall_type.slang_scalar_type == target_type.slang_scalar_type:
            return marshall_type
    if isinstance(target_type, rt.UnknownType):
        return marshall_type
    return None


def scalar_to_scalar_convertable(marshall_type: rt.SlangType, target_type: rt.SlangType):
    """Looser version of to_scalar that allows for implicit conversions, used by pure scalar marshallers."""
    if not isinstance(marshall_type, rt.ScalarType):
        return None
    if isinstance(target_type, rt.ScalarType):
        return target_type
    if isinstance(target_type, rt.UnknownType):
        return marshall_type
    return None


def scalar_to_pointer(marshall_type: rt.SlangType, target_type: rt.SlangType):
    """Looser version of to_scalar that allows for implicit conversions, used by pure scalar marshallers."""
    if not isinstance(marshall_type, rt.ScalarType):
        return None
    if not marshall_type.slang_scalar_type in (
        rt.TR.ScalarType.uint64,
        rt.TR.ScalarType.int64,
        rt.TR.ScalarType.int32,
        rt.TR.ScalarType.uint32,
    ):
        return None
    if isinstance(target_type, rt.PointerType) and not target_type.is_generic:
        return target_type
    return None


def _array_shapes_match(array_1: rt.ArrayType, array_2: rt.ArrayType):
    shape_1 = array_1.array_shape
    shape_2 = array_2.array_shape
    if len(shape_1.as_tuple()) != len(shape_2.as_tuple()):
        return False
    for dim_1, dim_2 in zip(shape_1.as_tuple(), shape_2.as_tuple()):
        if dim_1 > 0 and dim_2 > 0 and dim_1 != dim_2:
            return False
    return True


def _array_type_name(element_type: rt.SlangType, shape: rt.Shape):
    return element_type.full_name + "".join(f"[{x}]" for x in shape.as_tuple())


def array_to_array(marshall_type: rt.SlangType, target_type: rt.SlangType):
    """Attempt to match marshall vector type to target vector type, allowing for generic element/dims"""
    if not isinstance(marshall_type, rt.ArrayType):
        return None
    if isinstance(target_type, rt.ArrayType):
        if (
            not isinstance(target_type.element_type, rt.UnknownType)
            and marshall_type.element_type.full_name != target_type.element_type.full_name
        ):
            return None
        if not _array_shapes_match(marshall_type, target_type):
            return None
        return marshall_type
    return None


def array_to_array_scalarconvertable(marshall_type: rt.SlangType, target_type: rt.SlangType):
    """Attempt to match marshall vector type to target vector type, allowing for generic element/dims.
    This looser version allows for conversions of scalar element types to support passing python lists
    of numbers. To do so, when 2 scalar element types are found, a new array type is constructed with
    the target scalar type as element type and the shape of the marshall type."""
    if not isinstance(marshall_type, rt.ArrayType):
        return None
    if isinstance(target_type, rt.ArrayType):
        if not _array_shapes_match(marshall_type, target_type):
            return None
        if isinstance(target_type.element_type, rt.ScalarType) and isinstance(
            marshall_type.element_type, rt.ScalarType
        ):
            array_name = _array_type_name(target_type.element_type, marshall_type.array_shape)
            return marshall_type.program.find_type_by_name(array_name)
        elif (
            isinstance(target_type.element_type, rt.UnknownType)
            or marshall_type.element_type.full_name == target_type.element_type.full_name
        ):
            return marshall_type
    return None


def vector_to_vector(marshall_type: rt.SlangType, target_type: rt.SlangType):
    """Attempt to match marshall vector type to target vector type, allowing for generic element/dims"""
    if not isinstance(marshall_type, rt.VectorType):
        return None
    if isinstance(target_type, rt.VectorType):
        if (
            isinstance(target_type.element_type, rt.ScalarType)
            and marshall_type.slang_scalar_type != target_type.slang_scalar_type
        ):
            return None
        if target_type.num_elements > 0 and marshall_type.num_elements != target_type.num_elements:
            return None
        return marshall_type
    return None


def scalar_to_sized_vector(marshall_element_type: rt.SlangType, target_type: rt.SlangType):
    """Attempt to create a vector type from marshall scalar type to match target vector type. Used by
    containers that can hold a scalar and load vectors."""
    if not isinstance(marshall_element_type, rt.ScalarType):
        return None
    if isinstance(target_type, rt.VectorType):
        if (
            isinstance(target_type.element_type, rt.ScalarType)
            and marshall_element_type.slang_scalar_type != target_type.slang_scalar_type
        ):
            return None
        if target_type.num_elements == 0:
            return None
        return marshall_element_type.program.vector_type(
            marshall_element_type.slang_scalar_type, target_type.num_elements
        )
    return None


def container_to_generic_array_candidates(
    marshall_element_type: rt.SlangType, target_type: rt.SlangType
):
    results = []
    if isinstance(target_type, rt.ArrayType) and isinstance(
        target_type.element_type, rt.UnknownType
    ):
        array_shape = target_type.array_shape.as_tuple()
        results.append(marshall_element_type)
        if len(array_shape) >= 1 and array_shape[0] >= 1:
            array_name = f"{marshall_element_type.full_name}[{array_shape[0]}]"
            results.append(marshall_element_type.program.find_type_by_name(array_name))
        if len(array_shape) >= 2 and array_shape[0] >= 1 and array_shape[1] >= 1:
            array_name = f"{marshall_element_type.full_name}[{array_shape[0]}][{array_shape[1]}]"
            results.append(marshall_element_type.program.find_type_by_name(array_name))
    results = [r for r in results if r]
    if len(results) == 0:
        return None
    return results


def container_to_sized_array(
    marshall_element_type: rt.SlangType, target_type: rt.SlangType, max_dims: int
):
    """Attempt to create a vector type from marshall scalar type to match target vector type. Used by
    containers that can hold a scalar and load arrays."""
    if isinstance(target_type, rt.ArrayType):
        array_shape = target_type.array_shape
        if len(array_shape) > max_dims:
            return None
        for dim in array_shape.as_tuple():
            if dim == 0:
                return None
        array_name = _array_type_name(marshall_element_type, array_shape)
        return marshall_element_type.program.find_type_by_name(array_name)

    return None


def container_to_structured_buffer(
    marshall_element_type: rt.SlangType, marshall_rw: bool, target_type: rt.SlangType
):
    """Attempt to create a structured buffer type from marshall element type to match target structured buffer type."""
    if isinstance(target_type, rt.StructuredBufferType):
        if target_type.writable and not marshall_rw:
            return None
        if (
            not isinstance(target_type.element_type, rt.UnknownType)
            and marshall_element_type.full_name != target_type.element_type.full_name
        ):
            return None
        return marshall_element_type.program.find_type_by_name(
            f"{target_type.name}<{marshall_element_type.full_name}>"
        )
    return None


def container_to_byte_address_buffer(
    marshall_element_type: rt.SlangType, marshall_rw: bool, target_type: rt.SlangType
):
    """Attempt to create a byte address buffer type from marshall element type to match target byte address buffer type."""
    if isinstance(target_type, rt.ByteAddressBufferType):
        if target_type.writable and not marshall_rw:
            return None
        return target_type
    return None


def container_to_pointer(marshall_element_type: rt.SlangType, target_type: rt.SlangType):
    """Attempt to create a pointer type from marshall element type to match target pointer type.
    Also handles the container being a buffer of uint64 with a concrete pointer type"""
    if isinstance(target_type, rt.PointerType):
        if (
            not target_type.is_generic
            and isinstance(marshall_element_type, rt.ScalarType)
            and marshall_element_type.slang_scalar_type == rt.TR.ScalarType.uint64
        ):
            return target_type
        if (
            not target_type.is_generic
            and marshall_element_type.full_name != target_type.target_type.full_name
        ):
            return None
        return marshall_element_type.program.find_type_by_name(
            f"Ptr<{marshall_element_type.full_name}>"
        )
    return None
