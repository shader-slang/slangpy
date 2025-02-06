# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from slangpy.backend import TypeReflection
from slangpy.bindings.marshall import BindContext
from slangpy.core.utils import is_type_castable_on_host
from slangpy.reflection.reflectiontypes import ScalarType, SlangType, VectorType


def resolve_vector_generator_type(context: BindContext, bound_type: SlangType, fixed_dims: int, scalar_type: TypeReflection.ScalarType):
    """
    General purpose helper for simple generators that can output to vector or scalar types.
    """
    if isinstance(bound_type, VectorType):
        # Vector type - check match+conversions before just returning the bound type
        if bound_type.num_elements > 3:
            raise ValueError(
                f"Argument must be a vector of size 1, 2 or 3. Got {bound_type.num_elements}.")
        if fixed_dims != -1 and fixed_dims != bound_type.num_elements:
            raise ValueError(
                f"Argument must be a vector of size {fixed_dims}. Got {bound_type.num_elements}.")
        resolved_type = context.layout.vector_type(
            scalar_type, bound_type.shape[0])
        if not is_type_castable_on_host(resolved_type, bound_type):
            raise ValueError(
                f"Unable to convert argument of type {resolved_type.full_name} to {bound_type.full_name}.")
        return bound_type
    elif isinstance(bound_type, ScalarType):
        # Scalar type - check match+conversions before just returning the bound type
        if fixed_dims != -1 and fixed_dims != 1:
            raise ValueError(
                f"Argument must be a scalar or vector of size {fixed_dims}. Got {bound_type.full_name}.")
        resolved_type = context.layout.scalar_type(scalar_type)
        if not is_type_castable_on_host(resolved_type, bound_type):
            raise ValueError(
                f"Unable to convert argument of type {resolved_type.full_name} to {bound_type.full_name}.")
        return bound_type
    elif fixed_dims > 0:
        # Unknown type, but an explicit number of dimensions was passed so can still
        # resolve to a concrete type.
        return context.layout.vector_type(scalar_type, fixed_dims)
    else:
        # Unknown type, and no explicit dimensions passed, so can't resolve to a concrete type.
        raise ValueError(
            f"Argument must be a scalar or vector type. Got {bound_type.full_name}.")
