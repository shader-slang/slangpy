from typing import Optional, TypedDict, cast

TConcreteShape = tuple[int, ...]
TConcreteOrUndefinedShape = Optional[TConcreteShape]
TLooseShape = tuple[Optional[int], ...]
TLooseOrUndefinedShape = Optional[TLooseShape]

TArgShapesResult = TypedDict(
    "TArgShapesResult",
    {
        "type_shapes": list[list[int]],
        "arg_shapes": list[list[int]],
        "call_shape": list[int],
    },
)


# Given the shapes of the parameters (inferred from reflection) and inputs (passed in by the user), calculates the
# argument shapes and call shape.
# All parameters must have a shape, however individual dimensions can have an undefined size (None)
# Inputs can also be fully undefined
def calculate_argument_shapes(
    param_type_shapes: list[TLooseShape],
    input_shapes: list[TLooseOrUndefinedShape],
    input_transforms: Optional[list[TConcreteOrUndefinedShape]] = None,
    call_transforms: Optional[list[TConcreteOrUndefinedShape]] = None,
) -> TArgShapesResult:
    # Break the input shapes into type shapes and argument shapes
    (type_shapes, arg_shapes) = _split_type_and_argument_shapes(
        param_type_shapes, input_shapes, input_transforms
    )

    # Find the highest dimensionality of the argument shapes
    highest_output_dimensionality = 0
    for arg_shape in arg_shapes:
        if arg_shape is not None:
            highest_output_dimensionality = max(
                highest_output_dimensionality, len(arg_shape)
            )

    # Define a default function transform which basically maps argument
    # dimensions to call dimensions 1-1, with a bit of extra work to handle
    # arguments that aren't the same size or shapes that aren't defined.
    # This is effectively what numpy does.
    call_dim_end = highest_output_dimensionality - 1
    transforms: list[Optional[list[int]]] = []
    for arg_index, arg_shape in enumerate(arg_shapes):
        if arg_shape is not None:
            arg_dims = len(arg_shape)
            transforms.append([call_dim_end - i for i in range(arg_dims)])
        else:
            transforms.append(None)

    # Inject any custom transforms for the call shape
    if call_transforms is not None:
        for arg_index, arg_shape in enumerate(arg_shapes):
            if arg_shape is not None:
                call_remap = call_transforms[arg_index]
                if call_remap is not None:
                    if len(call_remap) != arg_dims:
                        raise ValueError(
                            f"Call remap {call_remap} must have the same number of dimensions as the argument shape {arg_shape}"
                        )
                    arg_dims = len(arg_shape)
                    transform = transforms[arg_index]
                    assert transform is not None
                    for i in range(arg_dims):
                        if call_remap[i] is not None:
                            transform[i] = call_remap[i]

    # Find the highest dimension in the mappings. Note: for a purely scalar
    # call, highest dimensionality can be 0, so we start at -1.
    highest_output_dimensionality = -1
    for transform in transforms:
        if transform is not None:
            for dim in transform:
                highest_output_dimensionality = max(highest_output_dimensionality, dim)

    # Call shape has the number of dimensions that the largest argument has
    call_shape: list[Optional[int]] = [
        None for _ in range(highest_output_dimensionality + 1)
    ]

    # Numpy rules for calculating broadcast dimension sizes, with additional
    # rules for handling undefined dimensions
    for arg_index, arg_shape in enumerate(arg_shapes):
        if arg_shape is not None:
            arg_dims = len(arg_shape)
            transform = transforms[arg_index]
            assert transform is not None
            for arg_dim_idx in range(arg_dims):
                call_dim_idx = transform[arg_dim_idx]
                arg_dim_size = arg_shape[arg_dim_idx]
                call_dim_size = call_shape[call_dim_idx]
                if call_dim_size is None:
                    call_dim_size = arg_dim_size
                elif call_dim_size == 1:
                    call_dim_size = arg_dim_size
                elif arg_dim_size == 1:
                    pass  # call dim already set and arg dim is 1 so can be broadcast
                elif arg_dim_size is not None and call_dim_size != arg_dim_size:
                    raise ValueError(
                        f"Arg {arg_index}, CS[{call_dim_idx}] != AS[{arg_dim_idx}], {call_dim_size} != {arg_dim_size}"
                    )
                call_shape[call_dim_idx] = call_dim_size

    # Assign the call shape to any fully undefined argument shapes
    for i in range(len(arg_shapes)):
        if arg_shapes[i] is None:
            arg_shapes[i] = call_shape
            transforms[i] = [i for i in range(len(call_shape))]

    # Raise an error if the call shape is still undefined
    if None in call_shape:
        raise ValueError(f"Call shape is ambiguous: {call_shape}")
    verified_call_shape = cast(list[int], call_shape)

    # Populate any still-undefined argument shapes from the call shape
    for arg_index, arg_shape in enumerate(arg_shapes):
        assert arg_shape is not None
        arg_dims = len(arg_shape)
        transform = transforms[arg_index]
        assert transform is not None
        for arg_dim_idx in range(arg_dims):
            call_dim_idx = transform[arg_dim_idx]
            if arg_shape[arg_dim_idx] is None:
                arg_shape[arg_dim_idx] = verified_call_shape[call_dim_idx]
        if None in arg_shape:
            raise ValueError(f"Arg {arg_index} shape is ambiguous: {arg_shape}")
    verified_arg_shapes = cast(list[list[int]], arg_shapes)

    return {
        "type_shapes": type_shapes,
        "arg_shapes": verified_arg_shapes,
        "call_shape": verified_call_shape,
    }


def _split_type_and_argument_shapes(
    param_type_shapes: list[TLooseShape],
    input_shapes: list[TLooseOrUndefinedShape],
    input_remaps: Optional[list[TConcreteOrUndefinedShape]] = None,
):

    type_shapes: list[list[int]] = []
    arg_shapes: list[Optional[list[Optional[int]]]] = []

    # Iterate over each pair of parameter type shape and input shape.
    for param_idx in range(len(param_type_shapes)):
        # Get both and check input is defined.
        param_type_shape = param_type_shapes[param_idx]
        input_shape = input_shapes[param_idx]
        if input_shape is not None:
            # If input defined, it must have enough dimensions to contain the parameter
            if len(input_shape) < len(param_type_shape):
                raise ValueError(
                    f"Input shape {input_shape} is large enough to contain parameter type shape {param_type_shape}"
                )

            # Optionally use the input remap to re-order input dimensions
            if input_remaps is not None:
                input_remap = input_remaps[param_idx]
                if input_remap is not None:
                    if len(input_remap) != len(input_shape):
                        raise ValueError(
                            f"Input remap {input_remap} must have the same number of dimensions as the input shape {input_shape}"
                        )
                    input_shape = [input_shape[i] for i in input_remap]

            # Verify / build concrete shaping
            # - where both are defined they must match
            # - where param is defined and input is not, set input to param
            # - where input is defined and param is not, set param to input
            type_len = len(param_type_shape)
            input_len = len(input_shape)
            type_end = type_len - 1
            input_end = input_len - 1
            new_param_type_shape: list[int] = []
            for i in range(type_len):
                param_dim_idx = type_end - i
                input_dim_idx = input_end - i
                param_dim_size = param_type_shape[param_dim_idx]
                input_dim_size = input_shape[input_dim_idx]
                if param_dim_size is not None and input_dim_size is not None:
                    if param_dim_size != input_dim_size:
                        raise ValueError(
                            f"Arg {param_idx}, PS[{param_dim_idx}] != IS[{input_dim_idx}], {param_dim_size} != {input_dim_size}"
                        )
                    new_param_type_shape.append(param_dim_size)
                elif param_dim_size is not None:
                    new_param_type_shape.append(param_dim_size)
                elif input_dim_size is not None:
                    new_param_type_shape.append(input_dim_size)
                else:
                    raise ValueError(f"Arg {param_idx} type shape is ambiguous")
            new_param_type_shape.reverse()
            type_shapes.append(new_param_type_shape)

            # Argment shape is what's left of the input shape
            arg_shape = list(input_shape)[: input_len - type_len]
            arg_shapes.append(arg_shape)
        else:
            # If input not defined, parameter shape is the argument shape
            if None in param_type_shape:
                raise ValueError(f"Arg {param_idx} type shape is ambiguous")
            type_shapes.append(cast(list[int], list(param_type_shape)))
            arg_shapes.append(None)

    return type_shapes, arg_shapes


def build_indexer(call_shape: list[int], arg_shape: list[int]):
    # Build the index expression
    index_expr = ""
    call_offset = len(call_shape) - len(arg_shape)
    for i in range(len(arg_shape)):
        if i < len(arg_shape):
            if arg_shape[i] == 1:
                index_expr += "0"
            else:
                index_expr += f"call_id[{call_offset+i}]"
        if i < len(arg_shape) - 1:
            index_expr += ", "
    return index_expr
