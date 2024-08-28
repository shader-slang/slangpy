from typing import Optional

TConcreteShape = tuple[int, ...]
TConcreteOrUndefinedShape = Optional[TConcreteShape]
TLooseShape = tuple[Optional[int], ...]
TLooseOrUndefinedShape = Optional[TLooseShape]


# Given the shapes of the parameters (inferred from reflection) and inputs (passed in by the user), calculates the
# argument shapes and call shape.
# All parameters must have a shape, however individual dimensions can have an undefined size (None)
# Inputs can also be fully undefined
def calculate_argument_shapes(
    param_type_shapes: list[TLooseShape],
    input_shapes: list[TLooseOrUndefinedShape],
    input_remaps: Optional[list[TConcreteOrUndefinedShape]] = None,
):

    type_shapes: list[list[Optional[int]]] = []
    arg_shapes: list[Optional[list[Optional[int]]]] = []
    max_dims = 0

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
            new_param_type_shape: list[Optional[int]] = []
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
                else:
                    new_param_type_shape.append(input_dim_size)
            new_param_type_shape.reverse()
            type_shapes.append(new_param_type_shape)

            # Argment shape is what's left of the input shape
            arg_shape = list(input_shape)[: input_len - type_len]
            max_dims = max(max_dims, len(arg_shape))
            arg_shapes.append(arg_shape)
        else:
            # If input not defined, parameter shape is the argument shape
            type_shapes.append(list(param_type_shape))
            arg_shapes.append(None)

    # Call shape has the number of dimensions that the largest argument has
    call_shape: list[Optional[int]] = [None for _ in range(max_dims)]

    # Numpy rules for calculating broadcast dimension sizes, with additional
    # rules for handling undefined dimensions
    call_dim_end = max_dims - 1
    for arg_index, arg_shape in enumerate(arg_shapes):
        if arg_shape is not None:
            arg_dims = len(arg_shape)
            arg_dim_end = arg_dims - 1
            for i in range(arg_dims):
                call_dim_idx = call_dim_end - i
                arg_dim_idx = arg_dim_end - i
                call_dim_size = call_shape[call_dim_idx]
                arg_dim_size = arg_shape[arg_dim_idx]
                if call_dim_size is None:
                    call_dim_size = arg_dim_size
                elif call_dim_size == 1:
                    call_dim_size = arg_dim_size
                elif arg_dim_size is not None and call_dim_size != arg_dim_size:
                    raise ValueError(
                        f"Arg {arg_index}, CS[{call_dim_idx}] != AS[{arg_dim_idx}], {call_dim_size} != {arg_dim_size}"
                    )
                call_shape[call_dim_idx] = call_dim_size

    # Assign the call shape to any fully undefined argument shapes
    for i in range(len(arg_shapes)):
        if arg_shapes[i] is None:
            arg_shapes[i] = call_shape

    # Raise an error if the call shape is still undefined
    if None in call_shape:
        raise ValueError(f"Call shape is ambiguous: {call_shape}")

    # Populate any still-undefined argument shapes from the call shape
    for arg_index, arg_shape in enumerate(arg_shapes):
        if arg_shape is not None:
            arg_dims = len(arg_shape)
            arg_dim_end = arg_dims - 1
            for i in range(arg_dims):
                call_dim_idx = call_dim_end - i
                arg_dim_idx = arg_dim_end - i
                if arg_shape[arg_dim_idx] is None:
                    arg_shape[arg_dim_idx] = call_shape[call_dim_idx]
            if None in arg_shape:
                raise ValueError(f"Arg {arg_index} shape is ambiguous: {arg_shape}")

    return {
        "type_shapes": type_shapes,
        "arg_shapes": arg_shapes,
        "call_shape": call_shape,
    }
