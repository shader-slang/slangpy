from typing import Optional, TypedDict

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
