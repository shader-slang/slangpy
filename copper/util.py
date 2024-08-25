from typing import Optional


def broadcast_shapes(shapes: list[tuple[int, ...]]) -> Optional[tuple[int, ...]]:
    max_batch_dim = max(len(shape) for shape in shapes)
    padded_shapes = [(1,) * (max_batch_dim - len(shape)) + shape for shape in shapes]
    broadcast_shape = tuple(max(dims) for dims in zip(*padded_shapes))

    for shape in padded_shapes:
        if any(a != 1 and a != b for a, b in zip(shape, broadcast_shape)):
            return None
    return broadcast_shape
