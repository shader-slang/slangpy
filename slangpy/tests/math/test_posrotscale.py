# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from slangpy import float3, quatf, posrotscalef
from slangpy.math import (
    posrotscale_from_translation,
    posrotscale_from_rotation,
    posrotscale_from_scaling,
    posrotscale_from_uniform_scaling,
    transform_point,
)


def test_default_constructor():
    prs = posrotscalef()
    assert prs.pos == float3(0.0, 0.0, 0.0)
    assert prs.rot == quatf.identity()
    assert prs.scale == float3(1.0, 1.0, 1.0)


def test_position_constructor():
    pos = float3(1.0, 2.0, 3.0)
    prs = posrotscalef(pos)
    assert prs.pos == pos
    assert prs.rot == quatf.identity()
    assert prs.scale == float3(1.0, 1.0, 1.0)


def test_rotation_constructor():
    rot = quatf(0.0, 0.0, 0.7071067, 0.7071068)
    prs = posrotscalef(rot)
    assert prs.pos == float3(0.0, 0.0, 0.0)
    assert prs.rot == rot
    assert prs.scale == float3(1.0, 1.0, 1.0)


def test_scale_constructor():
    scale = float3(2.0, 3.0, 4.0)
    prs = posrotscale_from_scaling(scale)
    assert prs.pos == float3(0.0, 0.0, 0.0)
    assert prs.rot == quatf.identity()
    assert prs.scale == scale


def test_full_constructor():
    pos = float3(1.0, 2.0, 3.0)
    rot = quatf(0.0, 0.0, 0.7071067, 0.7071068)
    scale = float3(2.0, 3.0, 4.0)
    prs = posrotscalef(pos, rot, scale)
    assert prs.pos == pos
    assert prs.rot == rot
    assert prs.scale == scale


def test_static_constructors():
    # Identity
    prs = posrotscalef.identity()
    assert prs.pos == float3(0.0, 0.0, 0.0)
    assert prs.rot == quatf.identity()
    assert prs.scale == float3(1.0, 1.0, 1.0)

    # Translation
    pos = float3(1.0, 2.0, 3.0)
    prs = posrotscale_from_translation(pos)
    assert prs.pos == pos
    assert prs.rot == quatf.identity()
    assert prs.scale == float3(1.0, 1.0, 1.0)

    # Rotation
    rot = quatf(0.0, 0.0, 0.7071067, 0.7071068)
    prs = posrotscale_from_rotation(rot)
    assert prs.pos == float3(0.0, 0.0, 0.0)
    assert prs.rot == rot
    assert prs.scale == float3(1.0, 1.0, 1.0)

    # Scaling
    scale = float3(2.0, 3.0, 4.0)
    prs = posrotscale_from_scaling(scale)
    assert prs.pos == float3(0.0, 0.0, 0.0)
    assert prs.rot == quatf.identity()
    assert prs.scale == scale

    # Uniform scaling
    prs = posrotscale_from_uniform_scaling(2.5)
    assert prs.pos == float3(0.0, 0.0, 0.0)
    assert prs.rot == quatf.identity()
    assert prs.scale == float3(2.5, 2.5, 2.5)


def test_equality_operators():
    prs1 = posrotscalef(float3(1.0, 2.0, 3.0), quatf.identity(), float3(2.0, 2.0, 2.0))
    prs2 = posrotscalef(float3(1.0, 2.0, 3.0), quatf.identity(), float3(2.0, 2.0, 2.0))
    prs3 = posrotscalef(float3(2.0, 2.0, 3.0), quatf.identity(), float3(2.0, 2.0, 2.0))

    assert prs1 == prs2
    assert prs1 != prs3


def test_transform_operations():
    # Transform point
    t = posrotscalef(float3(1.0, 2.0, 3.0), quatf.identity(), float3(2.0, 1.0, 0.5))
    point = float3(1.0, 1.0, 2.0)
    result = transform_point(t, point)

    # Scale: (1,1,2) -> (2,1,1)
    # Rotate: identity, so (2,1,1)
    # Translate: (2,1,1) + (1,2,3) = (3,3,4)
    assert result == float3(3.0, 3.0, 4.0)


def test_inverse():
    t = posrotscalef(float3(1.0, 2.0, 3.0), quatf.identity(), float3(2.0, 0.5, 4.0))
    inv = t.inverse()
    identity_check = t * inv

    # Should be close to identity
    assert abs(identity_check.pos.x) < 1e-5
    assert abs(identity_check.pos.y) < 1e-5
    assert abs(identity_check.pos.z) < 1e-5
    assert abs(identity_check.scale.x - 1.0) < 1e-5
    assert abs(identity_check.scale.y - 1.0) < 1e-5
    assert abs(identity_check.scale.z - 1.0) < 1e-5


def test_matrix_conversions():
    # to_matrix3x4
    t = posrotscalef(float3(1.0, 2.0, 3.0), quatf.identity(), float3(2.0, 3.0, 4.0))
    m = t.to_matrix3x4()

    assert m[0, 3] == 1.0  # translation x
    assert m[1, 3] == 2.0  # translation y
    assert m[2, 3] == 3.0  # translation z

    # Identity rotation with scale should give scaled identity 3x3 part
    assert abs(m[0, 0] - 2.0) < 1e-6
    assert abs(m[1, 1] - 3.0) < 1e-6
    assert abs(m[2, 2] - 4.0) < 1e-6

    # to_matrix4x4
    m4 = t.to_matrix4x4()
    assert m4[0, 3] == 1.0
    assert m4[1, 3] == 2.0
    assert m4[2, 3] == 3.0
    assert m4[3, 3] == 1.0

    # Identity rotation with scale
    assert abs(m4[0, 0] - 2.0) < 1e-6
    assert abs(m4[1, 1] - 3.0) < 1e-6
    assert abs(m4[2, 2] - 4.0) < 1e-6


def test_interpolation():
    # lerp
    a = posrotscalef(float3(0.0, 0.0, 0.0), quatf.identity(), float3(1.0, 1.0, 1.0))
    b = posrotscalef(float3(2.0, 2.0, 2.0), quatf.identity(), float3(3.0, 3.0, 3.0))

    mid = a.lerp(b, 0.5)
    assert abs(mid.pos.x - 1.0) < 1e-6
    assert abs(mid.pos.y - 1.0) < 1e-6
    assert abs(mid.pos.z - 1.0) < 1e-6
    assert abs(mid.scale.x - 2.0) < 1e-6
    assert abs(mid.scale.y - 2.0) < 1e-6
    assert abs(mid.scale.z - 2.0) < 1e-6


def test_normalize():
    # Create a posrotscale with a non-normalized quaternion
    t = posrotscalef(float3(1.0, 2.0, 3.0), quatf(0.0, 0.0, 1.0, 1.0), float3(2.0, 3.0, 4.0))
    normalized = t.normalize()

    # Check that position and scale are unchanged
    assert normalized.pos == t.pos
    assert normalized.scale == t.scale

    # Check that rotation is normalized
    quat_length = (
        normalized.rot.x**2 + normalized.rot.y**2 + normalized.rot.z**2 + normalized.rot.w**2
    ) ** 0.5
    assert abs(quat_length - 1.0) < 1e-6


def test_repr():
    t = posrotscalef(float3(1.0, 2.0, 3.0), quatf.identity(), float3(2.0, 3.0, 4.0))
    repr_str = repr(t)
    assert "posrotscalef" in repr_str
    assert "pos=" in repr_str
    assert "rot=" in repr_str
    assert "scale=" in repr_str
