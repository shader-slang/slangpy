# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from slangpy import float3, quatf, posrotf


def test_default_constructor():
    pr = posrotf()
    assert pr.pos == float3(0.0, 0.0, 0.0)
    assert pr.rot == quatf.identity()


def test_position_constructor():
    pos = float3(1.0, 2.0, 3.0)
    pr = posrotf(pos)
    assert pr.pos == pos
    assert pr.rot == quatf.identity()


def test_rotation_constructor():
    rot = quatf(0.0, 0.0, 0.7071067, 0.7071068)  # 90 degrees around Z
    pr = posrotf(rot)
    assert pr.pos == float3(0.0, 0.0, 0.0)
    assert pr.rot == rot


def test_position_and_rotation_constructor():
    pos = float3(1.0, 2.0, 3.0)
    rot = quatf(0.0, 0.0, 0.7071067, 0.7071068)
    pr = posrotf(pos, rot)
    assert pr.pos == pos
    assert pr.rot == rot


def test_static_constructors():
    # Identity
    pr = posrotf.identity()
    assert pr.pos == float3(0.0, 0.0, 0.0)
    assert pr.rot == quatf.identity()

    # Translation
    pos = float3(1.0, 2.0, 3.0)
    pr = posrotf.translation(pos)
    assert pr.pos == pos
    assert pr.rot == quatf.identity()

    # Rotation
    rot = quatf(0.0, 0.0, 0.7071067, 0.7071068)
    pr = posrotf.from_rotation(rot)
    assert pr.pos == float3(0.0, 0.0, 0.0)
    assert pr.rot == rot


def test_equality_operators():
    pr1 = posrotf(float3(1.0, 2.0, 3.0), quatf.identity())
    pr2 = posrotf(float3(1.0, 2.0, 3.0), quatf.identity())
    pr3 = posrotf(float3(2.0, 2.0, 3.0), quatf.identity())

    assert pr1 == pr2
    assert pr1 != pr3


def test_transform_operations():
    # Multiply transforms
    t1 = posrotf.translation(float3(1.0, 0.0, 0.0))
    t2 = posrotf.translation(float3(2.0, 0.0, 0.0))
    result = t1 * t2
    assert abs(result.pos.x - 3.0) < 1e-6
    assert abs(result.pos.y - 0.0) < 1e-6
    assert abs(result.pos.z - 0.0) < 1e-6

    # Transform point
    t = posrotf.translation(float3(1.0, 2.0, 3.0))
    point = float3(1.0, 1.0, 1.0)
    result = t * point
    assert result == float3(2.0, 3.0, 4.0)


def test_inverse():
    t = posrotf(float3(1.0, 2.0, 3.0), quatf.identity())
    inv = t.inverse()
    identity_check = t * inv

    assert abs(identity_check.pos.x) < 1e-6
    assert abs(identity_check.pos.y) < 1e-6
    assert abs(identity_check.pos.z) < 1e-6
    assert abs(identity_check.rot.w - 1.0) < 1e-6


def test_matrix_conversions():
    # to_matrix3x4
    t = posrotf(float3(1.0, 2.0, 3.0), quatf.identity())
    m = t.to_matrix3x4()
    assert m[0, 3] == 1.0
    assert m[1, 3] == 2.0
    assert m[2, 3] == 3.0

    # Identity rotation should give identity 3x3 part
    assert abs(m[0, 0] - 1.0) < 1e-6
    assert abs(m[1, 1] - 1.0) < 1e-6
    assert abs(m[2, 2] - 1.0) < 1e-6

    # to_matrix4x4
    m4 = t.to_matrix4x4()
    assert m4[0, 3] == 1.0
    assert m4[1, 3] == 2.0
    assert m4[2, 3] == 3.0
    assert m4[3, 3] == 1.0


def test_interpolation():
    # lerp
    a = posrotf(float3(0.0, 0.0, 0.0), quatf.identity())
    b = posrotf(float3(2.0, 2.0, 2.0), quatf.identity())

    mid = a.lerp(b, 0.5)
    assert abs(mid.pos.x - 1.0) < 1e-6
    assert abs(mid.pos.y - 1.0) < 1e-6
    assert abs(mid.pos.z - 1.0) < 1e-6

    # slerp
    mid = a.slerp(b, 0.5)
    assert abs(mid.pos.x - 1.0) < 1e-6
    assert abs(mid.pos.y - 1.0) < 1e-6
    assert abs(mid.pos.z - 1.0) < 1e-6


def test_normalize():
    # Create a posrot with a non-normalized quaternion
    t = posrotf(float3(1.0, 2.0, 3.0), quatf(0.0, 0.0, 1.0, 1.0))
    normalized = t.normalize()

    # Check that position is unchanged
    assert normalized.pos == t.pos

    # Check that rotation is normalized
    quat_length = (
        normalized.rot.x**2 + normalized.rot.y**2 + normalized.rot.z**2 + normalized.rot.w**2
    ) ** 0.5
    assert abs(quat_length - 1.0) < 1e-6


def test_repr():
    t = posrotf(float3(1.0, 2.0, 3.0), quatf.identity())
    repr_str = repr(t)
    assert "posrotf" in repr_str
    assert "pos=" in repr_str
    assert "rot=" in repr_str
