// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/math/quaternion.h"
#include "sgl/math/matrix.h"

using namespace sgl;

TEST_SUITE_BEGIN("quaternion");

template<typename T>
bool almost_equal(T a, T b, T epsilon = T(1e-5))
{
    return std::abs(a - b) < epsilon;
}

template<typename T, int N>
bool almost_equal(math::vector<T, N> a, math::vector<T, N> b, T epsilon = T(1e-5))
{
    return all(lt(abs(a - b), epsilon));
}

template<typename T>
bool almost_equal(math::quat<T> a, math::quat<T> b, T epsilon = T(1e-5))
{
    return almost_equal(math::vector<T, 4>(a.x, a.y, a.z, a.w), math::vector<T, 4>(b.x, b.y, b.z, b.w), epsilon);
}

template<typename T>
bool almost_equal_orientation(math::quat<T> a, math::quat<T> b, T epsilon = T(1e-5))
{
    if (dot(a, b) < T(0))
        b = -b;
    return almost_equal(a, b, epsilon);
}

#define CHECK_ALMOST_EQ(a, b) CHECK_MESSAGE(almost_equal(a, b), fmt::format("{} != {}", a, b))
#define CHECK_ALMOST_EQ_EPS(a, b, eps) CHECK_MESSAGE(almost_equal(a, b, eps), fmt::format("{} != {}", a, b))
#define CHECK_ALMOST_EQ_ORIENTATION(a, b) CHECK_MESSAGE(almost_equal_orientation(a, b), fmt::format("{} != {}", a, b))

TEST_CASE("constructors")
{
    // Default constructor
    {
        quatf q;
        CHECK_EQ(q.x, 0.f);
        CHECK_EQ(q.y, 0.f);
        CHECK_EQ(q.z, 0.f);
        CHECK_EQ(q.w, 1.f);
    }

    // Constructor with 4 floats
    {
        quatf q(1.f, 2.f, 3.f, 4.f);
        CHECK_EQ(q.x, 1.f);
        CHECK_EQ(q.y, 2.f);
        CHECK_EQ(q.z, 3.f);
        CHECK_EQ(q.w, 4.f);
    }

    // Constructor with vector + scalar
    {
        quatf q(float3(1.f, 2.f, 3.f), 4.f);
        CHECK_EQ(q.x, 1.f);
        CHECK_EQ(q.y, 2.f);
        CHECK_EQ(q.z, 3.f);
        CHECK_EQ(q.w, 4.f);
    }

    // Identity
    {
        quatf q = quatf::identity();
        CHECK_EQ(q.x, 0.f);
        CHECK_EQ(q.y, 0.f);
        CHECK_EQ(q.z, 0.f);
        CHECK_EQ(q.w, 1.f);
    }
}

TEST_CASE("access")
{
    {
        quatf q(1.f, 2.f, 3.f, 4.f);
        CHECK_EQ(q[0], 1.f);
        CHECK_EQ(q[1], 2.f);
        CHECK_EQ(q[2], 3.f);
        CHECK_EQ(q[3], 4.f);
    }
}

TEST_CASE("operators")
{
    // Unary + operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2 = +q1;
        CHECK(q2 == quatf(1.f, 2.f, 3.f, 4.f));
    }

    // Unary - operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2 = -q1;
        CHECK(q2 == quatf(-1.f, -2.f, -3.f, -4.f));
    }

    // Binary + operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(2.f, 3.f, 4.f, 5.f);
        quatf q3 = q1 + q2;
        CHECK(q3 == quatf(3.f, 5.f, 7.f, 9.f));
        quatf q4 = q1 + 2.f;
        CHECK(q4 == quatf(3.f, 4.f, 5.f, 6.f));
        quatf q5 = 2.f + q1;
        CHECK(q5 == quatf(3.f, 4.f, 5.f, 6.f));
    }

    // Binary - operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(2.f, 3.f, 4.f, 5.f);
        quatf q3 = q1 - q2;
        CHECK(q3 == quatf(-1.f, -1.f, -1.f, -1.f));
        quatf q4 = q1 - 2.f;
        CHECK(q4 == quatf(-1.f, 0.f, 1.f, 2.f));
        quatf q5 = 2.f - q1;
        CHECK(q5 == quatf(1.f, 0.f, -1.f, -2.f));
    }

    // Binary * operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2 = q1 * 2.f;
        CHECK(q2 == quatf(2.f, 4.f, 6.f, 8.f));
        quatf q3 = 3.f * q1;
        CHECK(q3 == quatf(3.f, 6.f, 9.f, 12.f));
    }

    // Binary / operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2 = q1 / 2.f;
        CHECK(q2 == quatf(0.5f, 1.f, 1.5f, 2.f));
    }

    // Binary == operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(1.f, 2.f, 3.f, 4.f);
        CHECK(q1 == q2);
        quatf q3(1.f, 2.f, 3.f, 5.f);
        CHECK_FALSE(q1 == q3);
    }

    // Binary != operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(1.f, 2.f, 3.f, 4.f);
        CHECK_FALSE(q1 != q2);
        quatf q3(1.f, 2.f, 3.f, 5.f);
        CHECK(q1 != q3);
    }

    // Component-wise == / !=
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(1.f, 2.f, 3.f, 4.f);
        CHECK(eq(q1, q2) == bool4(true, true, true, true));
        quatf q3(1.f, 2.f, 3.f, 5.f);
        CHECK(eq(q1, q3) == bool4(true, true, true, false));
        CHECK(ne(q1, q2) == bool4(false, false, false, false));
        CHECK(ne(q1, q3) == bool4(false, false, false, true));
    }
}

TEST_CASE("multiply")
{
    // Quaternion / quaternion multiplication
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(2.f, 3.f, 4.f, 5.f);
        quatf q3 = mul(q1, q2);
        CHECK(q3 == quatf(12.f, 24.f, 30.f, 0.f));
    }

    // Quaternion / vector multiplication
    {
        quatf q1(2.f, 3.f, 4.f, 5.f);
        float3 v1(2.f, 3.f, 4.f);
        float3 v2 = mul(q1, v1);
        CHECK(v2 == float3(2.f, 3.f, 4.f));
    }
}

TEST_CASE("float_checks")
{
    // isfinite
    {
        quatf q1(0.f, 0.f, 0.f, 0.f);
        CHECK(isfinite(q1) == bool4(true, true, true, true));
        quatf q2(std::numeric_limits<float>::infinity(), 0.f, 0.f, std::numeric_limits<float>::infinity());
        CHECK(isfinite(q2) == bool4(false, true, true, false));
    }
    // isinf
    {
        quatf q1(0.f, 0.f, 0.f, 0.f);
        CHECK(isinf(q1) == bool4(false, false, false, false));
        quatf q2(std::numeric_limits<float>::infinity(), 0.f, 0.f, std::numeric_limits<float>::infinity());
        CHECK(isinf(q2) == bool4(true, false, false, true));
    }
    // isnan
    {
        quatf q1(0.f, 0.f, 0.f, 0.f);
        CHECK(isnan(q1) == bool4(false, false, false, false));
        quatf q2(std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f, std::numeric_limits<float>::quiet_NaN());
        CHECK(isnan(q2) == bool4(true, false, false, true));
    }
}

TEST_CASE("functions")
{
    // dot product
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(2.f, 3.f, 4.f, 5.f);
        float d = dot(q1, q2);
        CHECK_EQ(d, 40.f);
    }

    // cross product
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(2.f, 3.f, 4.f, 5.f);
        quatf q3 = cross(q1, q2);
        CHECK(q3 == quatf(12.f, 24.f, 30.f, 0.f));
    }

    // length
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        float l = length(q1);
        CHECK_EQ(l, sqrtf(30.f));
    }

    // normalize
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2 = normalize(q1);
        CHECK(q2 == q1 * (1.f / sqrtf(30.f)));
    }

    // conjugate
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2 = conjugate(q1);
        CHECK(q2 == quatf(-1.f, -2.f, -3.f, 4.f));
    }

    // inverse
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2 = inverse(q1);
        CHECK(q2 == quatf(-1.f / 30.f, -2.f / 30.f, -3.f / 30.f, 4.f / 30.f));
    }

    // lerp
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(2.f, 3.f, 4.f, 5.f);
        quatf q3 = lerp(q1, q2, 0.5f);
        CHECK(q3 == quatf(1.5f, 2.5f, 3.5f, 4.5f));
    }
}

TEST_CASE("slerp")
{
    auto test = [&](float angle, float t, float expected_angle)
    {
        quatf from = quatf::identity();
        quatf to = math::quat_from_angle_axis(math::radians(angle), float3(0, 1, 0));
        quatf expected = math::quat_from_angle_axis(math::radians(expected_angle), float3(0, 1, 0));

        quatf q1 = slerp(from, to, t);
        CHECK_ALMOST_EQ_ORIENTATION(q1, expected);

        quatf q2 = slerp(to, from, 1.f - t);
        CHECK_ALMOST_EQ_ORIENTATION(q2, expected);
    };

    // Basic
    test(+160, 0.375f, +60);
    test(-160, 0.375f, -60);

    // Shorting
    test(+320, 0.375f, -15); // Mathematically, should be +120
    test(-320, 0.375f, +15); // Mathematically, should be -120

    // Lengthening short way
    test(320, 1.5f, -60); // Mathematically, should be 480 (ie -240)

    // Lengthening
    test(+70, 3, +210);
    test(-70, 3, -210);

    // Edge case that often causes NaNs
    test(0, .5f, 0);

    // This edge case is ill-defined for "intuitive" slerp and can't be tested.
    // test(180, .25f, 45);

    // Conversely, this edge case is well-defined for "intuitive" slerp.
    // For mathematical slerp, the axis is ill-defined and can take many values.
    test(360, .25f, 0);
}

TEST_CASE("euler")
{
    // pitch 90 degrees
    {
        // quatf q = math::quat_from_angle_axis(math::radians(90.f), float3(1.f, 0.f, 0.f));
        quatf q = quatf(std::sqrt(0.5f), 0.f, 0.f, std::sqrt(0.5f));
        CHECK_ALMOST_EQ(pitch(q), math::radians(90.f));
    }

    // pitch -60 degrees
    {
        // quatf q = math::quat_from_angle_axis(math::radians(-60.f), float3(1.f, 0.f, 0.f));
        quatf q = quatf(-0.5f, 0.f, 0.f, std::sqrt(0.75f));
        CHECK_ALMOST_EQ(pitch(q), math::radians(-60.f));
    }

    // yaw 90 degrees
    {
        // quatf q = math::quat_from_angle_axis(math::radians(90.f), float3(0.f, 1.f, 0.f));
        quatf q = quatf(0.f, std::sqrt(0.5f), 0.f, std::sqrt(0.5f));
        CHECK_ALMOST_EQ_EPS(yaw(q), math::radians(90.f), 1e-3f);
    }

    // yaw -60 degrees
    {
        // quatf q = math::quat_from_angle_axis(math::radians(-60.f), float3(0.f, 1.f, 0.f));
        quatf q = quatf(0.f, -0.5f, 0.f, std::sqrt(0.75f));
        CHECK_ALMOST_EQ(yaw(q), math::radians(-60.f));
    }

    // roll 90 degrees
    {
        // quatf q = math::quat_from_angle_axis(math::radians(90.f), float3(0.f, 0.f, 1.f));
        quatf q = quatf(0.f, 0.f, std::sqrt(0.5f), std::sqrt(0.5f));
        CHECK_ALMOST_EQ(roll(q), math::radians(90.f));
    }

    // roll -60 degrees
    {
        // quatf q = math::quat_from_angle_axis(math::radians(-60.f), float3(0.f, 0.f, 1.f));
        quatf q = quatf(0.f, 0.f, -0.5f, std::sqrt(0.75f));
        CHECK_ALMOST_EQ(roll(q), math::radians(-60.f));
    }

    // euler_angles
    {
        // quatf q1 = math::quat_from_angle_axis(math::radians(-60.f), float3(1.f, 0.f, 0.f));
        quatf q1 = quatf(-0.5f, 0.f, 0.f, std::sqrt(0.75f));
        CHECK_ALMOST_EQ(euler_angles(q1), float3(math::radians(-60.f), 0.f, 0.f));

        // quatf q2 = math::quat_from_angle_axis(math::radians(-60.f), float3(0.f, 1.f, 0.f));
        quatf q2 = quatf(0.f, -0.5f, 0.f, std::sqrt(0.75f));
        CHECK_ALMOST_EQ(euler_angles(q2), float3(0.f, math::radians(-60.f), 0.f));

        // quatf q3 = math::quat_from_angle_axis(math::radians(-60.f), float3(0.f, 0.f, 1.f));
        quatf q3 = quatf(0.f, 0.f, -0.5f, std::sqrt(0.75f));
        CHECK_ALMOST_EQ(euler_angles(q3), float3(0.f, 0.f, math::radians(-60.f)));

        // quatf q4 = mul(mul(q3, q2), q1);
        quatf q4 = quatf(-0.591506362f, -0.158493653f, -0.591506362f, 0.524519026f);
        CHECK_ALMOST_EQ(euler_angles(q4), float3(math::radians(-60.f), math::radians(-60.f), math::radians(-60.f)));
    }
}

TEST_CASE("quat_from_angle_axis")
{
    // 90 degrees X-axis
    {
        quatf q1 = math::quat_from_angle_axis(math::radians(90.f), float3(1.f, 0.f, 0.f));
        quatf q2 = quatf(std::sqrt(0.5f), 0.f, 0.f, std::sqrt(0.5f));
        CHECK_ALMOST_EQ(q1, q2);
    }

    // -60 degrees X-axis
    {
        quatf q1 = math::quat_from_angle_axis(math::radians(-60.f), float3(1.f, 0.f, 0.f));
        quatf q2 = quatf(-0.5f, 0.f, 0.f, std::sqrt(0.75f));
        CHECK_ALMOST_EQ(q1, q2);
    }

    // 90 degrees Y-axis
    {
        quatf q1 = math::quat_from_angle_axis(math::radians(90.f), float3(0.f, 1.f, 0.f));
        quatf q2 = quatf(0.f, std::sqrt(0.5f), 0.f, std::sqrt(0.5f));
        CHECK_ALMOST_EQ(q1, q2);
    }

    // -60 degrees Y-axis
    {
        quatf q1 = math::quat_from_angle_axis(math::radians(-60.f), float3(0.f, 1.f, 0.f));
        quatf q2 = quatf(0.f, -0.5f, 0.f, std::sqrt(0.75f));
        CHECK_ALMOST_EQ(q1, q2);
    }

    // 90 degrees Z-axis
    {
        quatf q1 = math::quat_from_angle_axis(math::radians(90.f), float3(0.f, 0.f, 1.f));
        quatf q2 = quatf(0.f, 0.f, std::sqrt(0.5f), std::sqrt(0.5f));
        CHECK_ALMOST_EQ(q1, q2);
    }

    // -60 degrees Z-axis
    {
        quatf q1 = math::quat_from_angle_axis(math::radians(-60.f), float3(0.f, 0.f, 1.f));
        quatf q2 = quatf(0.f, 0.f, -0.5f, std::sqrt(0.75f));
        CHECK_ALMOST_EQ(q1, q2);
    }
}

TEST_CASE("quat_from_rotation_between_vectors")
{
    auto test = [&](float3 v1, float3 axis, float angle)
    {
        quatf q1 = math::quat_from_angle_axis(math::radians(angle), axis);
        float3 v2 = transform_vector(q1, v1);
        quatf q2 = math::quat_from_rotation_between_vectors(v1, v2);
        float3 v3 = transform_vector(q2, v1);
        CHECK_ALMOST_EQ(v2, v3);
    };

    test(float3(0.f, 1.f, 0.f), float3(1.f, 0.f, 0.f), 90);
    test(float3(0.f, 0.f, 1.f), float3(1.f, 0.f, 0.f), -135);

    test(float3(1.f, 0.f, 0.f), float3(0.f, 1.f, 0.f), 90);
    test(float3(0.f, 0.f, 1.f), float3(0.f, 1.f, 0.f), -135);

    test(float3(1.f, 0.f, 0.f), float3(0.f, 0.f, 1.f), 90);
    test(float3(0.f, 1.f, 0.f), float3(0.f, 0.f, 1.f), -135);

    test(float3(1.f, 0.f, 0.f), normalize(float3(1.f, 1.f, 1.f)), 45);
    test(float3(0.f, 1.f, 0.f), normalize(float3(1.f, 1.f, 1.f)), 90);
    test(float3(0.f, 0.f, 1.f), normalize(float3(1.f, 1.f, 1.f)), 135);
}

TEST_CASE("quat_from_euler_angles")
{
    // 90 degrees X-axis
    {
        quatf q1 = math::quat_from_euler_angles(float3(math::radians(90.f), 0.f, 0.f));
        quatf q2 = quatf(std::sqrt(0.5f), 0.f, 0.f, std::sqrt(0.5f));
        CHECK_ALMOST_EQ(q1, q2);
    }

    // -60 degrees X-axis
    {
        quatf q1 = math::quat_from_euler_angles(float3(math::radians(-60.f), 0.f, 0.f));
        quatf q2 = quatf(-0.5f, 0.f, 0.f, std::sqrt(0.75f));
        CHECK_ALMOST_EQ(q1, q2);
    }

    // 90 degrees Y-axis
    {
        quatf q1 = math::quat_from_euler_angles(float3(0.f, math::radians(90.f), 0.f));
        quatf q2 = quatf(0.f, std::sqrt(0.5f), 0.f, std::sqrt(0.5f));
        CHECK_ALMOST_EQ(q1, q2);
    }

    // -60 degrees Y-axis
    {
        quatf q1 = math::quat_from_euler_angles(float3(0.f, math::radians(-60.f), 0.f));
        quatf q2 = quatf(0.f, -0.5f, 0.f, std::sqrt(0.75f));
        CHECK_ALMOST_EQ(q1, q2);
    }

    // 90 degrees Z-axis
    {
        quatf q1 = math::quat_from_euler_angles(float3(0.f, 0.f, math::radians(90.f)));
        quatf q2 = quatf(0.f, 0.f, std::sqrt(0.5f), std::sqrt(0.5f));
        CHECK_ALMOST_EQ(q1, q2);
    }

    // -60 degrees Z-axis
    {
        quatf q1 = math::quat_from_euler_angles(float3(0.f, 0.f, math::radians(-60.f)));
        quatf q2 = quatf(0.f, 0.f, -0.5f, std::sqrt(0.75f));
        CHECK_ALMOST_EQ(q1, q2);
    }
}

TEST_CASE("quat_from_matrix")
{
    auto test = [&](int x, int y, int z)
    {
        float3 euler_angles = radians(float3(x, y, z));
        quatf q1 = math::quat_from_euler_angles(euler_angles);
        float3x3 m = math::matrix_from_quat(q1);
        quatf q2 = math::quat_from_matrix(m);
        CHECK_ALMOST_EQ(q1, q2);
        CHECK_ALMOST_EQ_EPS(euler_angles, math::euler_angles(q2), 1e-3f);
    };

    test(0, 0, 0);
    test(90, 0, 0);
    test(0, 90, 0);
    test(0, 0, 90);
    test(-45, 0, 0);
    test(0, -45, 0);
    test(0, 0, -45);
    test(10, 20, 30);
    test(-30, -20, -10);
}

TEST_SUITE_END();
