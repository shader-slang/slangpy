// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/math/posrot.h"
#include "sgl/math/matrix.h"
#include <doctest/doctest.h>

using namespace sgl;
using namespace sgl::math;

TEST_CASE("posrot - constructors")
{
    SUBCASE("default constructor")
    {
        posrotf pr;
        CHECK(pr.pos.x == 0.0f);
        CHECK(pr.pos.y == 0.0f);
        CHECK(pr.pos.z == 0.0f);
        auto identity = quatf::identity();
        CHECK(pr.rot.w == identity.w);
        CHECK(pr.rot.x == identity.x);
        CHECK(pr.rot.y == identity.y);
        CHECK(pr.rot.z == identity.z);
    }

    SUBCASE("position constructor")
    {
        float3 pos(1.0f, 2.0f, 3.0f);
        posrotf pr(pos);
        CHECK(pr.pos.x == pos.x);
        CHECK(pr.pos.y == pos.y);
        CHECK(pr.pos.z == pos.z);
        auto identity = quatf::identity();
        CHECK(pr.rot.w == identity.w);
        CHECK(pr.rot.x == identity.x);
        CHECK(pr.rot.y == identity.y);
        CHECK(pr.rot.z == identity.z);
    }

    SUBCASE("rotation constructor")
    {
        quatf rot(0.0f, 0.0f, 0.707107f, 0.707107f); // 90 degrees around Z
        posrotf pr(rot);
        CHECK(pr.pos == float3(0.0f));
        CHECK(pr.rot == rot);
    }

    SUBCASE("position and rotation constructor")
    {
        float3 pos(1.0f, 2.0f, 3.0f);
        quatf rot(0.0f, 0.0f, 0.707107f, 0.707107f);
        posrotf pr(pos, rot);
        CHECK(pr.pos == pos);
        CHECK(pr.rot == rot);
    }

    SUBCASE("array constructor")
    {
        std::array<float, 7> a = {1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.707107f, 0.707107f};
        posrotf pr(a);
        CHECK(pr.pos == float3(1.0f, 2.0f, 3.0f));
        CHECK(pr.rot.x == doctest::Approx(0.0f));
        CHECK(pr.rot.y == doctest::Approx(0.0f));
        CHECK(pr.rot.z == doctest::Approx(0.707107f));
        CHECK(pr.rot.w == doctest::Approx(0.707107f));
    }
}

TEST_CASE("posrot - static constructors")
{
    SUBCASE("identity")
    {
        posrotf pr = posrotf::identity();
        CHECK(pr.pos == float3(0.0f));
        CHECK(pr.rot == quatf::identity());
    }

    SUBCASE("translation")
    {
        float3 pos(1.0f, 2.0f, 3.0f);
        posrotf pr = posrot_from_translation(pos);
        CHECK(pr.pos == pos);
        CHECK(pr.rot == quatf::identity());
    }

    SUBCASE("rotation")
    {
        quatf rot(0.0f, 0.0f, 0.707107f, 0.707107f);
        posrotf pr = posrot_from_rotation(rot);
        CHECK(pr.pos == float3(0.0f));
        CHECK(pr.rot == rot);
    }
}

TEST_CASE("posrot - comparison operators")
{
    posrotf pr1(float3(1.0f, 2.0f, 3.0f), quatf::identity());
    posrotf pr2(float3(1.0f, 2.0f, 3.0f), quatf::identity());
    posrotf pr3(float3(2.0f, 2.0f, 3.0f), quatf::identity());

    CHECK(pr1 == pr2);
    CHECK(pr1 != pr3);
}

TEST_CASE("posrot - transform operations")
{
    SUBCASE("multiply transforms")
    {
        posrotf t1 = posrot_from_translation(float3(1.0f, 0.0f, 0.0f));
        posrotf t2 = posrot_from_translation(float3(2.0f, 0.0f, 0.0f));
        posrotf result = t1 * t2;
        CHECK(result.pos.x == doctest::Approx(3.0f));
        CHECK(result.pos.y == doctest::Approx(0.0f));
        CHECK(result.pos.z == doctest::Approx(0.0f));
    }

    SUBCASE("transform point")
    {
        posrotf t = posrot_from_translation(float3(1.0f, 2.0f, 3.0f));
        float3 point(1.0f, 1.0f, 1.0f);
        float3 result = transform_point(t, point);
        CHECK(result == float3(2.0f, 3.0f, 4.0f));
    }

    SUBCASE("inverse")
    {
        posrotf t(float3(1.0f, 2.0f, 3.0f), quatf::identity());
        posrotf inv = inverse(t);
        posrotf identity_check = t * inv;

        CHECK(identity_check.pos.x == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(identity_check.pos.y == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(identity_check.pos.z == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(identity_check.rot.w == doctest::Approx(1.0f).epsilon(1e-6f));
    }
}

TEST_CASE("posrot - matrix conversion")
{
    SUBCASE("to_matrix3x4")
    {
        posrotf t(float3(1.0f, 2.0f, 3.0f), quatf::identity());
        auto m = matrix3x4_from_posrot(t);
        CHECK(m[0][3] == 1.0f);
        CHECK(m[1][3] == 2.0f);
        CHECK(m[2][3] == 3.0f);

        // Identity rotation should give identity 3x3 part
        CHECK(m[0][0] == doctest::Approx(1.0f));
        CHECK(m[1][1] == doctest::Approx(1.0f));
        CHECK(m[2][2] == doctest::Approx(1.0f));
    }

    SUBCASE("to_matrix4x4")
    {
        posrotf t(float3(1.0f, 2.0f, 3.0f), quatf::identity());
        auto m = matrix4x4_from_posrot(t);
        CHECK(m[0][3] == 1.0f);
        CHECK(m[1][3] == 2.0f);
        CHECK(m[2][3] == 3.0f);
        CHECK(m[3][3] == 1.0f);

        // Identity rotation should give identity 3x3 part
        CHECK(m[0][0] == doctest::Approx(1.0f));
        CHECK(m[1][1] == doctest::Approx(1.0f));
        CHECK(m[2][2] == doctest::Approx(1.0f));
    }

    SUBCASE("roundtrip conversion")
    {
        posrotf original(float3(1.0f, 2.0f, 3.0f), quatf::identity());
        auto m4 = matrix4x4_from_posrot(original);
        posrotf recovered = posrot_from_matrix4x4(m4);

        CHECK(recovered.pos.x == doctest::Approx(original.pos.x));
        CHECK(recovered.pos.y == doctest::Approx(original.pos.y));
        CHECK(recovered.pos.z == doctest::Approx(original.pos.z));
        CHECK(recovered.rot.w == doctest::Approx(original.rot.w));
        CHECK(recovered.rot.x == doctest::Approx(original.rot.x));
        CHECK(recovered.rot.y == doctest::Approx(original.rot.y));
        CHECK(recovered.rot.z == doctest::Approx(original.rot.z));
    }
}

TEST_CASE("posrot - interpolation")
{
    SUBCASE("lerp")
    {
        posrotf a(float3(0.0f, 0.0f, 0.0f), quatf::identity());
        posrotf b(float3(2.0f, 2.0f, 2.0f), quatf::identity());

        posrotf mid = lerp(a, b, 0.5f);
        CHECK(mid.pos.x == doctest::Approx(1.0f));
        CHECK(mid.pos.y == doctest::Approx(1.0f));
        CHECK(mid.pos.z == doctest::Approx(1.0f));
    }

    SUBCASE("slerp")
    {
        posrotf a(float3(0.0f, 0.0f, 0.0f), quatf::identity());
        posrotf b(float3(2.0f, 2.0f, 2.0f), quatf::identity());

        posrotf mid = slerp(a, b, 0.5f);
        CHECK(mid.pos.x == doctest::Approx(1.0f));
        CHECK(mid.pos.y == doctest::Approx(1.0f));
        CHECK(mid.pos.z == doctest::Approx(1.0f));
    }
}

TEST_CASE("posrot - normalize")
{
    // Create a posrot with a non-normalized quaternion
    posrotf t(float3(1.0f, 2.0f, 3.0f), quatf(0.0f, 0.0f, 1.0f, 1.0f));
    posrotf normalized = normalize(t);

    // Check that position is unchanged
    CHECK(normalized.pos == t.pos);

    // Check that rotation is normalized
    float quat_length = sqrt(
        normalized.rot.x * normalized.rot.x + normalized.rot.y * normalized.rot.y + normalized.rot.z * normalized.rot.z
        + normalized.rot.w * normalized.rot.w
    );
    CHECK(quat_length == doctest::Approx(1.0f));
}
