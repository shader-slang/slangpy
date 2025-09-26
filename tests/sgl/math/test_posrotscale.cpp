// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/math/posrotscale.h"
#include "sgl/math/posrot.h"
#include "sgl/math/matrix.h"
#include <doctest/doctest.h>

using namespace sgl;
using namespace sgl::math;

TEST_CASE("posrotscale - constructors")
{
    SUBCASE("default constructor")
    {
        posrotscalef prs;
        CHECK(prs.pos == float3(0.0f));
        CHECK(prs.rot == quatf::identity());
        CHECK(prs.scale == float3(1.0f));
    }

    SUBCASE("position constructor")
    {
        float3 pos(1.0f, 2.0f, 3.0f);
        posrotscalef prs(pos);
        CHECK(prs.pos == pos);
        CHECK(prs.rot == quatf::identity());
        CHECK(prs.scale == float3(1.0f));
    }

    SUBCASE("rotation constructor")
    {
        quatf rot(0.0f, 0.0f, 0.707107f, 0.707107f);
        posrotscalef prs(rot);
        CHECK(prs.pos == float3(0.0f));
        CHECK(prs.rot == rot);
        CHECK(prs.scale == float3(1.0f));
    }

    SUBCASE("scale constructor")
    {
        float3 scale(2.0f, 3.0f, 4.0f);
        posrotscalef prs(scale);
        CHECK(prs.pos == float3(0.0f));
        CHECK(prs.rot == quatf::identity());
        CHECK(prs.scale == scale);
    }

    SUBCASE("position and rotation constructor")
    {
        float3 pos(1.0f, 2.0f, 3.0f);
        quatf rot(0.0f, 0.0f, 0.707107f, 0.707107f);
        posrotscalef prs(pos, rot);
        CHECK(prs.pos == pos);
        CHECK(prs.rot == rot);
        CHECK(prs.scale == float3(1.0f));
    }

    SUBCASE("full constructor")
    {
        float3 pos(1.0f, 2.0f, 3.0f);
        quatf rot(0.0f, 0.0f, 0.707107f, 0.707107f);
        float3 scale(2.0f, 3.0f, 4.0f);
        posrotscalef prs(pos, rot, scale);
        CHECK(prs.pos == pos);
        CHECK(prs.rot == rot);
        CHECK(prs.scale == scale);
    }

    SUBCASE("posrot constructor")
    {
        posrotf pr(float3(1.0f, 2.0f, 3.0f), quatf::identity());
        posrotscalef prs(pr);
        CHECK(prs.pos == pr.pos);
        CHECK(prs.rot == pr.rot);
        CHECK(prs.scale == float3(1.0f));
    }

    SUBCASE("array constructor")
    {
        std::array<float, 10> a = {1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.707107f, 0.707107f, 2.0f, 3.0f, 4.0f};
        posrotscalef prs(a);
        CHECK(prs.pos == float3(1.0f, 2.0f, 3.0f));
        CHECK(prs.rot.x == doctest::Approx(0.0f));
        CHECK(prs.rot.y == doctest::Approx(0.0f));
        CHECK(prs.rot.z == doctest::Approx(0.707107f));
        CHECK(prs.rot.w == doctest::Approx(0.707107f));
        CHECK(prs.scale == float3(2.0f, 3.0f, 4.0f));
    }
}

TEST_CASE("posrotscale - static constructors")
{
    SUBCASE("identity")
    {
        posrotscalef prs = posrotscalef::identity();
        CHECK(prs.pos == float3(0.0f));
        CHECK(prs.rot == quatf::identity());
        CHECK(prs.scale == float3(1.0f));
    }

    SUBCASE("translation")
    {
        float3 pos(1.0f, 2.0f, 3.0f);
        posrotscalef prs = posrotscale_from_translation(pos);
        CHECK(prs.pos == pos);
        CHECK(prs.rot == quatf::identity());
        CHECK(prs.scale == float3(1.0f));
    }

    SUBCASE("rotation")
    {
        quatf rot(0.0f, 0.0f, 0.707107f, 0.707107f);
        posrotscalef prs = posrotscale_from_rotation(rot);
        CHECK(prs.pos == float3(0.0f));
        CHECK(prs.rot == rot);
        CHECK(prs.scale == float3(1.0f));
    }

    SUBCASE("scaling")
    {
        float3 scale(2.0f, 3.0f, 4.0f);
        posrotscalef prs = posrotscale_from_scaling(scale);
        CHECK(prs.pos == float3(0.0f));
        CHECK(prs.rot == quatf::identity());
        CHECK(prs.scale == scale);
    }

    SUBCASE("uniform_scaling")
    {
        posrotscalef prs = posrotscale_from_uniform_scaling(2.5f);
        CHECK(prs.pos == float3(0.0f));
        CHECK(prs.rot == quatf::identity());
        CHECK(prs.scale == float3(2.5f));
    }
}

TEST_CASE("posrotscale - comparison operators")
{
    posrotscalef prs1(float3(1.0f, 2.0f, 3.0f), quatf::identity(), float3(2.0f));
    posrotscalef prs2(float3(1.0f, 2.0f, 3.0f), quatf::identity(), float3(2.0f));
    posrotscalef prs3(float3(2.0f, 2.0f, 3.0f), quatf::identity(), float3(2.0f));

    CHECK(prs1 == prs2);
    CHECK(prs1 != prs3);
}

TEST_CASE("posrotscale - transform operations")
{
    SUBCASE("multiply transforms")
    {
        posrotscalef t1 = posrotscale_from_translation(float3(1.0f, 0.0f, 0.0f));
        posrotscalef t2 = posrotscale_from_scaling(float3(2.0f, 1.0f, 1.0f));
        posrotscalef result = t1 * t2;

        // The scaling should be applied first, then translation
        // So a point at (1,0,0) becomes (2,0,0) after scaling, then (3,0,0) after translation
        float3 test_point(1.0f, 0.0f, 0.0f);
        float3 transformed = transform_point(result, test_point);
        CHECK(transformed.x == doctest::Approx(3.0f));
    }

    SUBCASE("transform point")
    {
        posrotscalef t(float3(1.0f, 2.0f, 3.0f), quatf::identity(), float3(2.0f, 1.0f, 0.5f));
        float3 point(1.0f, 1.0f, 2.0f);
        float3 result = transform_point(t, point);

        // Scale: (1,1,2) -> (2,1,1)
        // Rotate: identity, so (2,1,1)
        // Translate: (2,1,1) + (1,2,3) = (3,3,4)
        CHECK(result == float3(3.0f, 3.0f, 4.0f));
    }

    SUBCASE("inverse")
    {
        posrotscalef t(float3(1.0f, 2.0f, 3.0f), quatf::identity(), float3(2.0f, 0.5f, 4.0f));
        posrotscalef inv = inverse(t);
        posrotscalef identity_check = t * inv;

        // Should be close to identity
        CHECK(identity_check.pos.x == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(identity_check.pos.y == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(identity_check.pos.z == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(identity_check.scale.x == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(identity_check.scale.y == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(identity_check.scale.z == doctest::Approx(1.0f).epsilon(1e-5f));
    }
}

TEST_CASE("posrotscale - matrix conversion")
{
    SUBCASE("to_matrix3x4")
    {
        posrotscalef t(float3(1.0f, 2.0f, 3.0f), quatf::identity(), float3(2.0f, 3.0f, 4.0f));
        auto m = matrix_from_posrotscale_3x4(t);

        CHECK(m[0][3] == 1.0f); // translation x
        CHECK(m[1][3] == 2.0f); // translation y
        CHECK(m[2][3] == 3.0f); // translation z

        // Identity rotation with scale should give scaled identity 3x3 part
        CHECK(m[0][0] == doctest::Approx(2.0f));
        CHECK(m[1][1] == doctest::Approx(3.0f));
        CHECK(m[2][2] == doctest::Approx(4.0f));
    }

    SUBCASE("to_matrix4x4")
    {
        posrotscalef t(float3(1.0f, 2.0f, 3.0f), quatf::identity(), float3(2.0f, 3.0f, 4.0f));
        auto m = matrix_from_posrotscale(t);

        CHECK(m[0][3] == 1.0f);
        CHECK(m[1][3] == 2.0f);
        CHECK(m[2][3] == 3.0f);
        CHECK(m[3][3] == 1.0f);

        // Identity rotation with scale
        CHECK(m[0][0] == doctest::Approx(2.0f));
        CHECK(m[1][1] == doctest::Approx(3.0f));
        CHECK(m[2][2] == doctest::Approx(4.0f));
    }

    SUBCASE("roundtrip conversion")
    {
        posrotscalef original(float3(1.0f, 2.0f, 3.0f), quatf::identity(), float3(2.0f, 0.5f, 3.0f));
        auto m4 = matrix_from_posrotscale(original);
        posrotscalef recovered = posrotscale_from_matrix4x4(m4);

        CHECK(recovered.pos.x == doctest::Approx(original.pos.x));
        CHECK(recovered.pos.y == doctest::Approx(original.pos.y));
        CHECK(recovered.pos.z == doctest::Approx(original.pos.z));
        CHECK(recovered.scale.x == doctest::Approx(original.scale.x));
        CHECK(recovered.scale.y == doctest::Approx(original.scale.y));
        CHECK(recovered.scale.z == doctest::Approx(original.scale.z));
    }
}

TEST_CASE("posrotscale - interpolation")
{
    SUBCASE("lerp")
    {
        posrotscalef a(float3(0.0f, 0.0f, 0.0f), quatf::identity(), float3(1.0f));
        posrotscalef b(float3(2.0f, 2.0f, 2.0f), quatf::identity(), float3(3.0f));

        posrotscalef mid = lerp(a, b, 0.5f);
        CHECK(mid.pos.x == doctest::Approx(1.0f));
        CHECK(mid.pos.y == doctest::Approx(1.0f));
        CHECK(mid.pos.z == doctest::Approx(1.0f));
        CHECK(mid.scale.x == doctest::Approx(2.0f));
        CHECK(mid.scale.y == doctest::Approx(2.0f));
        CHECK(mid.scale.z == doctest::Approx(2.0f));
    }
}

TEST_CASE("posrotscale - conversions")
{
    SUBCASE("to_posrot")
    {
        posrotscalef prs(float3(1.0f, 2.0f, 3.0f), quatf::identity(), float3(2.0f, 3.0f, 4.0f));
        posrotf pr = posrot_from_posrotscale(prs);

        CHECK(pr.pos == prs.pos);
        CHECK(pr.rot == prs.rot);
    }
}

TEST_CASE("posrotscale - mixed operations")
{
    SUBCASE("multiply with posrot")
    {
        posrotscalef prs = posrotscale_from_scaling(float3(2.0f));
        posrotf pr = posrot_from_translation(float3(1.0f, 0.0f, 0.0f));

        posrotscalef result = prs * pr;
        CHECK(result.scale.x == doctest::Approx(2.0f));
        CHECK(result.pos.x == doctest::Approx(2.0f)); // scaled translation
    }
}
