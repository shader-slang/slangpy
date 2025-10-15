// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/math/matrix.h"

using namespace sgl;

TEST_SUITE_BEGIN("matrix");

template<typename T, int N>
bool almost_equal(math::vector<T, N> a, math::vector<T, N> b, T epsilon = T(1e-5))
{
    return all(abs(a - b) < math::vector<T, N>(epsilon));
}

template<typename T>
bool almost_equal(math::quat<T> a, math::quat<T> b, T epsilon = T(1e-5))
{
    return all(math::vector<T, 4>(b.x - a.x, b.y - a.y, b.z - a.z, b.w - a.w) < math::vector<T, 4>(epsilon));
}

#define CHECK_ALMOST_EQ(a, b) CHECK_MESSAGE(almost_equal(a, b), fmt::format("{} != {}", a, b))

#define CHECK_EQ_VECTOR(a, b) CHECK_MESSAGE(all(a == b), fmt::format("{} != {}", a, b))

// Helper for constructing matrices row-by-row without clang-format screwing up the formatting
#define ROW(...) __VA_ARGS__

TEST_CASE("constructors")
{
    // Default constructor
    {
        float4x4 m;
        CHECK_EQ_VECTOR(m[0], float4(1, 0, 0, 0));
        CHECK_EQ_VECTOR(m[1], float4(0, 1, 0, 0));
        CHECK_EQ_VECTOR(m[2], float4(0, 0, 1, 0));
        CHECK_EQ_VECTOR(m[3], float4(0, 0, 0, 1));
    }

    // Initializer list constructor
    {
        float4x4 m({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        CHECK_EQ_VECTOR(m[0], float4(1, 2, 3, 4));
        CHECK_EQ_VECTOR(m[1], float4(5, 6, 7, 8));
        CHECK_EQ_VECTOR(m[2], float4(9, 10, 11, 12));
        CHECK_EQ_VECTOR(m[3], float4(13, 14, 15, 16));
    }

    // Identity
    {
        float4x4 m = float4x4::identity();
        CHECK_EQ_VECTOR(m[0], float4(1, 0, 0, 0));
        CHECK_EQ_VECTOR(m[1], float4(0, 1, 0, 0));
        CHECK_EQ_VECTOR(m[2], float4(0, 0, 1, 0));
        CHECK_EQ_VECTOR(m[3], float4(0, 0, 0, 1));
    }

    // Zeros
    {
        float4x4 m = float4x4::zeros();
        CHECK_EQ_VECTOR(m[0], float4(0, 0, 0, 0));
        CHECK_EQ_VECTOR(m[1], float4(0, 0, 0, 0));
        CHECK_EQ_VECTOR(m[2], float4(0, 0, 0, 0));
        CHECK_EQ_VECTOR(m[3], float4(0, 0, 0, 0));
    }
}

TEST_CASE("multilply")
{
    // Scalar multiplication
    {
        float4x4 m({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        float4x4 m2 = m * 2.f;
        CHECK_EQ_VECTOR(m2[0], float4(2, 4, 6, 8));
        CHECK_EQ_VECTOR(m2[1], float4(10, 12, 14, 16));
        CHECK_EQ_VECTOR(m2[2], float4(18, 20, 22, 24));
        CHECK_EQ_VECTOR(m2[3], float4(26, 28, 30, 32));
    }

    // Matrix/matrix multiplication
    {
        float4x4 m1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        float4x4 m2({-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16});
        float4x4 m3 = mul(m1, m2);
        CHECK_EQ_VECTOR(m3[0], float4(-90, -100, -110, -120));
        CHECK_EQ_VECTOR(m3[1], float4(-202, -228, -254, -280));
        CHECK_EQ_VECTOR(m3[2], float4(-314, -356, -398, -440));
        CHECK_EQ_VECTOR(m3[3], float4(-426, -484, -542, -600));
    }

    // Matrix/vector multiplication
    {
        float4x4 m({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        float4 v(1, 2, 3, 4);
        float4 v2 = mul(m, v);
        CHECK_EQ_VECTOR(v2, float4(30, 70, 110, 150));
    }

    // Vector/matrix multiplication
    {
        float4x4 m({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        float4 v(1, 2, 3, 4);
        float4 v2 = mul(v, m);
        CHECK_EQ_VECTOR(v2, float4(90, 100, 110, 120));
    }
}

TEST_CASE("transform_point")
{
    float4x4 m({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    float3 v(1, 2, 3);
    float3 v2 = transform_point(m, v);
    CHECK_EQ_VECTOR(v2, float3(18, 46, 74));
}

TEST_CASE("transform_vector")
{
    // 3x3
    {
        float3x3 m({1, 2, 3, 4, 5, 6, 7, 8, 9});
        float3 v(1, 2, 3);
        float3 v2 = transform_vector(m, v);
        CHECK_EQ_VECTOR(v2, float3(14, 32, 50));
    }

    // 4x4
    {
        float4x4 m({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        float3 v(1, 2, 3);
        float3 v2 = transform_vector(m, v);
        CHECK_EQ_VECTOR(v2, float3(14, 38, 62));
    }
}

TEST_CASE("transpose")
{
    // 3x3
    {
        float3x3 m({1, 2, 3, 4, 5, 6, 7, 8, 9});
        float3x3 m2 = transpose(m);
        CHECK_EQ_VECTOR(m2[0], float3(1, 4, 7));
        CHECK_EQ_VECTOR(m2[1], float3(2, 5, 8));
        CHECK_EQ_VECTOR(m2[2], float3(3, 6, 9));
    }

    // 4x4
    {
        float4x4 m({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        float4x4 m2 = transpose(m);
        CHECK_EQ_VECTOR(m2[0], float4(1, 5, 9, 13));
        CHECK_EQ_VECTOR(m2[1], float4(2, 6, 10, 14));
        CHECK_EQ_VECTOR(m2[2], float4(3, 7, 11, 15));
        CHECK_EQ_VECTOR(m2[3], float4(4, 8, 12, 16));
    }
}

TEST_CASE("translate")
{
    float4x4 m({
        ROW(1, 0, 0, 10),
        ROW(0, -1, 0, 20),
        ROW(0, 0, 1, 30),
        ROW(0, 0, 0, 1),
    });
    float4x4 m2 = translate(m, float3(1, 2, 3));
    CHECK_ALMOST_EQ(m2[0], float4(1, 0, 0, 11));
    CHECK_ALMOST_EQ(m2[1], float4(0, -1, 0, 18));
    CHECK_ALMOST_EQ(m2[2], float4(0, 0, 1, 33));
    CHECK_ALMOST_EQ(m2[3], float4(0, 0, 0, 1));
}

TEST_CASE("rotate")
{
    float4x4 m({
        ROW(1, 0, 0, 10),
        ROW(0, -1, 0, 20),
        ROW(0, 0, 1, 30),
        ROW(0, 0, 0, 1),
    });
    float4x4 m2 = rotate(m, math::radians(90.f), float3(0, 1, 0));
    CHECK_ALMOST_EQ(m2[0], float4(0, 0, 1, 10));
    CHECK_ALMOST_EQ(m2[1], float4(0, -1, 0, 20));
    CHECK_ALMOST_EQ(m2[2], float4(-1, 0, 0, 30));
    CHECK_ALMOST_EQ(m2[3], float4(0, 0, 0, 1));
}

TEST_CASE("scale")
{
    float4x4 m({
        ROW(1, 0, 0, 10),
        ROW(0, -1, 0, 20),
        ROW(0, 0, 1, 30),
        ROW(0, 0, 0, 1),
    });
    float4x4 m2 = scale(m, float3(2, 3, 4));
    CHECK_ALMOST_EQ(m2[0], float4(2, 0, 0, 10));
    CHECK_ALMOST_EQ(m2[1], float4(0, -3, 0, 20));
    CHECK_ALMOST_EQ(m2[2], float4(0, 0, 4, 30));
    CHECK_ALMOST_EQ(m2[3], float4(0, 0, 0, 1));
}

TEST_CASE("determinant")
{
    // 2x2
    {
        float2x2 m1 = float2x2({1, 2, 1, 2});
        CHECK_EQ(math::determinant(m1), 0);
        float2x2 m2 = float2x2({1, 2, 3, 4});
        CHECK_EQ(math::determinant(m2), -2);
    }

    // 3x3
    {
        float3x3 m1 = float3x3({1, 2, 3, 4, 5, 6, 7, 8, 9});
        CHECK_EQ(math::determinant(m1), 0);
        float3x3 m2 = float3x3({1, 2, 3, 6, 5, 4, 8, 7, 9});
        CHECK_EQ(math::determinant(m2), -21);
    }

    // 4x4
    {
        float4x4 m1 = float4x4({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        CHECK_EQ(math::determinant(m1), 0);
        float4x4 m2 = float4x4({1, 2, 3, 4, 8, 7, 6, 5, 9, 10, 12, 11, 15, 16, 13, 14});
        CHECK_EQ(math::determinant(m2), 72);
    }
}

TEST_CASE("inverse")
{
    // 2x2
    {
        float2x2 m = inverse(float2x2({1, 2, 3, 4}));
        CHECK_ALMOST_EQ(m[0], float2(-2, 1));
        CHECK_ALMOST_EQ(m[1], float2(1.5f, -0.5f));
    }

    // 3x3
    {
        float3x3 m = inverse(float3x3({1, 2, 3, 6, 5, 4, 8, 7, 9}));
        CHECK_ALMOST_EQ(m[0], float3(-0.809523f, -0.142857f, 0.333333f));
        CHECK_ALMOST_EQ(m[1], float3(1.047619f, 0.714285f, -0.666666f));
        CHECK_ALMOST_EQ(m[2], float3(-0.095238f, -0.428571f, 0.333333f));
    }

    // 4x4
    {
        float4x4 m = inverse(float4x4({1, 2, 3, 4, 8, 7, 6, 5, 9, 10, 12, 11, 15, 16, 13, 14}));
        CHECK_ALMOST_EQ(m[0], float4(1.125f, 1.25f, -0.5f, -0.375f));
        CHECK_ALMOST_EQ(m[1], float4(-1.652777f, -1.527777f, 0.5f, 0.625f));
        CHECK_ALMOST_EQ(m[2], float4(-0.625f, -0.25f, 0.5f, -0.125f));
        CHECK_ALMOST_EQ(m[3], float4(1.263888f, 0.638888f, -0.5f, -0.125f));
    }
}

TEST_CASE("extract_euler_angle_xyz")
{
    {
        // float4x4 m = math::matrix_from_rotation_xyz(math::radians(45.f), math::radians(45.f), math::radians(45.f));
        float4x4 m = float4x4({
            ROW(0.5f, -0.5f, 0.707107f, 0.f),
            ROW(0.853553f, 0.146446f, -0.5f, 0.f),
            ROW(0.146446f, 0.853553f, 0.5f, 0.f),
            ROW(0.f, 0.f, 0.f, 1.f),
        });
        float3 angles;
        math::extract_euler_angle_xyz(m, angles.x, angles.y, angles.z);
        CHECK_ALMOST_EQ(angles, math::radians(float3(45.f, 45.f, 45.f)));
    }

    {
        // float4x4 m = math::matrix_from_rotation_xyz(math::radians(20.f), math::radians(40.f), math::radians(60.f));
        float4x4 m = float4x4({
            ROW(0.383022f, -0.663414f, 0.642787f, 0.f),
            ROW(0.923720f, 0.279453f, -0.262002f, 0.f),
            ROW(-0.005813f, 0.694109f, 0.719846f, 0.f),
            ROW(0.f, 0.f, 0.f, 1.f),
        });
        float3 angles;
        math::extract_euler_angle_xyz(m, angles.x, angles.y, angles.z);
        CHECK_ALMOST_EQ(angles, math::radians(float3(20.f, 40.f, 60.f)));
    }
}

TEST_CASE("decompose")
{
    const auto test_decompose = [&](float4x4 m,
                                    float3 expected_scale,
                                    quatf expected_orientation,
                                    float3 expected_translation,
                                    float3 expected_skew,
                                    float4 expected_perspective,
                                    bool expected_result = true)
    {
        float3 scale;
        quatf orientation;
        float3 translation;
        float3 skew;
        float4 perspective;
        bool result = math::decompose(m, scale, orientation, translation, skew, perspective);
        if (expected_result) {
            CHECK_ALMOST_EQ(scale, expected_scale);
            CHECK_ALMOST_EQ(orientation, expected_orientation);
            CHECK_ALMOST_EQ(translation, expected_translation);
            CHECK_ALMOST_EQ(skew, expected_skew);
            CHECK_ALMOST_EQ(perspective, expected_perspective);
        }
        CHECK_EQ(result, expected_result);
    };

    // Zero matrix
    test_decompose(
        float4x4::zeros(), // matrix
        float3(),          // scale
        quatf(),           // orientation
        float3(),          // translation
        float3(),          // skew
        float4(),          // perspective
        false              // result
    );

    // Identity matrix
    test_decompose(
        float4x4::identity(),      // matrix
        float3(1.f, 1.f, 1.f),     // scale
        quatf::identity(),         // orientation
        float3(0.f, 0.f, 0.f),     // translation
        float3(0.f, 0.f, 0.f),     // skew
        float4(0.f, 0.f, 0.f, 1.f) // perspective
    );

    // Scale only
    test_decompose(
        float4x4({
            ROW(2.f, 0.f, 0.f, 0.f),
            ROW(0.f, 3.f, 0.f, 0.f),
            ROW(0.f, 0.f, 4.f, 0.f),
            ROW(0.f, 0.f, 0.f, 1.f),
        }),
        float3(2.f, 3.f, 4.f),     // scale
        quatf::identity(),         // orientation
        float3(0.f, 0.f, 0.f),     // translation
        float3(0.f, 0.f, 0.f),     // skew
        float4(0.f, 0.f, 0.f, 1.f) // perspective
    );

    // Orientation only
    // float4x4 m = math::matrix_from_rotation_x(math::radians(45.f));
    test_decompose(
        float4x4({
            ROW(1.f, 0.f, 0.f, 0.f),
            ROW(0.f, 0.707107f, -0.707107f, 0.f),
            ROW(0.f, 0.707107f, 0.707107f, 0.f),
            ROW(0.f, 0.f, 0.f, 1.f),
        }),
        float3(1.f, 1.f, 1.f),                // scale
        quatf(0.382683f, 0.f, 0.f, 0.92388f), // orientation
        float3(0.f, 0.f, 0.f),                // translation
        float3(0.f, 0.f, 0.f),                // skew
        float4(0.f, 0.f, 0.f, 1.f)            // perspective
    );

    // Translation only
    test_decompose(
        float4x4({
            ROW(1.f, 0.f, 0.f, 1.f),
            ROW(0.f, 1.f, 0.f, 2.f),
            ROW(0.f, 0.f, 1.f, 3.f),
            ROW(0.f, 0.f, 0.f, 1.f),
        }),
        float3(1.f, 1.f, 1.f),     // scale
        quatf::identity(),         // orientation
        float3(1.f, 2.f, 3.f),     // translation
        float3(0.f, 0.f, 0.f),     // skew
        float4(0.f, 0.f, 0.f, 1.f) // perspective
    );

    // Skew only
    test_decompose(
        float4x4({
            ROW(1.f, 2.f, 3.f, 0.f),
            ROW(0.f, 1.f, 4.f, 0.f),
            ROW(0.f, 0.f, 1.f, 0.f),
            ROW(0.f, 0.f, 0.f, 1.f),
        }),
        float3(1.f, 1.f, 1.f),     // scale
        quatf::identity(),         // orientation
        float3(0.f, 0.f, 0.f),     // translation
        float3(4.f, 3.f, 2.f),     // skew
        float4(0.f, 0.f, 0.f, 1.f) // perspective
    );

    // Perspective only
    test_decompose(
        float4x4({
            ROW(1.f, 0.f, 0.f, 0.f),
            ROW(0.f, 1.f, 0.f, 0.f),
            ROW(0.f, 0.f, 1.f, 0.f),
            ROW(0.1f, 0.2f, 0.3f, 1.f),
        }),
        float3(1.f, 1.f, 1.f),        // scale
        quatf::identity(),            // orientation
        float3(0.f, 0.f, 0.f),        // translation
        float3(0.f, 0.f, 0.f),        // skew
        float4(0.1f, 0.2f, 0.3f, 1.f) // perspective
    );

    // Affine transform
    float4x4 m = float4x4::identity();
    m = mul(math::matrix_from_scaling(float3(2.f, 3.f, 4.f)), m);
    m = mul(math::matrix_from_rotation_x(math::radians(45.f)), m);
    m = mul(math::matrix_from_translation(float3(1.f, 2.f, 3.f)), m);
    test_decompose(
        float4x4({
            ROW(2.f, 0.f, 0.f, 1.f),
            ROW(0.f, 2.12132f, -2.82843f, 2.f),
            ROW(0.f, 2.12132f, 2.82843f, 3.f),
            ROW(0.f, 0.f, 0.f, 1.f),
        }),
        float3(2.f, 3.f, 4.f),                // scale
        quatf(0.382683f, 0.f, 0.f, 0.92388f), // orientation
        float3(1.f, 2.f, 3.f),                // translation
        float3(0.f, 0.f, 0.f),                // skew
        float4(0.f, 0.f, 0.f, 1.f)            // perspective
    );
}

TEST_CASE("matrix_from_coefficients")
{
    // 3x3
    {
        const float values[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        float3x3 m = math::matrix_from_coefficients<float, 3, 3>(values);
        CHECK_EQ_VECTOR(m[0], float3(1, 2, 3));
        CHECK_EQ_VECTOR(m[1], float3(4, 5, 6));
        CHECK_EQ_VECTOR(m[2], float3(7, 8, 9));
    }

    // 4x4
    {
        const float values[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        float4x4 m = math::matrix_from_coefficients<float, 4, 4>(values);
        CHECK_EQ_VECTOR(m[0], float4(1, 2, 3, 4));
        CHECK_EQ_VECTOR(m[1], float4(5, 6, 7, 8));
        CHECK_EQ_VECTOR(m[2], float4(9, 10, 11, 12));
        CHECK_EQ_VECTOR(m[3], float4(13, 14, 15, 16));
    }
}

TEST_CASE("matrix_from_columns")
{
    // 2x4
    {
        math::matrix<float, 2, 4> m = math::matrix_from_columns(float2(1, 2), float2(3, 4), float2(5, 6), float2(7, 8));
        CHECK_EQ_VECTOR(m[0], float4(1, 3, 5, 7));
        CHECK_EQ_VECTOR(m[1], float4(2, 4, 6, 8));
    }

    // 4x2
    {
        math::matrix<float, 4, 2> m = math::matrix_from_columns(float4(1, 2, 3, 4), float4(5, 6, 7, 8));
        CHECK_EQ_VECTOR(m[0], float2(1, 5));
        CHECK_EQ_VECTOR(m[1], float2(2, 6));
        CHECK_EQ_VECTOR(m[2], float2(3, 7));
        CHECK_EQ_VECTOR(m[3], float2(4, 8));
    }

    // 4x4
    {
        float4x4 m = math::matrix_from_columns(
            float4(1, 2, 3, 4),
            float4(5, 6, 7, 8),
            float4(9, 10, 11, 12),
            float4(13, 14, 15, 16)
        );
        CHECK_EQ_VECTOR(m[0], float4(1, 5, 9, 13));
        CHECK_EQ_VECTOR(m[1], float4(2, 6, 10, 14));
        CHECK_EQ_VECTOR(m[2], float4(3, 7, 11, 15));
        CHECK_EQ_VECTOR(m[3], float4(4, 8, 12, 16));
    }
}

TEST_CASE("matrix_from_diagonal")
{
    // 3x3
    {
        float3x3 m = math::matrix_from_diagonal(float3(1, 2, 3));
        CHECK_EQ_VECTOR(m[0], float3(1, 0, 0));
        CHECK_EQ_VECTOR(m[1], float3(0, 2, 0));
        CHECK_EQ_VECTOR(m[2], float3(0, 0, 3));
    }

    // 4x4
    {
        float4x4 m = math::matrix_from_diagonal(float4(1, 2, 3, 4));
        CHECK_EQ_VECTOR(m[0], float4(1, 0, 0, 0));
        CHECK_EQ_VECTOR(m[1], float4(0, 2, 0, 0));
        CHECK_EQ_VECTOR(m[2], float4(0, 0, 3, 0));
        CHECK_EQ_VECTOR(m[3], float4(0, 0, 0, 4));
    }
}

TEST_CASE("perspective")
{
    float4x4 m = math::perspective(math::radians(45.f), 2.f, 0.1f, 1000.f);
    CHECK_ALMOST_EQ(m[0], float4(1.207107f, 0, 0, 0));
    CHECK_ALMOST_EQ(m[1], float4(0, 2.414213f, 0, 0));
    CHECK_ALMOST_EQ(m[2], float4(0, 0, -1.0001f, -0.1f));
    CHECK_ALMOST_EQ(m[3], float4(0, 0, -1.f, 0));
}

TEST_CASE("ortho")
{
    float4x4 m = math::ortho(-10.f, 10.f, -10.f, 10.f, 0.1f, 1000.f);
    CHECK_ALMOST_EQ(m[0], float4(0.1f, 0, 0, 0));
    CHECK_ALMOST_EQ(m[1], float4(0, 0.1f, 0, 0));
    CHECK_ALMOST_EQ(m[2], float4(0, 0, -0.001f, -0.0001f));
    CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1.0f));
}

TEST_CASE("matrix_from_translation")
{
    float4x4 m = math::matrix_from_translation(float3(1, 2, 3));
    CHECK_EQ_VECTOR(m[0], float4(1, 0, 0, 1));
    CHECK_EQ_VECTOR(m[1], float4(0, 1, 0, 2));
    CHECK_EQ_VECTOR(m[2], float4(0, 0, 1, 3));
    CHECK_EQ_VECTOR(m[3], float4(0, 0, 0, 1));
}

TEST_CASE("matrix_from_rotation")
{
    // Rotation around X-axis by 90 degrees
    {
        float4x4 m = math::matrix_from_rotation(math::radians(90.f), float3(1, 0, 0));
        CHECK_ALMOST_EQ(m[0], float4(1, 0, 0, 0));
        CHECK_ALMOST_EQ(m[1], float4(0, 0, -1, 0));
        CHECK_ALMOST_EQ(m[2], float4(0, 1, 0, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around X-axis by -45 degrees
    {
        float4x4 m = math::matrix_from_rotation(math::radians(-45.f), float3(1, 0, 0));
        CHECK_ALMOST_EQ(m[0], float4(1, 0, 0, 0));
        CHECK_ALMOST_EQ(m[1], float4(0, 0.707106f, 0.707106f, 0));
        CHECK_ALMOST_EQ(m[2], float4(0, -0.707106f, 0.707106f, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Y-axis by 90 degrees
    {
        float4x4 m = math::matrix_from_rotation(math::radians(90.f), float3(0, 1, 0));
        CHECK_ALMOST_EQ(m[0], float4(0, 0, 1, 0));
        CHECK_ALMOST_EQ(m[1], float4(0, 1, 0, 0));
        CHECK_ALMOST_EQ(m[2], float4(-1, 0, 0, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Y-axis by -45 degrees
    {
        float4x4 m = math::matrix_from_rotation(math::radians(-45.f), float3(0, 1, 0));
        CHECK_ALMOST_EQ(m[0], float4(0.707106f, 0, -0.707106f, 0));
        CHECK_ALMOST_EQ(m[1], float4(0, 1, 0, 0));
        CHECK_ALMOST_EQ(m[2], float4(0.707106f, 0, 0.707106f, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Z-axis by 90 degrees
    {
        float4x4 m = math::matrix_from_rotation(math::radians(90.f), float3(0, 0, 1));
        CHECK_ALMOST_EQ(m[0], float4(0, -1, 0, 0));
        CHECK_ALMOST_EQ(m[1], float4(1, 0, 0, 0));
        CHECK_ALMOST_EQ(m[2], float4(0, 0, 1, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Z-axis by -45 degrees
    {
        float4x4 m = math::matrix_from_rotation(math::radians(-45.f), float3(0, 0, 1));
        CHECK_ALMOST_EQ(m[0], float4(0.707106f, 0.707106f, 0, 0));
        CHECK_ALMOST_EQ(m[1], float4(-0.707106f, 0.707106f, 0, 0));
        CHECK_ALMOST_EQ(m[2], float4(0, 0, 1, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around oblique axis
    {
        float4x4 m = math::matrix_from_rotation(math::radians(60.f), normalize(float3(1, 1, 1)));
        CHECK_ALMOST_EQ(m[0], float4(0.666666f, -0.333333f, 0.666666f, 0.f));
        CHECK_ALMOST_EQ(m[1], float4(0.666666f, 0.666666f, -0.333333f, 0.f));
        CHECK_ALMOST_EQ(m[2], float4(-0.333333f, 0.666666f, 0.666666f, 0.f));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }
}

TEST_CASE("matrix_from_rotation_xyz")
{
    // Rotation around X-axis by 90 degrees
    {
        float4x4 m = math::matrix_from_rotation_x(math::radians(90.f));
        CHECK_ALMOST_EQ(m[0], float4(1, 0, 0, 0));
        CHECK_ALMOST_EQ(m[1], float4(0, 0, -1, 0));
        CHECK_ALMOST_EQ(m[2], float4(0, 1, 0, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around X-axis by 90 degrees
    {
        float4x4 m = math::matrix_from_rotation_xyz(math::radians(90.f), 0.f, 0.f);
        CHECK_ALMOST_EQ(m[0], float4(1, 0, 0, 0));
        CHECK_ALMOST_EQ(m[1], float4(0, 0, -1, 0));
        CHECK_ALMOST_EQ(m[2], float4(0, 1, 0, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around X-axis by -45 degrees
    {
        float4x4 m = math::matrix_from_rotation_x(math::radians(-45.f));
        CHECK_ALMOST_EQ(m[0], float4(1, 0, 0, 0));
        CHECK_ALMOST_EQ(m[1], float4(0, 0.707106f, 0.707106f, 0));
        CHECK_ALMOST_EQ(m[2], float4(0, -0.707106f, 0.707106f, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around X-axis by -45 degrees
    {
        float4x4 m = math::matrix_from_rotation_xyz(math::radians(-45.f), 0.f, 0.f);
        CHECK_ALMOST_EQ(m[0], float4(1, 0, 0, 0));
        CHECK_ALMOST_EQ(m[1], float4(0, 0.707106f, 0.707106f, 0));
        CHECK_ALMOST_EQ(m[2], float4(0, -0.707106f, 0.707106f, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Y-axis by 90 degrees
    {
        float4x4 m = math::matrix_from_rotation_y(math::radians(90.f));
        CHECK_ALMOST_EQ(m[0], float4(0, 0, 1, 0));
        CHECK_ALMOST_EQ(m[1], float4(0, 1, 0, 0));
        CHECK_ALMOST_EQ(m[2], float4(-1, 0, 0, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Y-axis by 90 degrees
    {
        float4x4 m = math::matrix_from_rotation_xyz(0.f, math::radians(90.f), 0.f);
        CHECK_ALMOST_EQ(m[0], float4(0, 0, 1, 0));
        CHECK_ALMOST_EQ(m[1], float4(0, 1, 0, 0));
        CHECK_ALMOST_EQ(m[2], float4(-1, 0, 0, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Y-axis by -45 degrees
    {
        float4x4 m = math::matrix_from_rotation_y(math::radians(-45.f));
        CHECK_ALMOST_EQ(m[0], float4(0.707106f, 0, -0.707106f, 0));
        CHECK_ALMOST_EQ(m[1], float4(0, 1, 0, 0));
        CHECK_ALMOST_EQ(m[2], float4(0.707106f, 0, 0.707106f, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Y-axis by -45 degrees
    {
        float4x4 m = math::matrix_from_rotation_xyz(0.f, math::radians(-45.f), 0.f);
        CHECK_ALMOST_EQ(m[0], float4(0.707106f, 0, -0.707106f, 0));
        CHECK_ALMOST_EQ(m[1], float4(0, 1, 0, 0));
        CHECK_ALMOST_EQ(m[2], float4(0.707106f, 0, 0.707106f, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Z-axis by 90 degrees
    {
        float4x4 m = math::matrix_from_rotation_z(math::radians(90.f));
        CHECK_ALMOST_EQ(m[0], float4(0, -1, 0, 0));
        CHECK_ALMOST_EQ(m[1], float4(1, 0, 0, 0));
        CHECK_ALMOST_EQ(m[2], float4(0, 0, 1, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Z-axis by 90 degrees
    {
        float4x4 m = math::matrix_from_rotation_xyz(0.f, 0.f, math::radians(90.f));
        CHECK_ALMOST_EQ(m[0], float4(0, -1, 0, 0));
        CHECK_ALMOST_EQ(m[1], float4(1, 0, 0, 0));
        CHECK_ALMOST_EQ(m[2], float4(0, 0, 1, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Z-axis by -45 degrees
    {
        float4x4 m = math::matrix_from_rotation_z(math::radians(-45.f));
        CHECK_ALMOST_EQ(m[0], float4(0.707106f, 0.707106f, 0, 0));
        CHECK_ALMOST_EQ(m[1], float4(-0.707106f, 0.707106f, 0, 0));
        CHECK_ALMOST_EQ(m[2], float4(0, 0, 1, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Z-axis by -45 degrees
    {
        float4x4 m = math::matrix_from_rotation_xyz(0.f, 0.f, math::radians(-45.f));
        CHECK_ALMOST_EQ(m[0], float4(0.707106f, 0.707106f, 0, 0));
        CHECK_ALMOST_EQ(m[1], float4(-0.707106f, 0.707106f, 0, 0));
        CHECK_ALMOST_EQ(m[2], float4(0, 0, 1, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around XYZ by 45 degrees
    {
        float4x4 m = math::matrix_from_rotation_xyz(math::radians(45.f), math::radians(45.f), math::radians(45.f));
        CHECK_ALMOST_EQ(m[0], float4(0.5f, -0.5f, 0.707107f, 0.f));
        CHECK_ALMOST_EQ(m[1], float4(0.853553f, 0.146446f, -0.5f, 0.f));
        CHECK_ALMOST_EQ(m[2], float4(0.146446f, 0.853553f, 0.5f, 0.f));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around XYZ by 20, 40, 60 degrees
    {
        float4x4 m = math::matrix_from_rotation_xyz(math::radians(20.f), math::radians(40.f), math::radians(60.f));
        CHECK_ALMOST_EQ(m[0], float4(0.383022f, -0.663414f, 0.642787f, 0));
        CHECK_ALMOST_EQ(m[1], float4(0.923720f, 0.279453f, -0.262002f, 0));
        CHECK_ALMOST_EQ(m[2], float4(-0.005813f, 0.694109f, 0.719846f, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }
}

TEST_CASE("matrix_from_scaling")
{
    float4x4 m = math::matrix_from_scaling(float3(2.f, 3.f, 4.f));
    CHECK_ALMOST_EQ(m[0], float4(2, 0, 0, 0));
    CHECK_ALMOST_EQ(m[1], float4(0, 3, 0, 0));
    CHECK_ALMOST_EQ(m[2], float4(0, 0, 4, 0));
    CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
}

TEST_CASE("matrix_from_look_at")
{
    // Right handed
    {
        float4x4 m = math::matrix_from_look_at(
            float3(10, 5, 0),
            float3(0, -5, 0),
            float3(0, 1, 0),
            math::Handedness::right_handed
        );
        CHECK_ALMOST_EQ(m[0], float4(0, 0, -1, 0));
        CHECK_ALMOST_EQ(m[1], float4(-0.707107f, 0.707107f, 0, 3.535535f));
        CHECK_ALMOST_EQ(m[2], float4(0.707107f, 0.707107f, 0, -10.606603f));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Left handed
    {
        float4x4 m = math::matrix_from_look_at(
            float3(10, 5, 0),
            float3(0, -5, 0),
            float3(0, 1, 0),
            math::Handedness::left_handed
        );
        CHECK_ALMOST_EQ(m[0], float4(0, 0, 1, 0));
        CHECK_ALMOST_EQ(m[1], float4(-0.707107f, 0.707107f, 0, 3.535535f));
        CHECK_ALMOST_EQ(m[2], float4(-0.707107f, -0.707107f, 0, 10.606603f));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }
}

TEST_CASE("matrix_from_quat")
{
    // Identity quaternion
    {
        float4x4 m = math::matrix_from_quat(quatf::identity());
        CHECK_ALMOST_EQ(m[0], float4(1, 0, 0, 0));
        CHECK_ALMOST_EQ(m[1], float4(0, 1, 0, 0));
        CHECK_ALMOST_EQ(m[2], float4(0, 0, 1, 0));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around oblique axis
    {
        quatf q = math::quat_from_angle_axis(math::radians(60.f), normalize(float3(1, 1, 1)));
        float4x4 m = math::matrix_from_quat(q);
        CHECK_ALMOST_EQ(m[0], float4(0.666666f, -0.333333f, 0.666666f, 0.f));
        CHECK_ALMOST_EQ(m[1], float4(0.666666f, 0.666666f, -0.333333f, 0.f));
        CHECK_ALMOST_EQ(m[2], float4(-0.333333f, 0.666666f, 0.666666f, 0.f));
        CHECK_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }
}

TEST_CASE("translate_2d")
{
    float3x3 m({
        ROW(1, 0, 10),
        ROW(0, -1, 20),
        ROW(0, 0, 1),
    });
    float3x3 m2 = translate_2d(m, float2(1, 2));
    CHECK_ALMOST_EQ(m2[0], float3(1, 0, 11));
    CHECK_ALMOST_EQ(m2[1], float3(0, -1, 18));
    CHECK_ALMOST_EQ(m2[2], float3(0, 0, 1));
}

TEST_CASE("rotate_2d")
{
    float3x3 m({
        ROW(1, 0, 10),
        ROW(0, -1, 20),
        ROW(0, 0, 1),
    });
    float3x3 m2 = rotate_2d(m, math::radians(90.f));
    CHECK_ALMOST_EQ(m2[0], float3(0, -1, 10));
    CHECK_ALMOST_EQ(m2[1], float3(-1, 0, 20));
    CHECK_ALMOST_EQ(m2[2], float3(0, 0, 1));
}

TEST_CASE("rotate_2d_direction")
{
    // Test to verify that rotate_2d follows counter-clockwise convention
    float3x3 identity = float3x3::identity();

    // Rotate identity matrix by 90 degrees
    float3x3 rotated = rotate_2d(identity, math::radians(90.f));

    // The result should be equivalent to matrix_from_rotation_2d(90°)
    float3x3 expected = math::matrix_from_rotation_2d(math::radians(90.f));

    CHECK_ALMOST_EQ(rotated[0], expected[0]);
    CHECK_ALMOST_EQ(rotated[1], expected[1]);
    CHECK_ALMOST_EQ(rotated[2], expected[2]);
}

TEST_CASE("scale_2d")
{
    float3x3 m({
        ROW(1, 0, 10),
        ROW(0, -1, 20),
        ROW(0, 0, 1),
    });
    float3x3 m2 = scale_2d(m, float2(2, 3));
    CHECK_ALMOST_EQ(m2[0], float3(2, 0, 10));
    CHECK_ALMOST_EQ(m2[1], float3(0, -3, 20));
    CHECK_ALMOST_EQ(m2[2], float3(0, 0, 1));
}

TEST_CASE("matrix_from_translation_2d")
{
    float3x3 m = math::matrix_from_translation_2d(float2(1, 2));
    CHECK_EQ_VECTOR(m[0], float3(1, 0, 1));
    CHECK_EQ_VECTOR(m[1], float3(0, 1, 2));
    CHECK_EQ_VECTOR(m[2], float3(0, 0, 1));
}

TEST_CASE("matrix_from_rotation_2d")
{
    // Rotation by 90 degrees
    {
        float3x3 m = math::matrix_from_rotation_2d(math::radians(90.f));
        CHECK_ALMOST_EQ(m[0], float3(0, -1, 0));
        CHECK_ALMOST_EQ(m[1], float3(1, 0, 0));
        CHECK_ALMOST_EQ(m[2], float3(0, 0, 1));
    }

    // Rotation by -45 degrees
    {
        float3x3 m = math::matrix_from_rotation_2d(math::radians(-45.f));
        CHECK_ALMOST_EQ(m[0], float3(0.707106f, 0.707106f, 0));
        CHECK_ALMOST_EQ(m[1], float3(-0.707106f, 0.707106f, 0));
        CHECK_ALMOST_EQ(m[2], float3(0, 0, 1));
    }

    // Rotation by 180 degrees
    {
        float3x3 m = math::matrix_from_rotation_2d(math::radians(180.f));
        CHECK_ALMOST_EQ(m[0], float3(-1, 0, 0));
        CHECK_ALMOST_EQ(m[1], float3(0, -1, 0));
        CHECK_ALMOST_EQ(m[2], float3(0, 0, 1));
    }

    // Identity rotation (0 degrees)
    {
        float3x3 m = math::matrix_from_rotation_2d(0.f);
        CHECK_ALMOST_EQ(m[0], float3(1, 0, 0));
        CHECK_ALMOST_EQ(m[1], float3(0, 1, 0));
        CHECK_ALMOST_EQ(m[2], float3(0, 0, 1));
    }
}

TEST_CASE("matrix_from_scaling_2d")
{
    float3x3 m = math::matrix_from_scaling_2d(float2(2.f, 3.f));
    CHECK_ALMOST_EQ(m[0], float3(2, 0, 0));
    CHECK_ALMOST_EQ(m[1], float3(0, 3, 0));
    CHECK_ALMOST_EQ(m[2], float3(0, 0, 1));
}

TEST_CASE("matrix_2d_rotation_direction")
{
    // Sanity check that the rotation is counter clockwise!
    float3x3 rot90 = math::matrix_from_rotation_2d(math::radians(90.f));
    float3 point(1.f, 0.f, 1.f); // Point at (1, 0)
    float3 rotated = mul(rot90, point);

    // For counter-clockwise rotation: (1,0) -> (0,1)
    // For clockwise rotation: (1,0) -> (0,-1)
    CHECK_ALMOST_EQ(rotated, float3(0.f, 1.f, 1.f)); // Expecting counter-clockwise
}

TEST_CASE("rotate_2d_vs_matrix_multiplication")
{
    // Test that rotate_2d(matrix, angle) gives the same result as
    // mul(matrix, matrix_from_rotation_2d(angle))

    // Test with various starting matrices
    float3x3 test_matrices[]
        = {// Identity matrix
           float3x3::identity(),

           // Translation matrix
           math::matrix_from_translation_2d(float2(5.f, 3.f)),

           // Scaling matrix
           math::matrix_from_scaling_2d(float2(2.f, 0.5f)),

           // Complex transformation matrix
           float3x3({
               ROW(2, 1, 4),
               ROW(-1, 3, 7),
               ROW(0, 0, 1),
           })
        };

    float angles[] = {0.f, math::radians(30.f), math::radians(90.f), math::radians(-45.f), math::radians(180.f)};

    for (const auto& matrix : test_matrices) {
        for (float angle : angles) {
            // Method 1: Using rotate_2d function
            float3x3 result1 = rotate_2d(matrix, angle);

            // Method 2: Using matrix multiplication
            // rotate_2d(m, angle) is equivalent to m * rotation_matrix
            // or equivalently m * matrix_from_rotation_2d(-angle)
            float3x3 rotation_matrix = math::matrix_from_rotation_2d(angle);
            float3x3 result2 = mul(matrix, rotation_matrix);

            // Results should be identical
            CHECK_ALMOST_EQ(result1[0], result2[0]);
            CHECK_ALMOST_EQ(result1[1], result2[1]);
            CHECK_ALMOST_EQ(result1[2], result2[2]);
        }
    }
}

TEST_CASE("2d_transform_combinations")
{
    // Build transformation matrix using matrix_from functions
    float3x3 transform_matrix = math::matrix_from_translation_2d(float2(5.f, 10.f));
    transform_matrix = mul(transform_matrix, math::matrix_from_rotation_2d(math::radians(90.f)));
    transform_matrix = mul(transform_matrix, math::matrix_from_scaling_2d(float2(2.f, 3.f)));

    // Transform point (1, 1) -> scale to (2, 3) -> rotate 90° to (-3, 2) -> translate to (2, 12)
    float3 point(1.f, 1.f, 1.f);
    float3 transformed = mul(transform_matrix, point);
    CHECK_ALMOST_EQ(transformed, float3(2.f, 12.f, 1.f));
}

TEST_CASE("formatter")
{
    float3x3 test0({1.1f, 1.2f, 1.3f, 2.1f, 2.2f, 2.3f, 3.1f, 3.2f, 3.3f});

    CHECK_EQ(fmt::format("{}", test0), "{{1.1, 1.2, 1.3}, {2.1, 2.2, 2.3}, {3.1, 3.2, 3.3}}");
    CHECK_EQ(
        fmt::format("{:e}", test0),
        "{{1.100000e+00, 1.200000e+00, 1.300000e+00}, "
        "{2.100000e+00, 2.200000e+00, 2.300000e+00}, "
        "{3.100000e+00, 3.200000e+00, 3.300000e+00}}"
    );
    CHECK_EQ(fmt::format("{:g}", test0), "{{1.1, 1.2, 1.3}, {2.1, 2.2, 2.3}, {3.1, 3.2, 3.3}}");
    CHECK_EQ(fmt::format("{:.1}", test0), "{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}");
    CHECK_EQ(fmt::format("{:.2f}", test0), "{{1.10, 1.20, 1.30}, {2.10, 2.20, 2.30}, {3.10, 3.20, 3.30}}");
}

TEST_SUITE_END();
