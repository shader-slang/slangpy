// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import slangpy;

[Differentiable]
float[5] matrix_vector_direct(float weights[5][8], float biases[5], float[8] x)
{
    float[5] result;
    for (int r = 0; r < 5; ++r)
    {
        float y = biases[r];
        for (int c = 0; c < 8; ++c)
            y += weights[r][c] * x[c];
        result[r] = y;
    }
    return result;
}

float[5] matrix_vector_tensorized(Tensor<float, 2> weights, Tensor<float, 1> biases, float[8] x)
{
    float[5] result;
    for (int r = 0; r < 5; ++r)
    {
        float y = biases[r];
        for (int c = 0; c < 8; ++c)
            y += weights[r, c] * x[c];

        result[r] = y;
    }
    return result;
}

[Differentiable]
float3 matrix_vector_matrices(float3x4 weights, float4 biases, float[4] x)
{
    float3 result;
    for (int r = 0; r < 3; ++r)
    {
        float y = biases[r];
        for (int c = 0; c < 4; ++c)
            y += weights[r][c] * x[c];

        result[r] = y;
    }
    return result;
}

[Differentiable]
float[5] matrix_vector_interfaces(ITensor<float, 2> weights, ITensor<float, 1> biases, float[8] x)
{
    float[5] result;
    for (int r = 0; r < 5; ++r)
    {
        float y = biases.get({r}); // Workaround for slang bug #5921: Have to avoid [] operator, or derivatives vanish
        for (int c = 0; c < 8; ++c)
            y += weights.get({r, c}) * x[c];

        result[r] = y;
    }
    return result;
}

[Differentiable]
float[FanOut] matrix_vector_generic<let FanIn : int, let FanOut : int>(ITensor<float, 2> weights, ITensor<float, 1> biases, float[FanIn] x)
{
    float[FanOut] result;
    [ForceUnroll]
    for (int r = 0; r < FanOut; ++r)
    {
        float y = biases.get({r});
        [ForceUnroll]
        for (int c = 0; c < FanIn; ++c)
            y += weights.get({r, c}) * x[c];

        result[r] = y;
    }
    return result;
}
