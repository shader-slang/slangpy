// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
struct AdamOptimizer
{
    float2 moments;

    __init(float param)
    {
        moments = float2(0.0f, -1.0f);
    }

    [mutating]
    void step(inout float param, inout float rawGrad, float learningRate, float gradientScale)
    {
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float epsilon = 1e-6f;

        bool isFirstIteration = moments.y < 0.0f;
        float blendX = isFirstIteration ? 0.0f : beta1;
        float blendY = isFirstIteration ? 0.0f : beta2;

        float grad = rawGrad / gradientScale;
        float gradSq = grad * grad;

        moments.x = lerp(grad, moments.x, blendX);
        moments.y = lerp(gradSq, moments.y, blendY);

        param -= learningRate * moments.x / (sqrt(moments.y) + epsilon);
        rawGrad = 0.0f;
    }
}
