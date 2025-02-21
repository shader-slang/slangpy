// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import NeuralModules;

struct AdamOptimizer
{
    float parameter;
    float2 moments;

    __init(half rawParam)
    {
        parameter = float(rawParam);
        moments = float2(0.0f, -1.0f);
    }

    [mutating]
    void step(inout half rawParam, inout half rawGrad, float learningRate, float gradientScale)
    {
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float epsilon = 1e-6f;

        bool isFirstIteration = moments.y < 0.0f;
        float blendX = isFirstIteration ? 0.0f : beta1;
        float blendY = isFirstIteration ? 0.0f : beta2;

        float grad = half(rawGrad) / gradientScale;
        float gradSq = grad * grad;

        moments.x = lerp(grad, moments.x, blendX);
        moments.y = lerp(gradSq, moments.y, blendY);

        parameter -= learningRate * moments.x / (sqrt(moments.y) + epsilon);
        rawParam = half(parameter);
        rawGrad = half(0.0f);
    }
}
