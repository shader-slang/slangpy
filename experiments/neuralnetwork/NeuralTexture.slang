// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

__exported import NeuralNetworks;

struct RNG
{
    uint state;

    __init(uint state) { this.state = state; }

    [mutating]
    uint next()
    {
        state = state * 2739110765U + 2739110765U;
        return state;
    }

    [mutating]
    float next1D()
    {
        // Use upper 24 bits and divide by 2^24 to get a number u in [0,1).
        // In floating-point precision this also ensures that 1.0-u != 0.0.
        uint bits = next();
        return (bits >> 8) * 0x1p-24;
    }

    [mutating]
    float2 next2D()
    {
        return { next1D(), next1D() };
    }
}

float4 evalModel<Model : IModel<float2, float3>>(Model model, float2 uv)
{
    float3 rgb = model.forward(uv);

    return float4(rgb, 1.0f);
}

[BackwardDifferentiable]
float evalModelLoss<Model : IModel<float2, float3>>(Model model, no_diff float2 inputUV, no_diff float3 targetRGB)
{
    let lossFunc = Losses::L2();
    
    float3 prediction = model.forward(inputUV);

    return lossFunc.eval(prediction, targetRGB);
}

void trainTexture<Model : IModel<float2, float3>>(Model model, inout RNG rng, Texture2D<float4> targetTex, SamplerState sampler, float lossScale)
{
    float2 inputUV = rng.next2D();

    float4 target = targetTex.SampleLevel(sampler, inputUV, 0.0f);

    bwd_diff(evalModelLoss)(model, inputUV, target.rgb, lossScale);
}

struct ToGrayscale<T : IReal, int N> : IModel<vector<T, N>, vector<T, N>>
{
    vector<T, N> channelWeights;

    [BackwardDifferentiable]
    OutputType forward(InputType x)
    {
        T gray = dot(channelWeights, x);
        return OutputType(gray);
    }
}