// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

__exported import network.NeuralModules;
__exported import network.Activations;
__exported import network.MLP;
__exported import network.Encodings;
__exported import network.Losses;

// Has to be an include rather than an import right now
// Otherwise, __init is not exposed to reflection
#include "network/Optimizer.slang"

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

float4 evalModel<Model : IModule<float, 2, 3>>(Model model, float2 inputUV)
{
    let prediction = model.forward(inputUV);
    return float4(prediction, 1.0f);
}

[BackwardDifferentiable]
float evalModelAndLoss<Model : IModule<float, 2, 3>>(Model model, no_diff float2 input, no_diff float3 target)
{
    let lossFunc = L2Loss();
    
    float3 prediction = float3(model.forward(input));

    return lossFunc.eval(prediction, target);
}

void trainTexture<Model : IModule<float, 2, 3>>(Model model, inout RNG rng, Texture2D<float4> targetTex, SamplerState sampler, float lossScale)
{
    float2 inputUV = rng.next2D();

    float4 target = targetTex.SampleLevel(sampler, inputUV, 0.0f);

    bwd_diff(evalModelAndLoss)(model, inputUV, target.rgb, lossScale);
}
