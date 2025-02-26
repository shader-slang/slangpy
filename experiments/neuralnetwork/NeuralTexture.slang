// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

__exported import network.NeuralModules;
__exported import network.ArrayModule;
__exported import network.Activations;
__exported import network.MLP;
__exported import network.Encodings;
__exported import network.Losses;
__exported import network.CoopVecModule;
__exported import network.CoopVecActivations;
__exported import network.CoopVecMLP;

// Have to include instead of import currently (waiting on MR #97)
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

float4 evalModel<Model : IModule<float2, float3>>(Model model, float2 inputUV)
{
    return float4(model.forward(inputUV), 1.0f);
}

[BackwardDifferentiable]
float evalModelAndLoss<Model : IModule<float2, float3>>(Model model, no_diff float2 input, no_diff float3 target)
{
    let lossFunc = L2Loss();
    
    float3 prediction = float3(model.forward(input));

    return lossFunc.eval(prediction, target);
}

void trainTexture<Model : IModule<float2, float3>>(Model model, inout RNG rng, Texture2D<float4> targetTex, SamplerState sampler, float lossScale)
{
    float2 inputUV = rng.next2D();

    float4 target = targetTex.SampleLevel(sampler, inputUV, 0.0f);

    bwd_diff(evalModelAndLoss)(model, inputUV, target.rgb, lossScale);
}

/*void test(inout RNG rng, Texture2D<float4> targetTex, SamplerState sampler, float lossScale)
{
    ChainedModulePair<half[2], half[20], half[3], FrequencyEncoding<half, 2, 5>, ChainedModulePair<half[20], DiffCoopVec<half, 20>, half[3], ArrayToCoopVec<half, 20>, ChainedModulePair<DiffCoopVec<half, 20>, DiffCoopVec<half, 64>, half[3], CoopVecLinearLayer<half, 20, 64> , ChainedModulePair<DiffCoopVec<half, 64>, DiffCoopVec<half, 64>, half[3], CoopVecAct::LeakyReLU<half, 64>, ChainedModulePair<DiffCoopVec<half, 64>, DiffCoopVec<half, 64>, half[3], CoopVecLinearLayer<half, 64, 64> , ChainedModulePair<DiffCoopVec<half, 64>, DiffCoopVec<half, 64>, half[3], CoopVecAct::LeakyReLU<half, 64>, ChainedModulePair<DiffCoopVec<half, 64>, DiffCoopVec<half, 64>, half[3], CoopVecLinearLayer<half, 64, 64> , ChainedModulePair<DiffCoopVec<half, 64>, DiffCoopVec<half, 64>, half[3], CoopVecAct::LeakyReLU<half, 64>, ChainedModulePair<DiffCoopVec<half, 64>, DiffCoopVec<half, 3>, half[3], CoopVecLinearLayer<half, 64, 3> , ChainedModulePair<DiffCoopVec<half, 3>, DiffCoopVec<half, 3>, half[3], CoopVecAct::Sigmoid<half, 3>, CoopVecToArray<half, 3>>>>>>>>>>> model;

    trainTexture(model, rng, targetTex, sampler, lossScale);
}*/
