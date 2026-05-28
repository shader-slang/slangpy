// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import bctypes;

extern static const bool USE_ADAM;
extern static const uint OPT_STEPS;

uniform float lr;
uniform float adam_beta_1;
uniform float adam_beta_2;

// Uncompressed input texture
Texture2D<float4> input_tex;

// Decoded BC texture
RWTexture2D<float4> decoded_tex;

[BackwardDifferentiable]
void l2_loss(BCTile tile, no_diff float4 target[16], out float loss[16])
{
    [ForceUnroll]
    for (int i = 0; i < 16; i++) {
        let delta = target[i] - tile.decode(i);
        loss[i] = dot(delta, delta);
    }
}

[shader("compute")]
[numthreads(4, 8, 1)]
void compute_main(int3 tid: SV_DispatchThreadID)
{
    uint2 dim;
    input_tex.GetDimensions(dim.x, dim.y);

    let texel = 4 * tid.xy;
    if (any(texel >= dim))
        return;

    // Adam moments
    BCTile m1, m2;

    // Load target values and initialize with mid-point interpolation
    // and minmax endpoints computed over the tile
    float4 target[16];
    BCTile tile = BCTile(0.5, float4(1), float4(0));
    for (int i = 0; i < 16; i++) {
        target[i] = input_tex[texel + int2(i % 4, i / 4)];
        tile.min_end_points = min(tile.min_end_points, target[i]);
        tile.max_end_points = max(tile.max_end_points, target[i]);
    }

    // To compress the texture we perform a number of gradient descent steps
    // to joint-optimize interpolation weights and endpoints for each tile.
    // Eventually the interpolation weights are quantized and frozen,
    // while the endpoints are fine tuned for a few more steps.
    bool frozen_weight = false;
    let end_points_fine_tuninig_thres = 0.85f;
    for (uint s = 0; s < OPT_STEPS; s++) {
        // Forward pass - compute error
        float loss[16];
        l2_loss(tile, target, loss);

        // Backward pass - compute gradients
        var dp_tile = diffPair(tile);
        __bwd_diff(l2_loss)(dp_tile, target, { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });

        // Use Adam optimizer to update gradients
        BCTile grad = BCTile(dp_tile.d);
        if (USE_ADAM) {
            m1 = adam_beta_1 * m1 + (1 - adam_beta_1) * grad;
            m2 = adam_beta_2 * m2 + (1 - adam_beta_2) * (grad * grad);
            let unbiased_m1 = rcp(1.f - pow(adam_beta_1, 1 + s)) * m1;
            let unbiased_m2 = rcp(1.f - pow(adam_beta_2, 1 + s)) * m2;
            grad = unbiased_m1 * (1E-10f + unbiased_m2).rsqrt();
        }

        // If weights are frozen zero their gradient
        if (frozen_weight)
            grad.weights = BCWeights(0);

        // Update BC tile via gradient descent & clamp its values to a valid range
        tile = tile - lr * grad;
        tile.clamp();

        // Time to freeze the BC weights?
        if (frozen_weight == false && float(s) / OPT_STEPS >= end_points_fine_tuninig_thres) {
            // Quantize & freeze weights
            // We will continue to fine tune the endpoints for a few more steps
            frozen_weight = true;
            tile.weights.quantize();
        }
    }

    // Quantize & write out decoded tile
    tile.quantize();
    for (int i = 0; i < 16; i++)
        decoded_tex[texel + int2(i % 4, i / 4)] = tile.decode_exact(i);
}
