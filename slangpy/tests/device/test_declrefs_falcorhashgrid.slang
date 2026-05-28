// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct OptimizerHashGrid<let C : uint, let L : uint, let P : uint> {
    static const uint32_t kNumChannels = C;
    static const uint32_t kNumLevels = L;
    static const uint32_t kNumChannelsPerLevel = P;

    // NOTE: Nested texture arrays are not supported on Vulkan. Using flat arrays, refer to getTexIdx().
    RWTexture2D<float> parameters[kNumLevels * kNumChannelsPerLevel];
    RWTexture2D<float> derivatives[kNumLevels * kNumChannelsPerLevel];
    RWTexture2D<float2> moments[kNumLevels * kNumChannelsPerLevel];

    static const uint kPi1 = 1;
    static const uint kPi2 = 2654435761;
    const uint resolutions[kNumLevels]; // Grid resolutions at the individual levels of the hierarchy.
    const uint2 sizes[kNumLevels];      // Buffer sizes at the individual levels of the hierarchy.

    uint2 getBufferSize(uint level) { return sizes[level]; }

    uint getTexIdx(uint level, uint channel) { return level * kNumChannelsPerLevel + channel; }

    uint2 getDenseIndex(uint u, uint v, uint level) { return uint2(u, v); }

    uint2 getHashedIndex(uint u, uint v, uint level)
    {
        uint index = ((u * kPi1) ^ (v * kPi2)) % (sizes[level].x * sizes[level].y);
        return uint2(index % sizes[level].x, index / sizes[level].x);
    }

    [BackwardDifferentiable]
    void getCodeBilinear(float2 uv, out float code[kNumChannels])
    {
        uv = frac(uv);

        uint2 index00, index10, index01, index11;
        [ForceUnroll]
        for (uint level = 0; level < kNumLevels; level++) {
            float2 uvScaled = uv * resolutions[level];
            uint u0 = uint(floor(uvScaled.x));
            uint u1 = min(u0 + 1, resolutions[level] - 1);
            uint v0 = uint(floor(uvScaled.y));
            uint v1 = min(v0 + 1, resolutions[level] - 1);
            uint numGridEntries = resolutions[level] * resolutions[level];
            if (numGridEntries == sizes[level].x * sizes[level].y) {
                index00 = getDenseIndex(u0, v0, level);
                index10 = getDenseIndex(u1, v0, level);
                index01 = getDenseIndex(u0, v1, level);
                index11 = getDenseIndex(u1, v1, level);
            } else {
                index00 = getHashedIndex(u0, v0, level);
                index10 = getHashedIndex(u1, v0, level);
                index01 = getHashedIndex(u0, v1, level);
                index11 = getHashedIndex(u1, v1, level);
            }

            float2 uvFrac = frac(uvScaled);
            [ForceUnroll]
            for (int ch = 0; ch < kNumChannelsPerLevel; ch++) {
                float code00 = getParam(level, ch, index00);
                float code10 = getParam(level, ch, index10);
                float code01 = getParam(level, ch, index01);
                float code11 = getParam(level, ch, index11);

                code[level * kNumChannelsPerLevel + ch]
                    = lerp(lerp(code00, code10, uvFrac[0]), lerp(code01, code11, uvFrac[0]), uvFrac[1]);
            }
        }
    }

    float getParam(uint level, uint channel, uint2 texel) { return parameters[getTexIdx(level, channel)][texel]; }

    [BackwardDerivativeOf(getParam)]
    void bwd_getParam(uint level, uint channel, uint2 texel, float d_out)
    {
#ifdef FALCOR_D3D12
        // TODO: Update to use Slang overload when https://github.com/shader-slang/slang/issues/2926 is fixed.
        let idx = getTexIdx(level, channel);
        NvInterlockedAddFp32(derivatives[idx], texel, d_out);
#endif
    }

    void getOptimizerState(uint level, uint channel, uint2 texel, out float p, out float d, out float2 m)
    {
        let idx = getTexIdx(level, channel);
        p = parameters[idx][texel];
        d = derivatives[idx][texel];
        m = moments[idx][texel];
    }

    void updateOptimizerState(uint level, uint channel, uint2 texel, float p, float2 m)
    {
        let idx = getTexIdx(level, channel);
        parameters[idx][texel] = p;
        moments[idx][texel] = m;
    }
};

// TODO: autogenerate this type inside `NeuralMaterialModel`.
typedef OptimizerHashGrid<NUM_LATENT_DIMS, NUM_HASHGRID_LEVELS, NUM_LATENT_DIMS_PER_LEVEL> LatentHashGrid;
