// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import coverage_color;

interface IToneMapper
{
    float3 map(float3 color);
}

struct NormalizeToneMapper : IToneMapper
{
    float3 map(float3 color)
    {
        float light = luminance(color);
        if (light <= 1e-6)
            return float3(0);
        if (light > 1.0)
            color /= light;
        return sqrt(saturate(color));
    }
}

struct ReinhardToneMapper : IToneMapper
{
    float3 map(float3 color)
    {
        return sqrt(saturate(color / (1.0 + color)));
    }
}

float3 toneMap(float3 color, uint mapperID)
{
    // Python registers both implementations with IDs 0 and 1. The ID is a
    // runtime argument, so all captures use the same linked program.
    IToneMapper mapper = createDynamicObject<IToneMapper>(mapperID, 0);
    return mapper.map(color);
}
