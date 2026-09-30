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
        if (light < 0.08)
            return sqrt(saturate(color * 1.5));
        if (light > 1.0)
            color /= light;
        return sqrt(saturate(color));
    }
}

struct ReinhardToneMapper : IToneMapper
{
    float3 map(float3 color)
    {
        // Desaturate extreme highlights before compressing their dynamic range.
        float light = luminance(color);
        if (max(color.r, max(color.g, color.b)) > 4.0)
            color = lerp(color, float3(light), 0.25);
        return sqrt(saturate(color / (1.0 + color)));
    }
}

float filmicChannel(float value)
{
    // Continuous toe, midtone ramp, and highlight shoulder.
    if (value < 0.05)
        return value * 2.0;
    if (value < 1.0)
        return 0.1 + (0.7 / 0.95) * (value - 0.05);
    return 0.8 + 0.2 * (1.0 - exp(-(value - 1.0)));
}

struct FilmicToneMapper : IToneMapper
{
    float3 map(float3 color)
    {
        return float3(filmicChannel(color.r), filmicChannel(color.g), filmicChannel(color.b));
    }
}

float3 toneMap(float3 color, uint mapperID)
{
    // Python registers three implementations with IDs 0, 1, and 2. The ID is a
    // runtime argument, so all captures use the same linked program.
    IToneMapper mapper = createDynamicObject<IToneMapper>(mapperID, 0);
    return mapper.map(color);
}
