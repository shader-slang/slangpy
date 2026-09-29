// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

float luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float3 toneMap(float3 color)
{
    float light = luminance(color);
    if (light <= 1e-6)
        return float3(0);
    // Normalize overbright colors while preserving their chromaticity.
    if (light > 1.0)
        color /= light;
    return sqrt(saturate(color));
}

float4 denoiseToneMap(
    int2 pixel,
    Texture2D<float4> image,
    float exposure,
    float edgeThreshold,
    bool enableDenoise)
{
    float4 center = image.Load(int3(pixel, 0));
    if (center.a == 0.0)
        return center;

    float3 color = center.rgb;
    if (enableDenoise)
    {
        uint width, height;
        image.GetDimensions(width, height);
        float centerLight = luminance(center.rgb);
        float3 sum = float3(0);
        float weightSum = 0.0;
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                int2 neighborPixel = pixel + int2(dx, dy);
                if (any(neighborPixel < 0) || any(neighborPixel >= int2(width, height)))
                    continue;
                float3 neighbor = image.Load(int3(neighborPixel, 0)).rgb;
                // Reject samples across strong edges instead of blurring them.
                if (abs(luminance(neighbor) - centerLight) > edgeThreshold)
                    continue;
                float weight = 1.0 / float(1 + dx * dx + dy * dy);
                sum += neighbor * weight;
                weightSum += weight;
            }
        }
        // The center sample always passes, so weightSum is nonzero.
        color = sum / weightSum;
    }
    return float4(toneMap(color * exposure), center.a);
}
