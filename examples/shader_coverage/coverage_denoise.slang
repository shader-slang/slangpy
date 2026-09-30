// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import coverage_color;

float3 denoise(Texture2D<float4> image, int2 pixel, float3 center, float edgeThreshold)
{
    uint width, height;
    image.GetDimensions(width, height);
    float centerLight = luminance(center);
    // Dark pixels need a stricter edge test; HDR neighborhoods tolerate more noise.
    if (centerLight < 0.08)
        edgeThreshold *= 0.5;
    else if (centerLight > 2.0)
        edgeThreshold *= 2.0;
    float3 sum = float3(0);
    float weightSum = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            // Treat the center separately so an isolated feature is preserved exactly.
            if (dx == 0 && dy == 0)
                continue;
            int2 neighborPixel = pixel + int2(dx, dy);
            if (any(neighborPixel < 0) || any(neighborPixel >= int2(width, height)))
                continue;
            float4 sample = image.Load(int3(neighborPixel, 0));
            if (sample.a == 0.0)
                continue;
            float3 neighbor = sample.rgb;
            // Reject samples across strong edges instead of blurring them.
            if (abs(luminance(neighbor) - centerLight) > edgeThreshold)
                continue;
            float weight = 1.0 / float(1 + dx * dx + dy * dy);
            // Give near-identical neighbors more influence than marginal matches.
            if (abs(luminance(neighbor) - centerLight) > edgeThreshold * 0.25)
                weight *= 0.5;
            sum += neighbor * weight;
            weightSum += weight;
        }
    }
    // An isolated bright pixel may have no acceptable neighbors.
    if (weightSum == 0.0)
        return center;
    return 0.5 * (center + sum / weightSum);
}
