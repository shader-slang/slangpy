// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import coverage_color;

float3 denoise(Texture2D<float4> image, int2 pixel, float3 center, float edgeThreshold)
{
    uint width, height;
    image.GetDimensions(width, height);
    float centerLight = luminance(center);
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
    return sum / weightSum;
}
