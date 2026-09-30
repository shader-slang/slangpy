// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import coverage_denoise;
import coverage_exposure;
import coverage_tonemap;

float4 denoiseToneMap(
    int2 pixel,
    Texture2D<float4> image,
    float exposure,
    float edgeThreshold,
    bool enableDenoise,
    uint mapperID)
{
    float4 center = image.Load(int3(pixel, 0));
    if (center.a == 0.0)
        return center;

    float3 color = center.rgb;
    if (enableDenoise)
        color = denoise(image, pixel, color, edgeThreshold);
    color = adjustExposure(color, exposure);
    return float4(toneMap(color, mapperID), center.a);
}
