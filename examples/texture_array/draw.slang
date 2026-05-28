// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

RWTexture2D g_output;
Texture2DArray<float4> g_texture;
SamplerState g_sampler;
uniform uint g_layer;
float g_mip_level;

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 dims;
    uint2 pixel = tid.xy;
    g_output.GetDimensions(dims.x, dims.y);
    if (any(pixel >= dims))
        return;

    float2 uv = float2(pixel) / dims.xy;

    float4 color = g_texture.Sample(g_sampler, float3(uv, g_layer), uint2(0), g_mip_level);

    g_output[pixel] = color;
}
