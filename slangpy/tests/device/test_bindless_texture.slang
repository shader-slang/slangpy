// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct TextureInfo
{
    Texture2D<float>.Handle texture;
    SamplerState.Handle sampler;
    float2 uv;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main(uint3 tid : SV_DispatchThreadID, StructuredBuffer<TextureInfo> texture_infos, RWStructuredBuffer<float> results)
{
    uint index = tid.x;
    TextureInfo info = texture_infos[index];
    results[index] = info.texture.SampleLevel(info.sampler, info.uv, 0);
}
