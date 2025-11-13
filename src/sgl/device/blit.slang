// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This shader uses preprocessor macros to adapt to different texture types and layouts.
// Ideally we would use Slang's link time specialization to achieve the same effect,
// but there are some limitations in the current implementation that prevent us from doing so.
// Mainly we cannot use link time specialization to change the resource format of the source texture
// as well as the render target output type.

// This shader expects the following defines to be set externally:
// - SRC_LAYOUT (LAYOUT_TEXTURE2D, LAYOUT_TEXTURE2D_ARRAY)
// - SRC_KIND (KIND_FLOAT, KIND_INT)
// - SRC_CHANNEL_COUNT (1, 2, 4)
// - SRC_TYPE (float, float2, float4, uint, uint2, uint4)
// - DST_KIND (KIND_FLOAT, KIND_INT)
// - DST_CHANNEL_COUNT (1, 2, 4)
// - DST_TYPE (float, float2, float4, uint, uint2, uint4)
// - DST_FORMAT_ATTR (slang texture format attribute like [format("rgba8")])
// - DST_SRGB (0 or 1, whether to convert the output to sRGB, only used in compute shader)

#define KIND_FLOAT 0
#define KIND_INT 1

#define LAYOUT_TEXTURE2D 0
#define LAYOUT_TEXTURE2D_ARRAY 1

#if SRC_LAYOUT == LAYOUT_TEXTURE2D
typedef Texture2D<SRC_TYPE> SrcTexture;
#elif SRC_LAYOUT == LAYOUT_TEXTURE2D_ARRAY
typedef Texture2DArray<SRC_TYPE> SrcTexture;
#else
#error "Invalid SRC_LAYOUT"
#endif

extension Texture2D<SRC_TYPE>
{
    SRC_TYPE sample(SamplerState s, float2 uv)
    {
#if SRC_KIND == KIND_FLOAT
        return this.Sample(s, uv);
#elif SRC_KIND == KIND_INT
        return asuint(this.Sample(s, uv));
#endif
    }
}

extension Texture2DArray<SRC_TYPE>
{
    SRC_TYPE sample(SamplerState s, float2 uv)
    {
#if SRC_KIND == KIND_FLOAT
        return this.Sample(s, float3(uv, 0));
#elif SRC_KIND == KIND_INT
        return asuint(this.Sample(s, float3(uv, 0)));
#endif
    }
}

SrcTexture src;
SamplerState sampler;

#ifndef __TARGET_CUDA__

struct VSOut {
    float4 pos : SV_Position;
    float2 uv : UV;
};

[shader("vertex")]
VSOut vs_main(uint vid: SV_VertexID)
{
    VSOut vs_out;
    vs_out.uv = float2((vid << 1) & 2, vid & 2);
    vs_out.pos = float4(vs_out.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return vs_out;
}

[shader("fragment")]
DST_TYPE fs_main(VSOut vs_out)
    : SV_Target
{
    float2 uv = vs_out.uv;
    DST_TYPE value = src.sample(sampler, uv);
    return value;
}

#endif // __TARGET_CUDA__

float linear_to_srgb(float linear)
{
    if (linear <= 0.0031308)
        return linear * 12.92;
    else
        return pow(linear, (1.0 / 2.4)) * (1.055) - 0.055;
}

float3 linear_to_srgb(float3 linear)
{
    return float3(linear_to_srgb(linear.x), linear_to_srgb(linear.y), linear_to_srgb(linear.z));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID, DST_FORMAT_ATTR RWTexture2D<DST_TYPE> dst)
{
    uint2 dim;
    dst.GetDimensions(dim.x, dim.y);
    if (any(tid.xy >= dim))
        return;
    float2 uv = (float2(tid.xy) + 0.5) / dim;
    DST_TYPE value = src.sample(sampler, uv);
#if DST_SRGB == 1
    value.xyz = linear_to_srgb(value.xyz);
#endif
    dst[tid.xy] = value;
}
