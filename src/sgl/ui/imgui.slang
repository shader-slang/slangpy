// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

SamplerState sampler;
Texture2D texture;
uniform float2 scale;
uniform float2 offset;
uniform bool is_srgb_format;

struct VsIn {
    float2 pos : POSITION;
    float2 uv : TEXCOORD0;
    float4 col : COLOR;
};

struct VsOut {
    float4 col : COLOR;
    float2 uv : TEXCOORD0;
    float4 pos : SV_Position;
};

float linear_to_srgb(float linear)
{
    return linear < 0.0031308f ? linear * 12.92f : pow(linear, 1.f / 2.4f) * 1.055f - 0.055f;
}

float4 linear_to_srgb(float4 linear)
{
    return float4(linear_to_srgb(linear.r), linear_to_srgb(linear.g), linear_to_srgb(linear.b), linear.a);
}

float srgb_to_linear(float srgb)
{
    return srgb < 0.04045f ? srgb * (1.f / 12.92f) : pow((srgb + 0.055f) * (1.f / 1.055f), 2.4f);
}

float4 srgb_to_linear(float4 srgb)
{
    return float4(srgb_to_linear(srgb.r), srgb_to_linear(srgb.g), srgb_to_linear(srgb.b), srgb.a);
}

[shader("vertex")]
VsOut vs_main(VsIn vs_in)
{
    VsOut vs_out;
    // ImGui outputs sRGB colors, but we want to do all our blending in linear space, so convert to linear here.
    vs_out.col = srgb_to_linear(vs_in.col);
    vs_out.uv = vs_in.uv;
    vs_out.pos = float4(vs_in.pos * scale + offset, 0.f, 1.f);
    return vs_out;
}

[shader("fragment")]
float4 fs_main(VsOut vs_out)
    : SV_Target
{
    float4 col = vs_out.col * texture.Sample(sampler, vs_out.uv);
    // If output texture is not sRGB, convert color to sRGB manually.
    if (!is_srgb_format)
        col = linear_to_srgb(col);
    return col;
}
