// SPDX-License-Identifier: Apache-2.0
import "slangpy";

float add_floats(float a, float b)
{
    return a + b;
}

float3 add_vectors(float3 a, float3 b)
{
    return a + b;
}

float4 sample_texture_at_uv(float2 uv, SamplerState sampler, Texture2D<float4> texture)
{
    return texture.SampleLevel(sampler, uv, 0);
}
