// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

float3 aces_film(float3 x)
{
    x *= 0.6;
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

struct ToneMapper {
    Texture2D<float4> input;
    RWTexture2D<float4> output;

    void execute(uint2 pixel)
    {
        float3 i = input[pixel].xyz;
        float3 o = aces_film(i);
        output[pixel] = float4(o, 1.0);
    }
}

ParameterBlock<ToneMapper> g_tone_mapper;

[[shader("compute")]]
[[numthreads(8, 8, 1)]]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    g_tone_mapper.execute(tid.xy);
}
