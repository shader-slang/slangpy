// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Accumulator {
    Texture2D<float4> input;
    RWTexture2D<float4> output;
    RWTexture2D<float4> accumulator;
    bool reset;

    void execute(uint2 pixel)
    {
        float4 i = input[pixel];
        float4 a = float4(0.0);
        if (!reset)
            a = accumulator[pixel];
        a += float4(i.xyz, 1.0);
        float4 o = float4(a.xyz / a.w, 1.0);
        accumulator[pixel] = a;
        output[pixel] = o;
    }
}

ParameterBlock<Accumulator> g_accumulator;

[[shader("compute")]]
[[numthreads(8, 8, 1)]]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    g_accumulator.execute(tid.xy);
}
