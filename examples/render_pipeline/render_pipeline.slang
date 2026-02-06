// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifdef __TARGET_WGPU__

struct VertexOutput
{
    float4 pos : SV_Position;
    float3 barycentrics : BARYCENTRICS;
};

[shader("vertex")]
VertexOutput vertex_main(float2 pos: POSITION, uint vid: SV_VertexID)
{
    VertexOutput output;
    output.pos = float4(pos, 0.0, 1.0);

    // Mimic SV_Barycentrics for backends that don't support it (e.g., WebGPU)
    uint id = vid % 3;
    output.barycentrics = float3(0, 0, 0);
    output.barycentrics[id] = 1.0;

    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target
{
    return float4(input.barycentrics, 1.f);
}

#else

[shader("vertex")]
float4 vertex_main(float2 pos: POSITION)
    : SV_Position
{
    return float4(pos, 0.0, 1.0);
}

[shader("fragment")]
float4 fragment_main(float3 bary: SV_Barycentrics)
    : SV_Target
{
    return float4(bary, 1.f);
}

#endif
