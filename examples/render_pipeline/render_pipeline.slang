// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

[shader("vertex")]
float4 vertex_main(float2 pos: POSITION)
    : SV_Position
{
    return float4(pos, 0.0, 1.0);
}

[shader("fragment")]
float4 fragment_main(float4 pos: SV_Position, float3 barycentrics: SV_Barycentrics)
    : SV_Target
{
    return float4(barycentrics, 1.f);
}
