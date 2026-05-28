// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct V2F {
    float4 pos : SV_Position;
};

uniform float2 vert_offset;
uniform float2 vert_scale;
uniform float vert_z;

[shader("vertex")]
V2F vertex_main(float3 pos: POSITION)
{
    V2F o;
    o.pos = float4(pos.xy * vert_scale.xy + vert_offset.xy, vert_z, 1);
    return o;
}

uniform float4 frag_color;

[shader("fragment")]
float4 fragment_main(V2F v)
    : SV_Target
{
    return frag_color;
}
