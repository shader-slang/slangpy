// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Params {
    float2 resolution;
    float2 mouse_pos;
    float mouse_radius;
    float time;
    float scale;
    float padding;
};

StructuredBuffer<Params> g_params;

struct VertexOutput {
    float4 position : SV_Position;
};

// Full screen triangle generated from the vertex index.
[shader("vertex")]
VertexOutput vertex_main(uint vertex_id: SV_VertexID)
{
    float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    VertexOutput output;
    output.position = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target
{
    Params params = g_params[0];
    float2 pixel = input.position.xy;
    float2 p = (pixel - 0.5 * params.resolution) / params.resolution.y * params.scale + params.time * 0.1;
    float d = length(frac(p) - 0.5);
    float3 color = smoothstep(0.3, 0.28, d) * (0.6 + 0.4 * cos(params.time + float3(0, 2, 4)));
    color += smoothstep(params.mouse_radius, 0, distance(pixel, params.mouse_pos)) * float3(1, 0, 0);
    return float4(color, 1);
}
