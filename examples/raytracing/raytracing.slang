// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

RaytracingAccelerationStructure tlas;
RWTexture2D<float4> render_texture;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint2 dim;
    render_texture.GetDimensions(dim.x, dim.y);
    if (any(tid.xy >= dim))
        return;

    float2 uv = float2(tid.xy) / float2(dim);

    RayDesc ray;
    ray.Origin = float3(uv * 2 - 1, 1);
    ray.Direction = float3(0, 0, -1);
    ray.TMin = 0;
    ray.TMax = 2;

    float3 color = float3(0);

    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, 0, 0xff, ray);
    q.Proceed();
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        color = float3(q.CommittedTriangleBarycentrics(), 0);
    }

    render_texture[tid.xy] = float4(color, 1);
}
