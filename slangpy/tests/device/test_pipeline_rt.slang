// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#if !defined(__TARGET_CUDA__)
[shader("compute")]
[numthreads(16, 16, 1)]
void raygrid(
    uint2 tid: SV_DispatchThreadID,
    RWTexture2D<float4> render_texture,
    RaytracingAccelerationStructure tlas,
    uniform int2 pos,
    uniform int2 size,
    uniform float dist
)
{
    if (any(tid.xy >= size))
        return;

    RayDesc ray;
    ray.Origin = float3(tid.xy, 0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0;
    ray.TMax = 2;

    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, 0, 0xff, ray);
    q.Proceed();
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float4 color = float4(0, 0, 0, 1);
        color[q.CommittedInstanceID() % 3] = 1;
        render_texture[tid.xy] = color;
    } else {
        render_texture[tid.xy] = float4(0, 0, 0, 1);
    }
}
#endif

struct Payload {
    float4 color;
}

[shader("miss")]
void rt_miss(inout Payload payload)
{
    payload.color = float4(0, 0, 0, 1);
}

[shader("closesthit")]
void rt_closest_hit(inout Payload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    float4 col = float4(0, 0, 0, 1);
    col[InstanceID() % 3] = 1;
    payload.color = col;
}

uniform RWTexture2D<float4> rt_render_texture;
uniform RaytracingAccelerationStructure rt_tlas;

[shader("raygeneration")]
void rt_ray_gen()
{
    uint2 pixel = DispatchRaysIndex().xy;

    RayDesc ray;
    ray.Origin = float3(pixel.xy, 0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0;
    ray.TMax = 2;

    Payload payload = {};

    TraceRay(
        rt_tlas,
        0,
        0xff,
        0 /* RayContributionToHitGroupIndex */,
        0 /* MultiplierForGeometryContributionHitGroupIndex */,
        0 /* MissShaderIndex */,
        ray,
        payload
    );

    rt_render_texture[pixel] = payload.color;
}
