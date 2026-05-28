// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

RaytracingAccelerationStructure tlas;
RWTexture2D<float4> render_texture;

struct Payload {
    float3 color;
}

[shader("miss")]
void miss(inout Payload payload)
{
    payload.color = float3(0, 0, 0);
}

[shader("closesthit")]
void closest_hit(inout Payload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    payload.color = float3(attribs.barycentrics, 0);
}

[shader("raygeneration")]
void ray_gen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;

    float2 uv = float2(pixel) / float2(dim);

    RayDesc ray;
    ray.Origin = float3(uv * 2 - 1, 1);
    ray.Direction = float3(0, 0, -1);
    ray.TMin = 0;
    ray.TMax = 2;

    Payload payload = {};

    TraceRay(
        tlas,
        0,
        0xff,
        0 /* RayContributionToHitGroupIndex */,
        0 /* MultiplierForGeometryContributionHitGroupIndex */,
        0 /* MissShaderIndex */,
        ray,
        payload
    );

    render_texture[pixel] = float4(payload.color, 1.f);
}
