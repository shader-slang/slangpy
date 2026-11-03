// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import "slangpy";

public struct Payload { public float3 color; };

[shader("miss")]
public void miss(inout Payload payload)
{
    payload.color = float3(1, 0, 1);
}

[shader("closesthit")]
public void closest_hit(inout Payload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    payload.color = float3(attribs.barycentrics, 0);
}

public float3 trace(int2 tid, RaytracingAccelerationStructure tlas)
{
    float2 uv = (float2(tid) + 0.5) / 64.0;

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

    return payload.color;
}
