// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

[raypayload]
struct Payload
{
    uint result : read(caller) : write(caller, closesthit, miss);
};

uniform RWStructuredBuffer<uint> result_buffer;
uniform RaytracingAccelerationStructure scene_bvh;

[shader("raygeneration")]
void ray_gen()
{
    uint ray_index = DispatchRaysIndex().x;

    static const float2 origins[] =
    {
        float2(-0.5, 0.0),
        float2(0.5, 0.0),
        float2(1.25, -1.0 / 3.0),
        float2(1.5, -1.0 / 6.0),
        float2(1.75, -1.0 / 3.0),
        float2(1.5, 1.0 / 6.0),
    };

    RayDesc ray;
    ray.Origin = float3(origins[ray_index], 0.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.001;
    ray.TMax = 100.0;

    Payload payload = {0};
    TraceRay(scene_bvh, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);
    result_buffer[ray_index] = payload.result;
}

[shader("miss")]
void miss(inout Payload payload)
{
    payload.result = 1;
}

[shader("closesthit")]
void closest_hit(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes)
{
    payload.result = 2;
}
