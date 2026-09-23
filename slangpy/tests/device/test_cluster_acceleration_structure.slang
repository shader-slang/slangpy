// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sgl.device.rhi;

RWStructuredBuffer<rhi::TriangleClusterArgs> triangle_args;
uniform uint64_t index_buffer_address;
uniform uint64_t vertex_buffer_address;
uniform uint vertex_buffer_stride;

[shader("compute")]
[numthreads(1, 1, 1)]
void write_triangle_args(uint3 tid : SV_DispatchThreadID)
{
    // Explicit initialization works around default initialization issues for structures with bit fields.
    rhi::TriangleClusterArgs args;
    args.clusterId = 0;
    args.clusterFlags = rhi::kClusterFlagNone;
    args.triangleCount = 1;
    args.vertexCount = 3;
    args.positionTruncateBitCount = 0;
    args.indexFormat = rhi::kClusterIndexFormat32bit;
    args.opacityMicromapIndexFormat = 0;
    args.baseGeometryIndexAndFlags = 0;
    args.indexBufferStride = uint16_t(0);
    args.vertexBufferStride = uint16_t(vertex_buffer_stride);
    args.geometryIndexAndFlagsBufferStride = uint16_t(0);
    args.opacityMicromapIndexBufferStride = uint16_t(0);
    args.indexBuffer = index_buffer_address;
    args.vertexBuffer = vertex_buffer_address;
    args.geometryIndexAndFlagsBuffer = 0;
    args.opacityMicromapArray = 0;
    args.opacityMicromapIndexBuffer = 0;
    args.instantiationBoundingBoxLimit = 0;
    triangle_args[tid.x] = args;
}

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
    RayDesc ray;
    ray.Origin = float3(0.25, 0.25, 0.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.001;
    ray.TMax = 100.0;

    Payload payload = {0};
    TraceRay(scene_bvh, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);
    result_buffer[0] = payload.result;
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
