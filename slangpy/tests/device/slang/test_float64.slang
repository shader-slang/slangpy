// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

[shader("compute")]
[numthreads(16, 1, 1)]
void main_srv(uint3 tid: SV_DispatchThreadID, StructuredBuffer<float64_t> data, RWStructuredBuffer<uint64_t> result)
{
    uint i = tid.x;
    float64_t value = data[i]; // 64-bit load
    uint2 tmp;
    asuint(value, tmp.x, tmp.y);
    result[i] = uint64_t(tmp.x) | uint64_t(tmp.y) << 32;
}

[shader("compute")]
[numthreads(16, 1, 1)]
void main_uav(uint3 tid: SV_DispatchThreadID, RWStructuredBuffer<float64_t> data, RWStructuredBuffer<uint64_t> result)
{
    uint i = tid.x;
    float64_t value = data[i]; // 64-bit load
    uint2 tmp;
    asuint(value, tmp.x, tmp.y);
    result[i] = uint64_t(tmp.x) | uint64_t(tmp.y) << 32;
}
