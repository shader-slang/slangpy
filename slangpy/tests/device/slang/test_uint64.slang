// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

[shader("compute")]
[numthreads(16, 1, 1)]
void main_srv(uint3 tid: SV_DispatchThreadID, StructuredBuffer<uint64_t> data, RWStructuredBuffer<uint> result)
{
    uint i = tid.x;
    uint64_t value = data[i]; // 64-bit load
    uint lo = (uint)value;
    uint hi = (uint)(value >> 32);
    result[2 * i] = lo;
    result[2 * i + 1] = hi;
}

[shader("compute")]
[numthreads(16, 1, 1)]
void main_uav(uint3 tid: SV_DispatchThreadID, RWStructuredBuffer<uint64_t> data, RWStructuredBuffer<uint> result)
{
    uint i = tid.x;
    uint64_t value = data[i]; // 64-bit load
    uint lo = (uint)value;
    uint hi = (uint)(value >> 32);
    result[2 * i] = lo;
    result[2 * i + 1] = hi;
}
