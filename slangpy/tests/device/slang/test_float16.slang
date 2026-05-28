// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

[shader("compute")]
[numthreads(16, 1, 1)]
void main_srv(uint3 tid: SV_DispatchThreadID, StructuredBuffer<float16_t> data, RWStructuredBuffer<uint16_t> result)
{
    uint i = tid.x;
    float16_t value = data[i]; // 16-bit load
    result[i] = uint16_t(f32tof16(value));
}

[shader("compute")]
[numthreads(16, 1, 1)]
void main_uav(uint3 tid: SV_DispatchThreadID, RWStructuredBuffer<float16_t> data, RWStructuredBuffer<uint16_t> result)
{
    uint i = tid.x;
    float16_t value = data[i]; // 16-bit load
    result[i] = uint16_t(f32tof16(value));
}
