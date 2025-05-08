// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

uint32_t pack(float16_t x, float16_t y)
{
    uint16_t ix = asuint16(x);
    uint16_t iy = asuint16(y);
    return (uint32_t(iy) << 16) | ix;
}

[shader("compute")]
[numthreads(16, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID, StructuredBuffer<float16_t> data, RWStructuredBuffer<uint32_t> result)
{
    uint i = tid.x;
    float16_t x = data[2 * i];
    float16_t y = data[2 * i + 1];
    result[i] = pack(x, y);
}
