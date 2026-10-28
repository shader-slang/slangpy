// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct BufferInfo
{
    ByteAddressBuffer.Handle source;
    RWByteAddressBuffer.Handle output;
    RWByteAddressBuffer.Handle gradients;
    uint sentinel;
};

[shader("compute")]
[numthreads(32, 1, 1)]
void compute_main(uint3 tid : SV_DispatchThreadID, StructuredBuffer<BufferInfo> infos, RWStructuredBuffer<uint> dimensions)
{
    BufferInfo info = infos[0];
    if (tid.x < 4)
        info.output.Store<float>(tid.x * 4, info.source.Load<float>(tid.x * 4) + 100.0);
    if (tid.x == 0)
    {
        uint source_size, output_size;
        info.source.GetDimensions(source_size);
        info.output.GetDimensions(output_size);
        dimensions[0] = source_size;
        dimensions[1] = output_size;
        dimensions[2] = info.sentinel;
    }
    // Exercise contended packed-half gradients as well as FP32 atomics.
    half old_value;
    info.gradients.InterlockedAddF16Emulated((tid.x % 2) * 2, half(1.0), old_value);
    info.gradients.InterlockedAddF32(4, 1.0);
}
