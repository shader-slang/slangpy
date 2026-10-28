// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct BufferInfo
{
    StructuredBuffer<float>.Handle ro_buffer;
    RWStructuredBuffer<float>.Handle rw_buffer;
    uint offset;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main(uint3 tid : SV_DispatchThreadID, StructuredBuffer<BufferInfo> buffer_infos, RWStructuredBuffer<float> results, RWStructuredBuffer<uint4> dimensions)
{
    uint index = tid.x;
    BufferInfo info = buffer_infos[index];
    uint ro_count, ro_stride, rw_count, rw_stride;
    info.ro_buffer.GetDimensions(ro_count, ro_stride);
    info.rw_buffer.GetDimensions(rw_count, rw_stride);
    dimensions[index] = uint4(ro_count, ro_stride, rw_count, rw_stride);

    // Read from read-only buffer
    float value = info.ro_buffer[info.offset];

    // Write to read-write buffer (add 100 to distinguish from read-only)
    info.rw_buffer[info.offset] = value + 100.0;

    // Store result for verification
    results[index] = value;
}
