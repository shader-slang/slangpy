// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

[shader("compute")]
[numthreads(16, 16, 1)]
void clear(uint3 tid: SV_DispatchThreadID, RWTexture2D<float4> render_texture)
{
    uint2 dim;
    render_texture.GetDimensions(dim.x, dim.y);
    if (any(tid.xy >= dim))
        return;
    render_texture[tid.xy] = float4(0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void count(uint3 tid: SV_DispatchThreadID, Texture2D<float4> render_texture, RWByteAddressBuffer count_buffer)
{
    uint2 dim;
    render_texture.GetDimensions(dim.x, dim.y);
    if (any(tid.xy >= dim))
        return;
    float4 val = render_texture[tid.xy];
    count_buffer.InterlockedAdd(0, val.x > 0 ? 1 : 0);
    count_buffer.InterlockedAdd(4, val.y > 0 ? 1 : 0);
    count_buffer.InterlockedAdd(8, val.z > 0 ? 1 : 0);
    count_buffer.InterlockedAdd(12, val.w > 0 ? 1 : 0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void setcolor(
    uint3 tid: SV_DispatchThreadID,
    RWTexture2D<float4> render_texture,
    uniform int2 pos,
    uniform int2 size,
    uniform float4 color
)
{
    if (any(tid.xy >= size))
        return;
    render_texture[tid.xy + pos] = color;
}
