// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

RWTexture2D<float4> g_texture;
uniform uint g_checker_size;

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    uint2 dim;
    g_texture.GetDimensions(dim.x, dim.y);
    uint2 pixel = tid.xy;
    if (any(pixel >= dim))
        return;
    // checkerboard pattern
    uint2 i = pixel / max(g_checker_size, 1);
    float4 color = (i.x % 2 == i.y % 2) ? float4(1, 1, 1, 1) : float4(0, 0, 0, 1);
    g_texture[pixel] = color;
}
