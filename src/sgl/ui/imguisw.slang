// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Vertex {
    float x;
    float y;
    float u;
    float v;
    uint c;
};

struct Triangle {
    int x0;
    int y0;
    int x1;
    int y1;
    int x2;
    int y2;
    int bbox_min_x;
    int bbox_min_y;
    int bbox_max_x;
    int bbox_max_y;
    float inv_area;
    float u0;
    float v0;
    float u1;
    float v1;
    float u2;
    float v2;
    float c0_r;
    float c0_g;
    float c0_b;
    float c0_a;
    float c1_r;
    float c1_g;
    float c1_b;
    float c1_a;
    float c2_r;
    float c2_g;
    float c2_b;
    float c2_a;
};

struct TileHeader {
    uint triangle_offset;
    uint triangle_count;
};

[format("rgba8")]
RWTexture2D<float4> output_texture;
SamplerState sampler;
Texture2D<float4> texture;
StructuredBuffer<Triangle> triangles;
StructuredBuffer<TileHeader> tile_headers;
StructuredBuffer<uint> tile_triangle_indices;
uniform bool is_srgb_format;

static constexpr int SUBPIXEL = 16;
static constexpr uint TILE_SIZE = 16;

float4 convert_color(uint c)
{
    float4 result;
    result.r = ((c >> 0) & 0xFF);
    result.g = ((c >> 8) & 0xFF);
    result.b = ((c >> 16) & 0xFF);
    result.a = ((c >> 24) & 0xFF);
    result /= 255.0;
    return result;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void cs_main(
    uint3 tid: SV_DispatchThreadID,
    uniform int2 dispatch_origin,
    uniform uint tile_header_offset,
    uniform uint tile_grid_width
)
{
    int2 dim;
    output_texture.GetDimensions(dim.x, dim.y);
    int2 local_pixel = int2(tid.xy);
    int2 pixel = dispatch_origin + local_pixel;
    if (any(pixel >= dim))
        return;

    // center sample
    const int px = pixel.x * SUBPIXEL + SUBPIXEL / 2;
    const int py = pixel.y * SUBPIXEL + SUBPIXEL / 2;

    float4 dst = output_texture[pixel];
    bool written = false;

    uint tile_x = uint(local_pixel.x) / TILE_SIZE;
    uint tile_y = uint(local_pixel.y) / TILE_SIZE;
    TileHeader tile = tile_headers[tile_header_offset + tile_y * tile_grid_width + tile_x];

    for (uint i = 0; i < tile.triangle_count; ++i) {
        Triangle tri = triangles[tile_triangle_indices[tile.triangle_offset + i]];

        if (pixel.x < tri.bbox_min_x || pixel.x >= tri.bbox_max_x || pixel.y < tri.bbox_min_y || pixel.y >= tri.bbox_max_y)
            continue;

        // Compute edge deltas.
        int e0_dx = tri.x1 - tri.x0;
        int e0_dy = tri.y1 - tri.y0;
        int e1_dx = tri.x2 - tri.x1;
        int e1_dy = tri.y2 - tri.y1;
        int e2_dx = tri.x0 - tri.x2;
        int e2_dy = tri.y0 - tri.y2;

        // Compute edge values at pixel center.
        int e0 = e0_dx * (py - tri.y0) - e0_dy * (px - tri.x0);
        int e1 = e1_dx * (py - tri.y1) - e1_dy * (px - tri.x1);
        int e2 = e2_dx * (py - tri.y2) - e2_dy * (px - tri.x2);

        // Check for backfacing triangle.
        bool backface = tri.inv_area < 0.0;

        // using top-left fill rule: require e >= 0 for edges that are top-left, else > 0
        // Here we use >= 0 everywhere, but you should compute per-edge top-left flag to be exact.
        if (backface) {
            if (e0 > 0 || e1 > 0 || e2 > 0)
                continue; // outside triangle
        } else {
            if (e0 < 0 || e1 < 0 || e2 < 0)
                continue; // outside triangle
        }

        // barycentric weights from edge functions (use fixed-point values but convert to float)
        float w0 = float(e1) * tri.inv_area;
        float w1 = float(e2) * tri.inv_area;
        float w2 = float(e0) * tri.inv_area;

        // interpolate uv and color (we stored premultiplied color components in verts)
        float u = w0 * tri.u0 + w1 * tri.u1 + w2 * tri.u2;
        float v = w0 * tri.v0 + w1 * tri.v1 + w2 * tri.v2;

        float4 c0 = float4(tri.c0_r, tri.c0_g, tri.c0_b, tri.c0_a);
        float4 c1 = float4(tri.c1_r, tri.c1_g, tri.c1_b, tri.c1_a);
        float4 c2 = float4(tri.c2_r, tri.c2_g, tri.c2_b, tri.c2_a);

        float4 src = w0 * c0 + w1 * c1 + w2 * c2;

        float4 sampled = texture.Sample(sampler, float2(u, v));
        src *= sampled;

        float a = src.a;
        dst = (1.0 - a) * dst + src * a;
        written = true;
    }

    if (written)
        output_texture[pixel] = dst;
}
