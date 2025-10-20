// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Vertex {
    float x;
    float y;
    float u;
    float v;
    uint c;
};

[format("rgba8")]
RWTexture2D<float4> output_texture;
SamplerState sampler;
Texture2D<float4> texture;
StructuredBuffer<Vertex> vertices;
StructuredBuffer<uint16_t> indices;
uniform bool is_srgb_format;

static constexpr int SUBPIXEL = 16;

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
    uniform uint start_vertex,
    uniform uint start_index,
    uniform uint vertex_count,
    uniform int2 clip_min,
    uniform int2 clip_max
)
{
    int2 dim;
    output_texture.GetDimensions(dim.x, dim.y);
    int2 pixel = tid.xy;
    if (any(pixel >= dim))
        return;
    if (any(pixel < clip_min) || any(pixel >= clip_max))
        return;

    // center sample
    const int px = pixel.x * SUBPIXEL + SUBPIXEL / 2;
    const int py = pixel.y * SUBPIXEL + SUBPIXEL / 2;

    // float4 dst = output_texture.Load(pixel);
    float4 dst = 0.0;
    bool written = false;

    for (uint i = 0; i < vertex_count; i += 3) {
        Vertex v0 = vertices[start_vertex + indices[start_index + i]];
        Vertex v1 = vertices[start_vertex + indices[start_index + i + 1]];
        Vertex v2 = vertices[start_vertex + indices[start_index + i + 2]];

        int x0 = int(v0.x * SUBPIXEL);
        int y0 = int(v0.y * SUBPIXEL);
        int x1 = int(v1.x * SUBPIXEL);
        int y1 = int(v1.y * SUBPIXEL);
        int x2 = int(v2.x * SUBPIXEL);
        int y2 = int(v2.y * SUBPIXEL);

        int minx = min(x0, min(x1, x2));
        int maxx = max(x0, max(x1, x2));
        int miny = min(y0, min(y1, y2));
        int maxy = max(y0, max(y1, y2));

        // Early out if pixel center is outside triangle bbox.
        if (px < minx || px >= maxx || py < miny || py >= maxy)
            continue;

        // Compute edge deltas.
        int e0_dx = x1 - x0;
        int e0_dy = y1 - y0;
        int e1_dx = x2 - x1;
        int e1_dy = y2 - y1;
        int e2_dx = x0 - x2;
        int e2_dy = y0 - y2;

        // Compute edge values at pixel center.
        int e0 = e0_dx * (py - y0) - e0_dy * (px - x0);
        int e1 = e1_dx * (py - y1) - e1_dy * (px - x1);
        int e2 = e2_dx * (py - y2) - e2_dy * (px - x2);

        // Check for backfacing triangle.
        int area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
        if (area == 0)
            continue;
        bool backface = area < 0;

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
        float w0 = (float)e1 / area;
        float w1 = (float)e2 / area;
        float w2 = (float)e0 / area;

        // interpolate uv and color (we stored premultiplied color components in verts)
        float u = w0 * v0.u + w1 * v1.u + w2 * v2.u;
        float v = w0 * v0.v + w1 * v1.v + w2 * v2.v;
        u = fmod(u, 1.0);
        v = fmod(v, 1.0);

        float4 c0 = convert_color(v0.c);
        float4 c1 = convert_color(v1.c);
        float4 c2 = convert_color(v2.c);

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
