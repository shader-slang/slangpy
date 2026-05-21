// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Software rasterizer compute shaders for ImGui rendering.
// This is a fallback path for when we don't have native graphics API support (e.g. on CUDA).
// The rasterizer is a simple tile-based design with bucket culling.
// The main motivation is to be able to run ImGui rendering on CUDA, but it should be portable to other compute backends as well.

// TODO: This is a temporary workaround until we have proper format conversion support on CUDA.
// see https://github.com/shader-slang/slang/issues/11088
// The hack expects a rgba8 format and treats it as a RWTexture<uint> on CUDA,
// doing manual packing/unpacking of RGBA values to uints.
#define CUDA_SURFACE_CONVERSION_HACK defined(__TARGET_CUDA__)

#ifndef FORMAT_ATTR
#define FORMAT_ATTR [format("rgba8")] // placeholder to avoid syntax errors
#endif

/// Subpixel precision (number of subpixels per pixel).
static constexpr int SUBPIXEL = 16;
/// Tile size.
static constexpr uint TILE_SIZE = 16;
/// Number of words (32 bits) in the tile bitmask. Each bit corresponds to a bucket of triangles.
#define TILE_BITMASK_WORDS 4

/// Input vertex format (20 bytes per vertex).
struct Vertex {
    float x;
    float y;
    float u;
    float v;
    uint c;
};

/// Precomputed triangle data for rasterization (64 bytes per triangle).
struct Triangle {
    /// Vertex UVs (24 bytes)
    float2 uv0, uv1, uv2;
    /// Subpixel vertex positions (24 bytes)
    int2 p0, p1, p2;
    /// Signed area in subpixel^2 (4 bytes)
    int area;
    /// Packed RGBA vertex colors (12 bytes)
    uint c0, c1, c2;
};

/// Bounding box for a triangle, in pixel coordinates (8 bytes per triangle).
struct BBox {
    int16_t2 min;
    int16_t2 max;
};

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

/// One thread per triangle. Reads vertices/indices, writes precomputed triangle data, packed bbox,
/// and sets bits in per-tile bitmasks for bucket culling.
[shader("compute")]
[numthreads(256, 1, 1)]
void setup_triangles(
    uint3 tid: SV_DispatchThreadID,
    StructuredBuffer<Vertex> vertices,
    StructuredBuffer<uint16_t> indices,
    RWStructuredBuffer<Triangle> triangles,
    RWStructuredBuffer<BBox> bboxes,
    RWStructuredBuffer<uint> tile_bitmasks,
    uniform uint start_vertex,
    uniform uint start_index,
    uniform uint triangle_count,
    uniform uint triangle_offset,
    uniform int2 clip_min,
    uniform int2 clip_max,
    uniform uint tile_bitmask_offset,
    uniform uint tiles_x,
    uniform uint tiles_y,
    uniform uint bucket_size
)
{
    uint tri_idx = tid.x;
    if (tri_idx >= triangle_count)
        return;

    uint base = tri_idx * 3;
    Vertex v0 = vertices[start_vertex + indices[start_index + base + 0]];
    Vertex v1 = vertices[start_vertex + indices[start_index + base + 1]];
    Vertex v2 = vertices[start_vertex + indices[start_index + base + 2]];

    int2 p0 = int2(int(v0.x * SUBPIXEL), int(v0.y * SUBPIXEL));
    int2 p1 = int2(int(v1.x * SUBPIXEL), int(v1.y * SUBPIXEL));
    int2 p2 = int2(int(v2.x * SUBPIXEL), int(v2.y * SUBPIXEL));

    // Signed area (2x, in subpixel^2 space)
    int area = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);

    // Compute bbox in pixel space
    BBox bbox;
    bbox.min = int16_t2(max(clip_min, (min(p0, min(p1, p2))) / SUBPIXEL));
    bbox.max = int16_t2(min(clip_max, (max(p0, max(p1, p2)) + SUBPIXEL - 1) / SUBPIXEL));

    uint dst = triangle_offset + tri_idx;

    // Write packed bbox
    bboxes[dst] = bbox;

    // Write triangle data
    Triangle td;
    td.p0 = p0;
    td.p1 = p1;
    td.p2 = p2;
    td.area = area;
    td.uv0 = float2(v0.u, v0.v);
    td.uv1 = float2(v1.u, v1.v);
    td.uv2 = float2(v2.u, v2.v);
    td.c0 = v0.c;
    td.c1 = v1.c;
    td.c2 = v2.c;

    triangles[dst] = td;

    // Set tile bitmask bits for bucket culling.
    // Determine which bucket this triangle belongs to.
    uint bucket_index = tri_idx / bucket_size;
    uint word_index = bucket_index / 32;
    uint word_bit = 1u << (bucket_index % 32);

    // Compute tile range that this triangle's bbox overlaps (relative to clip rect).
    // bbox is already clamped to clip rect, so we just need to convert to tile coords.
    uint2 tile_min = uint2(max(0, int2(bbox.min) - clip_min)) / TILE_SIZE;
    uint2 tile_max = uint2((max(0, int2(bbox.max) - clip_min)) + TILE_SIZE - 1) / TILE_SIZE;

    // Set the bucket bit in all overlapping tiles.
    for (uint ty = tile_min.y; ty < tile_max.y; ty++) {
        for (uint tx = tile_min.x; tx < tile_max.x; tx++) {
            uint tile_index = ty * tiles_x + tx;
            uint addr = tile_bitmask_offset + tile_index * TILE_BITMASK_WORDS + word_index;
            InterlockedOr(tile_bitmasks[addr], word_bit);
        }
    }
}

groupshared uint shared_mask[TILE_BITMASK_WORDS];

/// One thread per pixel. Uses tile bitmasks to find candidate triangles, then does bbox and edge tests,
/// and blends the triangle color into the output texture.
/// Also clears tile bitmasks for next frame (self-clearing).
/// Note that the triangle data and bboxes are indexed by the global triangle ID, not the bucket index,
/// so we need to iterate over all triangles in the bucket and do a bbox test before fetching triangle data.
[shader("compute")]
[numthreads(16, 16, 1)]
void draw_triangles(
    uint3 tid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    uint group_index: SV_GroupIndex,
    StructuredBuffer<Triangle> triangles,
    StructuredBuffer<BBox> bboxes,
    RWStructuredBuffer<uint> tile_bitmasks,
    SamplerState sampler,
    Texture2D<float4> texture,
#if CUDA_SURFACE_CONVERSION_HACK
    RWTexture2D<uint> output_texture,
#else
    FORMAT_ATTR RWTexture2D<float4> output_texture,
#endif
    uniform bool is_srgb_format,
    uniform uint triangle_count,
    uniform uint triangle_offset,
    uniform int2 clip_min,
    uniform int2 clip_max,
    uniform uint tile_bitmask_offset,
    uniform uint tiles_x,
    uniform uint bucket_size
)
{
    // Load tile bitmask into groupshared memory.
    uint tile_index = gid.y * tiles_x + gid.x;
    uint tile_base = tile_bitmask_offset + tile_index * TILE_BITMASK_WORDS;
    if (group_index < TILE_BITMASK_WORDS)
        shared_mask[group_index] = tile_bitmasks[tile_base + group_index];
    GroupMemoryBarrierWithGroupSync();

    int2 dim;
    output_texture.GetDimensions(dim.x, dim.y);
    int2 pixel = int2(tid.xy) + clip_min;
    bool valid = all(pixel < dim) && all(pixel >= clip_min) && all(pixel < clip_max);

    // If tile bitmask is zero, skip the entire tile.
    if (valid) {
        valid = false;
        bool tile_has_work = false;
        [ForceUnroll]
        for (uint w = 0; w < TILE_BITMASK_WORDS; w++) {
            if (shared_mask[w] != 0) {
                valid = true;
                break;
            }
        }
    }

    float4 dst;
    if (valid) {
        const int px = pixel.x * SUBPIXEL + SUBPIXEL / 2;
        const int py = pixel.y * SUBPIXEL + SUBPIXEL / 2;

#if CUDA_SURFACE_CONVERSION_HACK
        uint raw = (output_texture[pixel]);
        dst = float4(raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF, (raw >> 24) & 0xFF) / 255.0;
#else
        dst = output_texture[pixel];
#endif

        // Iterate over buckets using the bitmask.
        for (uint w = 0; w < TILE_BITMASK_WORDS; w++) {
            uint mask = shared_mask[w];
            while (mask != 0) {
                uint bit = firstbitlow(mask);
                mask &= mask - 1;

                uint bucket = w * 32 + bit;
                uint start = bucket * bucket_size;
                uint end = min(start + bucket_size, triangle_count);

                for (uint i = start; i < end; i++) {
                    // Bbox rejection from compact separate buffer (8 bytes per tri)
                    BBox bbox = bboxes[triangle_offset + i];
                    if (pixel.x < bbox.min.x || pixel.x >= bbox.max.x || pixel.y < bbox.min.y || pixel.y >= bbox.max.y)
                        continue;

                    // Only fetch full triangle data after bbox pass
                    Triangle tri = triangles[triangle_offset + i];

                    // Skip degenerate triangles
                    if (tri.area == 0)
                        continue;

                    // Edge evaluation
                    int e0_dx = tri.p1.x - tri.p0.x;
                    int e0_dy = tri.p1.y - tri.p0.y;
                    int e1_dx = tri.p2.x - tri.p1.x;
                    int e1_dy = tri.p2.y - tri.p1.y;
                    int e2_dx = tri.p0.x - tri.p2.x;
                    int e2_dy = tri.p0.y - tri.p2.y;

                    int e0 = e0_dx * (py - tri.p0.y) - e0_dy * (px - tri.p0.x);
                    int e1 = e1_dx * (py - tri.p1.y) - e1_dy * (px - tri.p1.x);
                    int e2 = e2_dx * (py - tri.p2.y) - e2_dy * (px - tri.p2.x);

                    bool backface = tri.area < 0;
                    if (backface) {
                        if (e0 > 0 || e1 > 0 || e2 > 0)
                            continue;
                    } else {
                        if (e0 < 0 || e1 < 0 || e2 < 0)
                            continue;
                    }

                    // Barycentric weights
                    float w0 = float(e1) / float(tri.area);
                    float w1 = float(e2) / float(tri.area);
                    float w2 = float(e0) / float(tri.area);

                    // Interpolate UV
                    float u = w0 * tri.uv0.x + w1 * tri.uv1.x + w2 * tri.uv2.x;
                    float v = w0 * tri.uv0.y + w1 * tri.uv1.y + w2 * tri.uv2.y;
                    u = fmod(u, 1.0);
                    v = fmod(v, 1.0);

                    // Interpolate vertex color
                    float4 c0 = convert_color(tri.c0);
                    float4 c1 = convert_color(tri.c1);
                    float4 c2 = convert_color(tri.c2);
                    float4 src = w0 * c0 + w1 * c1 + w2 * c2;

                    // Sample texture and blend
                    float4 sampled = texture.Sample(sampler, float2(u, v));
                    src *= sampled;
                    float a = src.a;
                    dst = (1.0 - a) * dst + src * a;
                }
            }
        }

#if CUDA_SURFACE_CONVERSION_HACK
        raw = (
            ((uint(min(255.0, dst.r * 255.0 + 0.5)) & 0xFF)) |
            ((uint(min(255.0, dst.g * 255.0 + 0.5)) & 0xFF) << 8) |
            ((uint(min(255.0, dst.b * 255.0 + 0.5)) & 0xFF) << 16) |
            ((uint(min(255.0, dst.a * 255.0 + 0.5)) & 0xFF) << 24)
        );
        output_texture[pixel] = raw;
#else
        output_texture[pixel] = dst;
#endif
    }

    // Clear the tile bitmask for next frame (self-clearing).
    GroupMemoryBarrierWithGroupSync();
    if (group_index < TILE_BITMASK_WORDS)
        tile_bitmasks[tile_base + group_index] = 0;
}
