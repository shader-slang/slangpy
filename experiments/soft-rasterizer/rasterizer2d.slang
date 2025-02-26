// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import "slangpy";

// Simple Camera struct to convert between screen space and world space
// coordinates, so that vertex positions can be specified in world space.
struct Camera
{
    // World origin.
    float2 o;

    // World scale.
    float2 scale;

    // Frame dimensions (i.e. image resolution).
    float2 frameDim;

    // Convert from
    // screen coordinates [(0, 0), (W, H)] to
    // world coordinates [(o.x - scale.x, o.y - scale.y), (o.x + scale.x, o.y + scale.y)]
    [Differentiable]
    float2 screenToWorld(float2 uv)
    {
        float2 xy = uv / frameDim;
        float2 ndc = xy * 2.0f - 1.0f;
        return ndc * scale + o;
    }
};

// Simple Vertex struct, only stores position but more data can be added
// as needed, such as color, normal, or texture coordinates.
struct Vertex
{
    float3 position;

    __init(float3 p)
    {
        position = p;
    }
}


[Differentiable]
float distanceToVertex(float2 xy2, float2 v)
{
    // Compute the distance from a point to a vertex
    return length(xy2 - v);
}

[Differentiable]
float distanceToEdge(float2 u, float2 v, float2 pt)
{
    float2 e = v - u;
    float2 n = float2(-e.y, e.x);
    float2 d = pt - u;
    float n_dot_d = dot(n, d);

    // Compute the signed distance from a point to an edge
    if (dot(pt - u, v - u) < 0.f)
        return distanceToVertex(pt, u) * (sign(n_dot_d)); // u is the closest vertex
    else if (dot(pt - v, u - v) < 0.f)
        return distanceToVertex(pt, v) * (sign(n_dot_d)); // v is the closest vertex
    else
    {
        return n_dot_d / length(n); // closest edge
    }
}

[Differentiable]
float distanceToTriangle(float2 xy, float2 v1, float2 v2, float2 v3)
{
    // Minimum distance to the edge of the triangle
    float d1 = distanceToEdge(v2, v1, xy);
    float d2 = distanceToEdge(v3, v2, xy);
    float d3 = distanceToEdge(v1, v3, xy);

    int sd1 = sign(d1);
    int sd2 = sign(d2);
    int sd3 = sign(d3);

    if (sd1 > 0.f && sd2 > 0.f)
        return distanceToVertex(xy, v2); // v2 is the closest vertex
    else if (sd1 > 0.f && sd3 > 0.f)
        return distanceToVertex(xy, v1); // v1 is the closest vertex
    else if (sd2 > 0.f && sd3 > 0.f)
        return distanceToVertex(xy, v3); // v3 is the closest vertex
    else 
        return max(max(d1, d2), d3);

}

[Differentiable]
float sigmoid(float x, float sigma)
{
    return 1.0 / (1.0 + exp(-x / sigma));
}

[Differentiable]
float softTriangle(float2 xy, float2 v1, float2 v2, float2 v3, float sigma)
{
    float d = distanceToTriangle(xy, v1, v2, v3);
    return sigmoid(-d, sigma);
}

// Rasterize function called from python, called for each pixel in the window.
// The soft rasterizer instead computes a probability value for a pixel being
// within the triangle, rather than the simple boolean value of the
// fwd-rasterizer. These functions are differentiable, allowing returning
// gradient values for training.
[Differentiable]
float4 rasterize(Camera camera, float2 vertices[3], int2 pixelCoord)
{
    // Adjust pixel coordinate to have 0,0 at the bottom left for ease of use.
    float2 fragCoord = float2(pixelCoord);
    fragCoord.y = camera.frameDim.y - fragCoord.y;

    // Three vertices of the triangle, passed in from python.
    float2 v1 = vertices[0];
    float2 v2 = vertices[1];
    float2 v3 = vertices[2];

    // Convert the screen space fragCoord to world space.
    float2 screenSample = fragCoord.xy + 0.5;
    float2 worldSample = camera.screenToWorld(screenSample);

    // Sigma for the sigmoid function
    float sigma = 0.02;

    // Green triangle color.
    float4 triangleColor = float4(0.0, 1.0, 0.0, 1.0);

    // Compute hit probability for the pixel being inside the triangle bounds.
    float hit = softTriangle(worldSample, v1, v2, v3, sigma);
    float4 outColor = hit * triangleColor;
    return outColor;
}
