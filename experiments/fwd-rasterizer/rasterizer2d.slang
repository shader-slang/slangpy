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


// Perform a simple 2D point-in-triangle test, returning true if the test
// position is inside the triangle bounds.
// xy: 2D test position
// v1: vertex position 1
// v2: vertex position 2
// v3: vertex position 3
bool triangleTest(float2 xy, float2 v1, float2 v2, float2 v3)
{
    float2 e0 = v2 - v1;
    float2 e1 = v3 - v2;
    float2 e2 = v1 - v3;

    float2 k0 = float2(-e0.y, e0.x);
    float2 k1 = float2(-e1.y, e1.x);
    float2 k2 = float2(-e2.y, e2.x);

    float d0 = dot(k0, xy - v1);
    float d1 = dot(k1, xy - v2);
    float d2 = dot(k2, xy - v3);

    // Check the three half-plane values.
    if(d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0) {
        return true;
    } else {
        return false;
    }
}

// Rasterize function called from python, called for each pixel in the window.
// This tests if the pixelCoord is inside the triangle bounds, if true then
// set the color to green, otherwise set the color to black.
float4 rasterize(Camera camera, float2 vertices[3], int2 pixelCoord)
{
    // Adjust the incoming pixel coordinate to have 0,0 at the bottom left
    // which also places world space -1,-1 at the bottom left for ease of use.
    float2 fragCoord = float2(pixelCoord);
    fragCoord.y = camera.frameDim.y - fragCoord.y;

    // Default background color is black.
    float4 outColor = float4(0.0, 0.0, 0.0, 0.0);

    // Three vertices of the triangle, passed in from python.
    float2 v1 = vertices[0];
    float2 v2 = vertices[1];
    float2 v3 = vertices[2];

    // Convert the screen space fragCoord to world space.
    float2 screenSample = fragCoord.xy + 0.5;
    float2 worldSample = camera.screenToWorld(screenSample);

    // Render green pixels inside the triangle.
    if (triangleTest(worldSample, v1, v2, v3)) {
        outColor = float4(0.0, 1.0, 0.0, 1.0);
    }

    return outColor;
}
