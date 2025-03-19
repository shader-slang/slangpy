// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// shaders.slang

struct Uniforms
{
    float2 screenSize;
    float focalLength, frameHeight;
    float3 cameraDir;
    float3 cameraUp;
    float3 cameraRight;
    float3 cameraPosition;
};

bool rayPlaneIntersection(float3 rayOrigin, float3 rayDir, float3 normal, float3 center, out float t)
{
    float denom = dot(normal, rayDir);
    if (abs(denom) > 0.0001f)
    {
        t = dot(center - rayOrigin, normal) / denom;
        return t >= 0;
    }
    t = 0;
    return false;
}

bool raySquareIntersection(float3 rayOrigin, float3 rayDir, float3 center, float3 normal, float3 right, float halfEdgeLength, out float t, out float u, out float v)
{
    u = 0;
    v = 0;
    if (rayPlaneIntersection(rayOrigin, rayDir, normal, center, t))
    {
        float3 intersectionPoint = rayOrigin + rayDir * t;
        float3 toCenter = intersectionPoint - center;
        float3 up = cross(normal, right);
        float rightLength = dot(right, toCenter);
        float upLength = dot(up, toCenter);
        u = rightLength / halfEdgeLength * 0.5 + 0.5;
        v = upLength / halfEdgeLength * 0.5 + 0.5;
        return abs(rightLength) <= halfEdgeLength && abs(upLength) <= halfEdgeLength;
    }
    return false;
}

float4 raytraceScene(
    int2 threadIdx,
    Uniforms uniforms,
    Texture2D tex,
    SamplerState samplerState)
{
    if (threadIdx.x >= (int)uniforms.screenSize.x) return 0;
    if (threadIdx.y >= (int)uniforms.screenSize.y) return 0;

    float frameWidth = uniforms.screenSize.x / uniforms.screenSize.y * uniforms.frameHeight;
    float imageY = (threadIdx.y / uniforms.screenSize.y - 0.5f) * uniforms.frameHeight;
    float imageX = (threadIdx.x / uniforms.screenSize.x - 0.5f) * frameWidth;
    float imageZ = uniforms.focalLength;
    float3 rayDir = normalize(uniforms.cameraDir.xyz*imageZ - uniforms.cameraUp.xyz * imageY + uniforms.cameraRight.xyz * imageX);

    float4 resultColor = 0;
    float t = 0;
    float u = 0, v = 0;
    if (raySquareIntersection(uniforms.cameraPosition, rayDir, float3(0, 0, 0), float3(0, 1, 0), float3(1, 0, 0), 5.0f, t, u, v))
    {
        resultColor = tex.SampleLevel(samplerState, float2(u, v), 0);
    }
    return resultColor;
}
