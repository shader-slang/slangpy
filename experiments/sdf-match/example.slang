// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
no_diff float targetSDF(no_diff float3 samplePoint)
{
    return length(samplePoint) - 1.0f;
}

[Differentiable]
float learnedSDF(float3 samplePoint, float[4] sdf_params) {
    float3 center = float3(sdf_params[0], sdf_params[1], sdf_params[2]);
    float dist = length(samplePoint - center);
    return dist - sdf_params[3];
}


[Differentiable]
float forward(no_diff float3 samplePoint, float[4] sdf_params)
{
   float target = targetSDF(samplePoint);
   float predicted = learnedSDF(samplePoint, sdf_params);
   float diff = target - predicted;

   return diff * diff;
}

static const float3 cameraPosition = float3(0.0f, 0.0f, -5.0f);
static const float3 lightPosition = float3(0.0f, 0.0f, -3.0f);
static const int MAX_STEPS = 128;    // Max ray marching steps
static const float MAX_DIST = 100.0; // Maximum render distance
static const float SURFACE_DIST = 0.001; // Surface hit precision
static const float[4] ref_params = {0.0, 0.0, 0.0, 1.0}; // Reference SDF parameters

struct Ray
{
    float3 origin;
    float3 direction;
};

interface ISDFModel
{
    float sdfFunc(float3 p);
    float3 estimateNormal(float3 p);
    float3 getObjColor();
}

struct LearnedSDFModle : ISDFModel
{
    float[4] parameters;
    float3 color = float3(1.0, 0.6, 0.3);

    __init(float[4] inParams, float3 inColor)
    {
        parameters = inParams;
        color = inColor;
    }

    float sdfFunc(float3 p)
    {
        return learnedSDF(p, parameters);
    }

    float3 estimateNormal(float3 p)
    {
        var dp = diffPair(p);
        var dparam = diffPair(parameters);
        bwd_diff(learnedSDF)(dp, dparam, 1.0);
        return normalize(dp.d);
    }

    float3 getObjColor()
    {
        return color;
    }
}

struct TargetSDFModle : ISDFModel
{
    float3 color = float3(0.5, 0.6, 1.0);

    __init(float3 inColor)
    {
        color = inColor;
    }

    float sdfFunc(float3 p)
    {
        return targetSDF(p);
    }

    float3 estimateNormal(float3 p)
    {
        return normalize(p);
    }

    float3 getObjColor()
    {
        return color;
    }
}

struct RayMarcher
{
    Ray ray;
    float3 getModelColor(ISDFModel sdfModel)
    {
        float dist = march(sdfModel);

        float3 color = float3(0.0);
        if (dist > 0.0)
        {
            float3 hitPoint = ray.origin + ray.direction * dist;
            float3 normal = sdfModel.estimateNormal(hitPoint);
            float3 objColor = sdfModel.getObjColor();
            color = objColor * computeLighting(hitPoint, normal);
        }

        return color;
    }

    float march(ISDFModel sdfModel)
    {
        float totalDist = 0.0;
        for (int i = 0; i < MAX_STEPS; i++)
        {
            float3 p = ray.origin + ray.direction * totalDist;
            float dist = sdfModel.sdfFunc(p);

            if (dist < SURFACE_DIST)
                return totalDist; // Hit the surface

            if (totalDist > MAX_DIST)
                return -1.0;     // Exceeded max distance

            totalDist += dist;
        }
        return -1.0; // No hit
    }

    float computeLighting(float3 p, float3 normal)
    {
        float3 lightDir = normalize(lightPosition - p);
        float diffuse = max(dot(normal, lightDir), 0.0);
        return diffuse;
    }
}

float4 RunRayMarch(float2 screenSize, int2 gridId, float sdf_params[4])
{
    if (gridId.x >= screenSize.x || gridId.y >= screenSize.y)
        return float4(0.0);

    float2 uv = (gridId.xy * 1.0f / screenSize.xy) * 2.0 - 1.0;    // Normalized screen coordinates to [-1, 1]
    uv.x *= screenSize.x / screenSize.y; // Maintain aspect rati

    float3 rayOrigin = cameraPosition;
    float3 rayDir = normalize(float3(uv, 1.0));

    ISDFModel sdfModel[2];

    sdfModel[0] = TargetSDFModle(float3(0.5, 0.6, 1.0));
    sdfModel[1] = LearnedSDFModle(sdf_params, float3(1.0, 0.6, 0.3));

    Ray ray = {rayOrigin, rayDir};
    RayMarcher rayMarcher = {ray};

    float3 color = float3(0.0f);
    for (uint i = 0; i < sdfModel.getCount(); i++)
    {
        color += rayMarcher.getModelColor(sdfModel[i]);
    }

    color = color / sdfModel.getCount();
    return float4(color, 1.0);
}
