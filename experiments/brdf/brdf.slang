// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import "slangpy";


// Simple Camera struct to convert between screen space and world space
// coordinates, so that positions can be specified in world space.
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

// Simple struct to store the BRDF lighting properties.
struct Properties
{
    float3 baseColor;
    float roughness;
    float metallic;
    float specular;
};


#define PI 3.14159265358979323846f

// Below are fairly standard BRDF lighting functions.

// Simple mathematical helper functions for clamp and square.
float clamp(float x, float x_min, float x_max)
{
    return max(min(x, x_max), x_min);
}

float sqr(float x)
{
    return x * x;
}

// Schlick Fresnel approximation for specular reflections.
float SchlickFresnel(float u)
{
    float m = clamp(1 - u, 0, 1);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

float GTR1(float NdotH, float a)
{
    if (a >= 1)
        return 1 / PI;
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return (a2 - 1) / (PI * log(a2) * t);
}

float GTR2(float NdotH, float a)
{
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return a2 / (PI * t * t);
}

float smithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

// Compute BRDF lighting for a given light drection (L), view direction (V) and
// normal vector (N), using given BRDF lighting properties.
float3 BRDF(float3 L, float3 V, float3 N, Properties properties)
{
    const float subsurface = 0;
    const float specularTint = 0;
    const float anisotropic = 0;
    const float sheen = 0;
    const float sheenTint = .5;
    const float clearcoat = 0;
    const float clearcoatGloss = 1;

    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if (NdotL < 0 || NdotV < 0)
        return float3(0);

    float3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);

    float3 Cdlin = properties.baseColor;
    float Cdlum = .3f * Cdlin[0] + .6f * Cdlin[1] + .1f * Cdlin[2]; // luminance approx.

    float3 Ctint = Cdlum > 0 ? Cdlin / Cdlum : float3(1); // normalize lum. to isolate hue+sat
    float3 Cspec0 = lerp(properties.specular * float3(.08f) *
        lerp(float3(1.f), Ctint, specularTint), Cdlin, properties.metallic);
    float3 Csheen = lerp(float3(1), Ctint, sheenTint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and lerp in diffuse retro-reflection based on roughness
    float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fd90 = 0.5f + 2 * LdotH * LdotH * properties.roughness;
    float Fd = lerp(1.f, Fd90, FL) * lerp(1.f, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LdotH * LdotH * properties.roughness;
    float Fss = lerp(1.f, Fss90, FL) * lerp(1.f, Fss90, FV);
    float ss = 1.25f * (Fss * (1 / (NdotL + NdotV) - .5f) + .5f);

    // specular
    float a = sqr(properties.roughness);
    float Ds = GTR2(NdotH, a);
    float FH = SchlickFresnel(LdotH);
    float3 Fs = lerp(Cspec0, float3(1), FH);
    float Gs;
    Gs = smithG_GGX(NdotL, a);
    Gs *= smithG_GGX(NdotV, a);

    // sheen
    float3 Fsheen = FH * sheen * Csheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = GTR1(NdotH, lerp(.1, .001, clearcoatGloss));
    float Fr = lerp(.04f, 1.f, FH);
    float Gr = smithG_GGX(NdotL, .25f) * smithG_GGX(NdotV, .25f);

    return ((1 / PI) * lerp(Fd, ss, subsurface) * Cdlin + Fsheen) *
        (1 - properties.metallic) + Gs * Fs * Ds + .25f * clearcoat * Gr * Fr * Dr;
}


// Below are ray-sphere intersection test equations.

// The quadratic equation is not numerically stable with limited precision
// floating point values, so we use a more numerically stable version to
// get the results from the quadratic equation.
float2 getDepthDistances(float4 eq)
{
    float discr = sqrt(eq.w);
    float d1, d2, z1, z2;

    if (eq.y >= 0) {
        d1 = 2.0 * eq.x;
        d2 = -eq.y - discr;
        if (d1 == 0.0)
            z1 = 0.0;
        else
            z1 = (-eq.y - discr) / d1;
        if (d2 == 0.0)
            z2 = 0.0;
        else
            z2 = (2.0 * eq.z) / d2;
    } else {
        d1 = -eq.y + discr;
        d2 = 2.0 * eq.x;
        if (d1 == 0.0)
            z1 = 0.0;
        else
            z1 = (2.0 * eq.z) / d1;
        if (d2 == 0.0)
            z2 = 0.0;
        else
            z2 = (-eq.y + discr) / d2;
    }

    return float2(z1, z2);
}

// Get the front and back z values using the sphere equation results and
// ray direction.
float2 getDepths(float4 eq, float3 dir)
{
    float2 z = getDepthDistances(eq);
    return float2((dir * z.x).z, (dir * z.y).z);
}

// Simple quadratic equation for the sphere.
float4 sphereEquation(float3 spherePos, float radius, float3 ray, float3 origin)
{
    float3 l = normalize(ray);
    float3 p = spherePos;
    float r2 = radius * radius;

    float a = dot(l, l);
    float b = 2.0 * dot(l, (origin - p));
    float c = dot((origin - p), (origin - p)) - r2;
    float discr = (b * b) - (4.0 * a * c);

    return float4(a, b, c, discr);
}

// Perform a ray-sphere intersection test for a given sphere and radius. This
// returns true if the result is inside the sphere, and computes front and
// back depth values for the intersection. Note that this test is in world
// space due to the world space camera position (0,0,1). This is somewhat
// hacky and would be better in eye space (0,0,0) but currently left in world
// space for somplicity.
bool sphereIntersection(float3 spherePos, float radius, float3 ray,
                        out float frontDepth, out float backDepth)
{
    // World space origin of the ray is at (0,0,1).
    float4 eq = sphereEquation(spherePos, radius, ray, float3(0.0, 0.0, 1.0));

    frontDepth = 0;
    backDepth = 0;

    // No intersections.
    if (eq.w < 0) {
        return false;
    }

    float2 depths = getDepths(eq, ray);

    frontDepth = min(-depths.x, -depths.y);
    backDepth = max(-depths.x, -depths.y);

    return true;
}


// BRDF function called from python, called for each pixel in the window. This
// tests if the pixelCoord is inside a sphere centered at world space (0,0) and
// then computes the lighting using BRDF for the given pixel. The z value is
// calculated using a ray-sphere intersection test to give a 3D sphere for
// better looking lighting and specular highlights.
float4 computeSphereBRDF(Camera camera, Properties properties, int2 pixelCoord)
{
    // Default black background color.
    float4 outColor = float4(0.0, 0.0, 0.0, 0.0);

    // Adjust pixel coordinate to have 0,0 at the bottom left for ease of use.
    float2 fragCoord = float2(pixelCoord);
    fragCoord.y = camera.frameDim.y - fragCoord.y;

    // Convert the screen space fragCoord to world space.
    float2 screenSample = fragCoord.xy + 0.5;
    float2 worldSample = camera.screenToWorld(screenSample);

    // Create a sphere centered at world space origin (0,0). The x and y world
    // space coords have been calculated previously. The z coord is calculated
    // with a ray-sphere intersection test, treating the pixel itself as a ray
    // going through the sphere, giving a front and back z value.
    float front = 0, back = 0;
    bool isInsideSphere =
        sphereIntersection(float3(0.0), 0.5, float3(worldSample.xy, 1.0), front, back);
    if (isInsideSphere) {
        // Light is looking directly at the sphere.
        float3 L = float3(0.0, 0.0, 1.0);

        // Since this is a world-space sphere centered at (0,0), the normal is
        // just the normalized world space position at the front. In this case
        // the view direction for a given pixel also happens to be the same as
        // the normal.
        float3 P = float3(worldSample, front);
        float3 V = normalize(P);
        float3 N = normalize(P);

        float3 lighting = BRDF(L, V, N, properties);
        outColor = float4(lighting, 1.0);
    }

    return outColor;
}
