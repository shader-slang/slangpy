// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import slangpy;


// Disney BRDF functions.

#define PI 3.14159265358979323846f

// Simple mathematical helper functions for clamp and square.
[Differentiable]
float clamp(float x, float x_min, float x_max)
{
    return max(min(x, x_max), x_min);
}

[Differentiable]
float sqr(float x)
{
    return x * x;
}

// Schlick Fresnel approximation for specular reflections.
[Differentiable]
float SchlickFresnel(float u)
{
    float m = clamp(1 - u, 0, 1);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

[Differentiable]
float GTR1(float NdotH, float a)
{
    if (a >= 1)
        return 1 / PI;
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return (a2 - 1) / (PI * log(a2) * t);
}

[Differentiable]
float GTR2(float NdotH, float a)
{
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return a2 / (PI * t * t);
}

[Differentiable]
float smithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

// Compute BRDF lighting for a given light drection (L), view direction (V) and
// normal vector (N), using given BRDF lighting properties.
[Differentiable]
float3 DisneyBRDF(float3 baseColor, float3 L, float3 V, float3 N, float roughness, float metallic, float specular)
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

    float3 Cdlin = baseColor;
    float Cdlum = .3f * Cdlin[0] + .6f * Cdlin[1] + .1f * Cdlin[2]; // luminance approx.

    float3 Ctint = Cdlum > 0 ? Cdlin / Cdlum : float3(1); // normalize lum. to isolate hue+sat
    float3 Cspec0 = lerp(specular * float3(.08f) *
        lerp(float3(1.f), Ctint, specularTint), Cdlin, metallic);
    float3 Csheen = lerp(float3(1), Ctint, sheenTint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and lerp in diffuse retro-reflection based on roughness
    float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fd90 = 0.5f + 2 * LdotH * LdotH * roughness;
    float Fd = lerp(1.f, Fd90, FL) * lerp(1.f, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LdotH * LdotH * roughness;
    float Fss = lerp(1.f, Fss90, FL) * lerp(1.f, Fss90, FV);
    float ss = 1.25f * (Fss * (1 / (NdotL + NdotV) - .5f) + .5f);

    // specular
    float a = sqr(roughness);
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
        (1 - metallic) + Gs * Fs * Ds + .25f * clearcoat * Gr * Fr * Dr;
}

[Differentiable]
float3 disneyBrdf(no_diff float3 albedo, float3 normal, no_diff float3 lightDir)
{
    // Fixed light and view directions for this example.
    float3 N = normalize(normal);
    float3 L = normalize(lightDir);              // Light direction.
    float3 V = normalize(float3(0.0, 0.0, 1.0)); // View direction.

    float roughness = 0.4;
    float metallic = 0.0;
    float specular = 1.0;

    float lightIntensity = 5.0;

    float3 brdf = DisneyBRDF(albedo, L, V, N, roughness, metallic, specular);
    return brdf * lightIntensity * max(0, dot(N, L));
}


// Phong BRDF.
[Differentiable]
float3 phong(no_diff float3 albedo, float3 normal, no_diff float3 lightDir)
{
    // Fixed light and view directions for this example.
    //float3 N = normalize(normal);
    float3 N = normal;
    float3 L = normalize(lightDir);              // Light direction.
    float3 V = normalize(float3(0.0, 0.0, 1.0)); // View direction.
    float3 R = normalize(reflect(-L, N));        // Reflected light direction.

    // Material properties.
    float3 ambient = float3(0.0);                // No ambient color needed.
    float3 diffuse = albedo;                     // Diffuse color.
    float3 specular = float3(1.0);               // Specular color.
    float shininess = 128;                       // Specular shininess.

    // Light properties.
    float3 lightColor = float3(1.0, 1.0, 1.0);   // White light.
    float lightIntensity = 3;

    // Calculate lighting components.
    float NdotL = max(dot(N, L), 0.0); // Diffuse factor.
    float RdotV = max(dot(R, V), 0.0); // Specular factor.

    // Combine components.
    float3 ambientTerm = ambient;
    float3 diffuseTerm = diffuse * NdotL;
    float3 specularTerm = specular * pow(RdotV, shininess);

    // Final color.
    float3 outColor = (ambientTerm + diffuseTerm + specularTerm) * lightColor * lightIntensity;
    return outColor;
}


// Wrapper to more easily change which BRDF to use.
[Differentiable]
float3 brdf(no_diff float3 albedo, float3 normal, no_diff float3 lightDir)
{
    //return phong(albedo, normal, lightDir);
    return disneyBrdf(albedo, normal, lightDir);
}

// Helper function to turn pixel coordinates into normalized UVs
float2 pixelToUV(int2 pixel, int2 resolution)
{
    return (float2(pixel) + 0.5f) / float2(resolution);
}



// Called from main.py to copy loaded 4 channel texture into 3 channel albedo tensor
float3 toAlbedoMap(float4 color)
{
    return color.xyz;
}

// Called from main.py to copy loaded 4 channel texture into 3 channel normal tensor
float3 toNormalMap(float4 color)
{
    // Convert normal from [0,1] to [-1,1] range.
    return pow(color.xyz,1/2.2)*2-1;
}

// Calculate the L2 loss between two float4 values.
// Returns a scalar value representing the mean squared error.
[Differentiable]
float calculateL2Loss(no_diff float3 target, float3 input)
{
    float3 diff = target - input;
    // Square each component and sum them.
    float sumSquaredDiff = diff.x * diff.x + 
                           diff.y * diff.y + 
                           diff.z * diff.z;
    // Take the mean (divide by 3 since float4 has 3 components).
    return sumSquaredDiff / 3.0;
}

no_diff float3 sampleTensor(Tensor<float3,2> tensor, int2 pixelPos)
{
    // Get the color value from the tensor at the given pixel position.
    return tensor.get( { pixelPos.y, pixelPos.x });
}

no_diff float3 sampleTensor(Tensor<float3,2> tensor, float2 uv)
{
    // Turn UV into clamped integer coordinate...
    int2 resolution = int2(tensor.shape[1], tensor.shape[0]);
    int2 texel = int2(uv * float2(resolution));
    texel = clamp(texel, int2(0), resolution - 1);
    // ...then sample from that coordinate
    return sampleTensor(tensor, texel);
}

float3 downSample(Tensor<float3, 2> input, int2 pixelPos)
{
    float3 res = 0;
    res += sampleTensor(input, pixelPos * 2 + int2(0, 0));
    res += sampleTensor(input, pixelPos * 2 + int2(1, 0));
    res += sampleTensor(input, pixelPos * 2 + int2(0, 1));
    res += sampleTensor(input, pixelPos * 2 + int2(1, 1));
    return res * 0.25;
}


float3 renderFullRes(float3 albedo, float3 normal, float3 lightDir)
{
    return brdf(albedo, normal, lightDir);
}


// Functions used for training in (5).
float3 baseNormal()
{
    return normalize(float3(1, 1, 1));
}

[Differentiable]
float calculateLoss(no_diff float3 targetColor, no_diff float3 albedoInput, float3 normalInput, no_diff float3 lightDir)
{
    float3 inputColor = brdf(albedoInput, normalInput, lightDir);
    float loss = calculateL2Loss(targetColor, inputColor);
    return loss;
}

[Differentiable]
float forward(no_diff float3 targetColor, no_diff float3 albedoInput, float3 normalInput, no_diff float3 lightDir)
{
    // TODO: Is this correct? I think we want target loss to be 0 to match (2).
    float targetLoss = 0;
    float currentLoss = calculateLoss(targetColor, albedoInput, normalInput, lightDir);
    float diff = targetLoss - currentLoss;

    return diff * diff;
}

// Perform one step of gradient descent and clear the gradient to zero for the next iteration
void gradientDescent(inout float3 value, inout float3 gradient, float learningRate)
{
    value -= gradient * learningRate;
    gradient = float3(0.0f);
}

// Mode 4: Render calculated L2 loss between (2) and (3).
float3 renderLoss(float3 avgColor, float3 albedo, float3 normal, float3 lightDir)
{
    // BRDF from downsampled inputs, avgColor is the downsampled output from full res inputs.
    float loss = calculateLoss(avgColor, albedo, normal, lightDir);
    return float3(loss, loss, loss);
}


// Resample tensor and process into presentable colors.
float4 showTensorFloat3(Tensor<float3, 2> tensor, float2 uv)
{
    float3 color = sampleTensor(tensor, uv);
    return float4(abs(color), 1.0);
}
