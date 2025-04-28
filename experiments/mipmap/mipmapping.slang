// SPDX-License-Identifier: Apache-2.0
import slangpy;


// TODO: Trying out some different BRDF functions.
// TODO: Looks like they only seem to show differences in the specular components.

#define PI 3.14159265358979323846f

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
float3 BRDF(float3 albedo, float3 normal, float3 lightDir, float roughness = 0.4, float metallic = 0.4, float specular = 1.0)
{
    float3 viewDir = float3(0.0, 0.0, 1.0);

    float3 L = normalize(lightDir);
    float3 V = normalize(viewDir);
    float3 N = normalize(normal);

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

    float albedoIntensity = 3;

    float3 Cdlin = albedo * albedoIntensity;
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





// see http://en.wikipedia.org/wiki/Schlick%27s_approximation
float fresnelSchlick(float r0, float vDotH)
{
    return r0 + (1.0 - r0) * pow(1.0 - vDotH, 5.0);
}

// see http://graphicrants.blogspot.de/2013/08/specular-brdf-reference.html
// see http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
float geometricShadowingSchlickBeckmann(float nDotV, float k)
{
    return nDotV / (nDotV * (1.0 - k) + k);
}

// see http://graphicrants.blogspot.de/2013/08/specular-brdf-reference.html
// see http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
float geometricShadowingSmith(float nDotL, float nDotV, float k)
{
    return geometricShadowingSchlickBeckmann(nDotL, k) * geometricShadowingSchlickBeckmann(nDotV, k);
}

// Cook-Torrence BRDF.
float3 cookTorrence(float3 albedo, float3 normal, float3 lightDir, float roughness = 0.1)
{
    float3 viewDir = float3(0, 0, 1);


    // See https://github.com/McNopper/OpenGL/blob/master/Example32/shader/brdf.frag.glsl
    // basis, N, V, K
    // basis: tangent, bitangent, normal matrix, normal: normal vector, eye: eyeDir vector, k: roughness float
    float3 N = normalize(normal);
    float3 V = normalize(viewDir);
    float3 L = normalize(lightDir);
    float3 H = normalize(reflect(-L, N));

    float nDotL = dot(N, L);
    float nDotV = dot(N, V);
    float nDotH = dot(N, H);

    // Lighted and visible?
    if (nDotL > 0 && nDotV > 0) {
        float vDotH = max(0, dot(V, H));

        // Fresnel.
        float refCoefficient = 1.0;
        float F = fresnelSchlick(refCoefficient, vDotH);

        // Geometric Shadowing.
        float G = geometricShadowingSmith(nDotL, nDotV, roughness);

        // Lo = BRDF * L * NdotL / PDF
        //
        // BRDF is D * F * G / (4 * NdotL * NdotV).
        // PDF is D * NdotH / (4 * VdotH).
        // D and 4 * NdotL are canceled out, which results in this formula: Lo = color * L * F * G * VdotH / (NdotV * NdotH)		
        float colorFactor = F * G * vDotH / (nDotV * nDotH);

        // Note: Needed for robustness. With specific parameters, a NaN can be the result.
        if (isnan(colorFactor)) {
            return float3(0);
        }

        return albedo * max(0, colorFactor);
    }

    return float3(0);
}



// Phong BRDF.
[Differentiable]
float3 phong(no_diff float3 albedo, float3 normal, no_diff float3 lightDir)
{
    // Fixed light and view directions for this example.
    float3 N = normalize(normal);
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
    // TODO: Using phong for now for simplicity.
    return phong(albedo, normal, lightDir);
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

no_diff float3 sampleTensor(Tensor<float3,2> tensor, no_diff int2 pixelPos)
{
    // Get the color value from the tensor at the given pixel position.
    return tensor.get( { pixelPos.y, pixelPos.x });
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

/*
// TODO: Decide if we want to keep these.
// Mode 2: Calculate BRDF from full res inputs, then take the average.
float3 renderDownsampledOutputs(Tensor<float3,2> albedoMap, Tensor<float3,2> normalMap, float3 lightDir, int2 pixelPos, int l)
{
    float3 result = 0;
    result += calculatePhongLighting(sampleTensor(albedoMap, pixelPos * l + int2(0, 0)), sampleTensor(normalMap, pixelPos * l + int2(0, 0)), lightDir);
    result += calculatePhongLighting(sampleTensor(albedoMap, pixelPos * l + int2(1, 0)), sampleTensor(normalMap, pixelPos * l + int2(1, 0)), lightDir);
    result += calculatePhongLighting(sampleTensor(albedoMap, pixelPos * l + int2(0, 1)), sampleTensor(normalMap, pixelPos * l + int2(0, 1)), lightDir);
    result += calculatePhongLighting(sampleTensor(albedoMap, pixelPos * l + int2(1, 1)), sampleTensor(normalMap, pixelPos * l + int2(1, 1)), lightDir);
    return result * 0.25;
}

// Mode 3: Take inputs from downsampled inputs, then calculate BRDF.
float3 renderDownsampledInputs(Tensor<float3, 2> albedoMap, Tensor<float3, 2> normalMap, float3 lightDir, int2 pixelPos, int l)
{
    float3 albedo = 0;
    albedo += sampleTensor(albedoMap, pixelPos * l + int2(0, 0));
    albedo += sampleTensor(albedoMap, pixelPos * l + int2(1, 0));
    albedo += sampleTensor(albedoMap, pixelPos * l + int2(0, 1));
    albedo += sampleTensor(albedoMap, pixelPos * l + int2(1, 1));
    albedo *= 0.25;

    float3 normal = 0;
    normal += sampleTensor(normalMap, pixelPos * l + int2(0, 0));
    normal += sampleTensor(normalMap, pixelPos * l + int2(1, 0));
    normal += sampleTensor(normalMap, pixelPos * l + int2(0, 1));
    normal += sampleTensor(normalMap, pixelPos * l + int2(1, 1));
    normal *= 0.25;

    return calculatePhongLighting(albedo, normal, lightDir);
}
*/

// Functions used for training in (5).
[Differentiable]
float calculateLoss(no_diff float3 targetColor, no_diff float3 albedoInput, float3 normalInput, no_diff float3 lightDir)
{
    float3 inputColor = brdf(albedoInput, normalInput, lightDir);
    float loss = calculateL2Loss(targetColor, inputColor);
    return loss;
}

// TODO: This version doesn't seem to work, I needed to explicitly declare the tensors and a samplePos, but I'm not sure why.
[Differentiable]
float forward(no_diff float3 targetColor, no_diff float3 albedoInput, float3 normalInput, no_diff float3 lightDir)
{
    // TODO: Is this correct? I think we want target loss to be 0 to match (2).
    float targetLoss = 0;
    float currentLoss = calculateLoss(targetColor, albedoInput, normalInput, lightDir);
    float diff = targetLoss - currentLoss;

    return diff * diff;
}

/*
[Differentiable]
float forward(Tensor<float3, 2> targetColorTensor, Tensor<float3, 2> albedoInputTensor, float3 normalInput, no_diff float3 lightDir, no_diff float2 samplePos)
{
    int2 pixelPos = int2(int(samplePos.x), int(samplePos.y));
    float3 targetColor = sampleTensor(targetColorTensor, pixelPos);
    float3 albedoInput = sampleTensor(albedoInputTensor, pixelPos);

    float targetLoss = 0;
    float currentLoss = calculateLoss(targetColor, albedoInput, normalInput, lightDir);
    float diff = targetLoss - currentLoss;

    return diff * diff;
}
*/


// Mode 4: Render calculated L2 loss between (2) and (3).
float3 renderLoss(float3 avgColor, float3 albedo, float3 normal, float3 lightDir)
{
    // BRDF from downsampled inputs, avgColor is the downsampled output from full res inputs.
    float loss = calculateLoss(avgColor, albedo, normal, lightDir);
    return float3(loss, loss, loss);
}


// Write pixel color outputs from the tensor input.
void showTensorFloat3(Tensor<float3, 2> tensor, RWTexture2D<float4> output, int2 pixelCoord, int2 offset)
{
    // Convert the tensor value to a float4 color.
    uint width;
    uint height;
    output.GetDimensions(width,height);

    int srcY = (tensor.shape[0] * pixelCoord.y) / height;
    int srcX = (tensor.shape[1] * pixelCoord.x) / width;
    float3 color = tensor.get( { srcY, srcX });
    output[pixelCoord] = float4(abs(color), 1.0);
}


float3 testRender(Tensor<float3, 2> albedoInputTensor, float3 normalInput, float3 lightDir, float2 samplePos)
{
    int2 pixelPos = int2(int(samplePos.x), int(samplePos.y));
    float3 albedoInput = sampleTensor(albedoInputTensor, pixelPos);
    return brdf(albedoInput, normalInput, lightDir);
}
