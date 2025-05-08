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
float3 disneyBrdf(no_diff float3 albedo, float3 normal, float roughness, no_diff float3 lightDir, no_diff float3 viewDir)
{
    float3 N = normalize(normal);
    float3 L = normalize(lightDir);              // Light direction.
    float3 V = normalize(viewDir);               // View direction.

    float metallic = 0.0;
    float specular = 1.0;

    float lightIntensity = 5.0;

    float3 brdf = DisneyBRDF(albedo, L, V, N, roughness, metallic, specular);
    return brdf * lightIntensity * max(0, dot(N, L));
}


// Phong BRDF.
[Differentiable]
float3 phong(no_diff float3 albedo, float3 normal, no_diff float3 lightDir, no_diff float3 viewDir)
{
    float3 N = normalize(normal);
    float3 L = normalize(lightDir);              // Light direction.
    float3 V = normalize(viewDir);               // View direction.
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
float3 brdf(no_diff float3 albedo, float3 normal, float roughness, no_diff float3 lightDir, no_diff float3 viewDir)
{
    //return phong(albedo, normal, lightDir, viewDir);
    return disneyBrdf(albedo, normal, roughness, lightDir, viewDir);
}

// Helper function to turn pixel coordinates into normalized UVs.
float2 pixelToUV(int2 pixel, int2 resolution)
{
    return (float2(pixel) + 0.5f) / float2(resolution);
}



// Called from main.py to copy loaded 4 channel texture into 3 channel albedo tensor.
float3 toAlbedoMap(float4 color)
{
    return color.xyz;
}

// Called from main.py to copy loaded 4 channel texture into 3 channel normal tensor.
float3 toNormalMap(float4 color)
{
    // Convert normal from [0,1] to [-1,1] range.
    return pow(color.xyz,1/2.2)*2-1;
}

// Called from main.py to copy loaded 4 channel texture into 1 channel roughnss tensor.
float toRoughnessMap(float4 color)
{
    // All components should have the same value.
    return color.x;
}


// Calculate the L2 loss between two float3 values.
// Returns a scalar value representing the mean squared error.
[Differentiable]
float calculateL2Loss(no_diff float3 target, float3 input)
{
    float3 diff = target - input;
    // Square each component and sum them.
    float sumSquaredDiff = diff.x * diff.x + 
                           diff.y * diff.y + 
                           diff.z * diff.z;
    // Take the mean (divide by 3 since float3 has 3 components).
    return sumSquaredDiff / 3.0;
}



// Functions to facilitate downsampling of float and float3 tensors at given
// coords. Note that Slangpy transposes x and y.
float3 sampleTensorFloat3(Tensor<float3,2> tensor, int2 pixelPos)
{
    // Get the color value from the tensor at the given pixel position.
    return tensor.get( { pixelPos.y, pixelPos.x });
}

float3 sampleTensorFloat3(Tensor<float3,2> tensor, float2 uv)
{
    // Turn UV into clamped integer coordinate...
    int2 resolution = int2(tensor.shape[1], tensor.shape[0]);
    int2 texel = int2(uv * float2(resolution));
    texel = clamp(texel, int2(0), resolution - 1);
    // ...then sample from that coordinate
    return sampleTensorFloat3(tensor, texel);
}

float3 downSampleFloat3(Tensor<float3, 2> input, int2 pixelPos)
{
    float3 res = 0;
    res += sampleTensorFloat3(input, pixelPos * 2 + int2(0, 0));
    res += sampleTensorFloat3(input, pixelPos * 2 + int2(1, 0));
    res += sampleTensorFloat3(input, pixelPos * 2 + int2(0, 1));
    res += sampleTensorFloat3(input, pixelPos * 2 + int2(1, 1));
    return res * 0.25;
}

float sampleTensorFloat(Tensor<float,2> tensor, int2 pixelPos)
{
    // Get the color value from the tensor at the given pixel position.
    return tensor.get( { pixelPos.y, pixelPos.x });
}

float sampleTensorFloat(Tensor<float,2> tensor, float2 uv)
{
    // Turn UV into clamped integer coordinate...
    int2 resolution = int2(tensor.shape[1], tensor.shape[0]);
    int2 texel = int2(uv * float2(resolution));
    texel = clamp(texel, int2(0), resolution - 1);
    // ...then sample from that coordinate
    return sampleTensorFloat(tensor, texel);
}

float downSampleFloat(Tensor<float, 2> input, int2 pixelPos)
{
    float res = 0;
    res += sampleTensorFloat(input, pixelPos * 2 + int2(0, 0));
    res += sampleTensorFloat(input, pixelPos * 2 + int2(1, 0));
    res += sampleTensorFloat(input, pixelPos * 2 + int2(0, 1));
    res += sampleTensorFloat(input, pixelPos * 2 + int2(1, 1));
    return res * 0.25;
}


// Render full res BRDF from given inputs.
float3 renderFullRes(float3 albedo, float3 normal, float roughness, float3 lightDir, float3 viewDir)
{
    return brdf(albedo, normal, roughness, lightDir, viewDir);
}


// Base normal of 0,0,1 to begin traning the normal tensor.
float3 baseNormal()
{
    return float3(0, 0, 1);
}

// As above, but a base roughness of 1.
float baseRoughness()
{
    return 1.0f;
}


// Calculate the loss between a target color and calculated BRDF from the given
// inputs. This is used for training the normal and roughness values so these
// need to be differentiable. Other inputs are not trained.
[Differentiable]
float calculateLoss(no_diff float3 targetColor, no_diff float3 albedo, float3 normal, float roughness, no_diff float3 lightDir, no_diff float3 viewDir)
{
    float3 inputColor = brdf(albedo, normal, roughness, lightDir, viewDir);
    float loss = calculateL2Loss(targetColor, inputColor);
    return loss;
}


// Perform one step of gradient descent and clear the gradient to zero for the
// next iteration.
void gradientDescentFloat3(inout float3 value, inout float3 gradient, float learningRate)
{
    value -= gradient * learningRate;
    gradient = float3(0.0f);
}

void gradientDescentFloat(inout float value, inout float gradient, float learningRate)
{
    value -= gradient * learningRate;
    gradient = 0.0f;
}


// Adam optimization step. Performs a similar role as the gradient descent
// function but less likely to get stuck in local minima.
void adamFloat3(inout float3 value, inout float3 gradient, inout float3 mPrev, inout float3 vPrev, float learningRate)
{
    // Standard Adam default values.
    const float ADAM_ETA = 0.002;
    const float ADAM_BETA_1 = 0.9;
    const float ADAM_BETA_2 = 0.999;
    const float ADAM_EPSILON = 1e-8;

    // Adam optimization.
    float3 gradient2 = gradient * gradient;

    float3 m = ADAM_BETA_1 * mPrev + (1.0 - ADAM_BETA_1) * gradient;
    float3 v = ADAM_BETA_2 * vPrev + (1.0 - ADAM_BETA_2) * gradient2;

    mPrev = m;
    vPrev = v;

    float3 mHat = m / (1.0f - ADAM_BETA_1);
    float3 vHat = v / (1.0f - ADAM_BETA_2);

    float3 update = (ADAM_ETA / (sqrt(vHat) + ADAM_EPSILON)) * mHat;

    // Subtract the optimized result from the trained normal and reset the gradient.
    value -= update;

    gradient = float3(0.0f);
}

void adamFloat(inout float value, inout float gradient, inout float mPrev, inout float vPrev, float learningRate)
{
    // Standard Adam default values.
    const float ADAM_ETA = 0.002;
    const float ADAM_BETA_1 = 0.9;
    const float ADAM_BETA_2 = 0.999;
    const float ADAM_EPSILON = 1e-8;

    // Adam optimization.
    float gradient2 = gradient * gradient;

    float m = ADAM_BETA_1 * mPrev + (1.0 - ADAM_BETA_1) * gradient;
    float v = ADAM_BETA_2 * vPrev + (1.0 - ADAM_BETA_2) * gradient2;

    mPrev = m;
    vPrev = v;

    float mHat = m / (1.0f - ADAM_BETA_1);
    float vHat = v / (1.0f - ADAM_BETA_2);

    float update = (ADAM_ETA / (sqrt(vHat) + ADAM_EPSILON)) * mHat;

    // Subtract the optimized result from the trained normal and reset the gradient.
    value -= update;

    gradient = 0.0f;
}



// Render calculated L2 loss between a given avgColor and BRDF calculated from
// given inputs.
float3 renderLoss(float3 avgColor, float3 albedo, float3 normal, float roughness, float3 lightDir, float3 viewDir)
{
    // BRDF from downsampled inputs, avgColor is the downsampled output from full res inputs.
    float loss = calculateLoss(avgColor, albedo, normal, roughness, lightDir, viewDir);
    return float3(loss, loss, loss);
}


// Resample tensor and process into presentable colors.
float4 showTensorFloat3(Tensor<float3, 2> tensor, float2 uv, bool gammaCorrect)
{
    float3 color = sampleTensorFloat3(tensor, uv);
    color = abs(color);
    if (gammaCorrect)
        color = pow(color, 1.0f/2.2f);
    return float4(color, 1.0);
}


// Calculate the loss between the downsampled result, and rendering the BRDF
// from low res inputs.
float3 lossLowres(float3 downsampled, float3 lowres)
{
    float loss = calculateL2Loss(downsampled, lowres);
    return float3(loss, loss, loss);
}

// Get the difference between the trained loss, and the loss from the above
// function. Green pixels indicate a better result and red pixels indicate a
// worse result.
float3 lossDiff(float3 trainedLoss, float3 lowresLoss)
{
    float3 color = float3(0.0);
    if (trainedLoss.x < lowresLoss.x) {
        color.y = trainedLoss.x;
    } else if (trainedLoss.x > lowresLoss.x) {
        color.x = lowresLoss.x;
    }
    return color;
}


// Main rendering function to render different tensors showing the training
// progress of the normal and roughness.
float4 showTrainingProgress(Tensor<float3, 2> avgColor, Tensor<float3, 2> loss, Tensor<float3, 2> initialNormal, Tensor<float3, 2> optimizedNormal, Tensor<float, 2> optimizedRoughness, float2 uv)
{
    uv.x *= 4.0f;
    int column = int(uv.x);
    uv.x -= column;
    int row = int(uv.y);

    float3 color;
    if (row == 0) {
        if (column == 0) {
            // Render BRDF, downsampled after being calculated from full res inputs.
            color = pow(sampleTensorFloat3(avgColor, uv), 1.0f/2.2f);
        } else if (column == 1) {
            // Render difference in the loss between the above, and the BRDF calculated from downsampled inputs.
            color = sampleTensorFloat3(loss, uv);
        } else if (column == 2) {
            // Render the trained normal map.
            color = normalize(sampleTensorFloat3(optimizedNormal, uv)) * 0.5f + 0.5f;
        } else if (column == 3) {
            // Render the trained roughness.
            color = float3(sampleTensorFloat(optimizedRoughness, uv));
        }
    }

    return float4(color, 1.0f);
}
