// SPDX-License-Identifier: Apache-2.0

// Return a color value from the texture. Ideally the sample should be a
// point sampler as we are performing the filtering and mipmapping ourselves.
no_diff float4 getTexColor(SamplerState sampler, Texture2D texture, float2 uv)
{
    float4 color = texture.Sample(sampler, uv, uint2(0), 0);
    return color;
}

// Box filter downsampling function that takes a center UV coordinate and samples a 2x2 region.
[Differentiable]
float4 boxFilterDownsample(SamplerState sampler, Texture2D texture, float2 uv, float2 texelSize)
{
    // Calculate the four sample positions (2x2 grid).
    float2 topLeft = uv + float2(-0.5, 0.5) * texelSize;
    float2 topRight = uv + float2(0.5, 0.5) * texelSize;
    float2 bottomLeft = uv + float2(-0.5, -0.5) * texelSize;
    float2 bottomRight = uv + float2(0.5, -0.5) * texelSize;

    // Sample the texture at each position.
    float4 c00 = getTexColor(sampler, texture, topLeft);
    float4 c10 = getTexColor(sampler, texture, topRight);
    float4 c01 = getTexColor(sampler, texture, bottomLeft);
    float4 c11 = getTexColor(sampler, texture, bottomRight);

    // Average the four samples (box filter).
    return (c00 + c10 + c01 + c11) * 0.25;
}

// Calculate the L2 loss between two float4 values.
// Returns a scalar value representing the mean squared error.
[Differentiable]
float calculateL2Loss(float4 a, float4 b)
{
    float4 diff = a - b;
    // Square each component and sum them
    float sumSquaredDiff = diff.x * diff.x + 
                          diff.y * diff.y + 
                          diff.z * diff.z + 
                          diff.w * diff.w;
    // Take the mean (divide by 4 since float4 has 4 components)
    return sumSquaredDiff / 4.0;
}

// Calculate Phong lighting given material properties and normal.
[Differentiable]
float4 calculatePhongLighting(float4 materialColor, float4 normalColor)
{
    // Convert normal from [0,1] to [-1,1] range and normalize.
    float3 N = normalize(normalColor.xyz * 2.0 - 1.0);

    // Fixed light and view directions for this example.
    float3 L = normalize(float3(0.0, 0.0, 1.0));  // Light direction.
    float3 V = normalize(float3(0.0, 0.0, 1.0));  // View direction.
    float3 R = reflect(-L, N);                    // Reflected light direction.

    // Material properties.
    float3 ambient = float3(0.0, 0.0, 0.0);       // No ambient color needed.
    float3 diffuse = materialColor.xyz;           // Diffuse color.
    float3 specular = float3(1.0, 1.0, 1.0);      // Specular color (white).
    float shininess = 2.0;                        // Specular shininess.

    // Light properties.
    float3 lightColor = float3(1.0, 1.0, 1.0);    // White light.
    float lightIntensity = 1.0;

    // Calculate lighting components.
    float NdotL = max(dot(N, L), 0.0);            // Diffuse factor.
    float RdotV = max(dot(R, V), 0.0);            // Specular factor.

    // Combine components.
    float3 ambientTerm = ambient;
    float3 diffuseTerm = diffuse * NdotL;
    float3 specularTerm = specular * pow(RdotV, shininess);

    // Final color.
    float3 color = (ambientTerm + diffuseTerm + specularTerm) * lightColor * lightIntensity;

    return float4(color, materialColor.w);
}


// Below are functions for the 5 modes described by main.py

// Mode 1: Evaluate pixel color using full res textures with a simple BRDF.
[Differentiable]
float4 genFullResPixelColor(SamplerState sampler, Texture2D materialTex, Texture2D normalTex, float2 fragCoord, float2 windowSize)
{
    // Convert fragCoord to uv texture coord.
    float2 uv = float2(fragCoord.x / windowSize.x, fragCoord.y / windowSize.y);

    // Get material (diffuse) texture color, and normal.
    float4 materialColor = getTexColor(sampler, materialTex, uv);
    float4 normalColor = getTexColor(sampler, normalTex, uv);

    // Calculate final color using a simple BRDF, in this case phong.
    float4 phong = calculatePhongLighting(materialColor, normalColor);

    return phong;
}

// Mode 2: Evaluate 4 adjacent pixel colors using the same approach as mode 1,
// and give the average result. This is simlar to evaluating the BRDF using
// lower res mipmap inputs, but should give a result closer to mode 1.
[Differentiable]
float4 genLowResPixelColorFromAvg(SamplerState sampler, Texture2D materialTex, Texture2D normalTex, float2 fragCoord, float2 windowSize)
{
    float4 outColor = float4(0.0, 0.0, 0.0, 0.0);

    // Calculate the four sample positions (2x2 grid).
    float2 samplePos[4] = {
        fragCoord + float2(-1, 1), // Top left.
        fragCoord + float2(1, 1), // Top right.
        fragCoord + float2(-1, -1), // Bottom left.
        fragCoord + float2(1, -1), // Bottom right.
    };

    // Use method 1 with the 4 sample positions, and take the average.
    for (int i = 0; i < 4; i++) {
        outColor += genFullResPixelColor(sampler, materialTex, normalTex, samplePos[i], windowSize);
    }

    outColor /= 4.0;
    return outColor;
}

// Mode 3: Evaluate the BRDF using lower res inputs, ie, lower res mipmap
// levels from both the material texture and normal map to calculate
// final pixel colors. This should look less correct than mode 1 or 2.
[Differentiable]
float4 genLowResPixelColorFromMipmap(SamplerState sampler, Texture2D materialTex, Texture2D normalTex, float2 fragCoord, float2 windowSize)
{
    float2 uv = float2(fragCoord.x / windowSize.x, fragCoord.y / windowSize.y);

    // Calculate texel size for proper sampling. This can be modified to
    // generate different mipmap levels.
    float2 texelSize = float2(4.0 / windowSize.x, 4.0 / windowSize.y);

    // Get the downsampled material color and normal using a box filter.
    float4 lowResColor = boxFilterDownsample(sampler, materialTex, uv, texelSize);
    float4 lowResNormal = boxFilterDownsample(sampler, normalTex, uv, texelSize);

    // Calculate final color using the same BRDF as the full res function, this
    // time using the lower res inputs.
    float4 phong = calculatePhongLighting(lowResColor, lowResNormal);

    return phong;
}

// Mode 4: Show the difference between mode 1 and mode 3, ie, the L2 loss
// function which shows how much the color value needs to change to reach the
// correct value.
[Differentiable]
float4 genLowResPixelColorLoss(SamplerState sampler, Texture2D materialTex, Texture2D normalTex, float2 fragCoord, float2 windowSize)
{
    // Get colors from both methods.
    float4 fullColor = genFullResPixelColor(sampler, materialTex, normalTex, fragCoord, windowSize);
    float4 mipmapColor = genLowResPixelColorFromMipmap(sampler, materialTex, normalTex, fragCoord, windowSize);

    // Calculate L2 loss.
    float loss = calculateL2Loss(fullColor, mipmapColor);

    // Return the loss as a grayscale color (all channels same value)
    // The brighter the color, the larger the difference.
    return float4(loss, loss, loss, 1.0);
}

// Render the resulting pixel color using modes 1 to 4, as described in
// main.py. Mode 5 is handled separately.
float4 renderMipmapMode(SamplerState sampler, Texture2D materialTex, Texture2D normalTex, float2 windowSize, int mode, int2 pixelCoord)
{
    // Adjust the incoming pixel coordinate to have 0,0 at the bottom left
    // which also places world space -1,-1 at the bottom left for ease of use.
    float2 fragCoord = float2(pixelCoord);
    fragCoord.y = windowSize.y - fragCoord.y;

    // Default background color is black.
    float4 outColor = float4(0.0, 0.0, 0.0, 0.0);

    if (mode == 1) {
        outColor = genFullResPixelColor(sampler, materialTex, normalTex, fragCoord, windowSize);
    } else if (mode == 2) {
        outColor = genLowResPixelColorFromAvg(sampler, materialTex, normalTex, fragCoord, windowSize);
    } else if (mode == 3) {
        outColor = genLowResPixelColorFromMipmap(sampler, materialTex, normalTex, fragCoord, windowSize);
    } else if (mode == 4) {
        outColor = genLowResPixelColorLoss(sampler, materialTex, normalTex, fragCoord, windowSize);
    }

    return outColor;
}


// Give the difference between the expected and current loss values for a
// given sample point.
[Differentiable]
float targetLoss(SamplerState sampler, Texture2D materialTex, Texture2D normalTex, float2 windowSize, float2 samplePoint, float currentlossVal)
{
    float2 fragCoord = float2(samplePoint);
    fragCoord.y = windowSize.y - fragCoord.y;

    // Get colors from both modes (1) and (3).
    float4 fullColor = genFullResPixelColor(sampler, materialTex, normalTex, fragCoord, windowSize);
    float4 mipmapColor = genLowResPixelColorFromMipmap(sampler, materialTex, normalTex, fragCoord, windowSize);

    // Calculate L2 loss for the given sample point.
    float loss = calculateL2Loss(fullColor, mipmapColor);
    return currentlossVal - loss;
}


// Give the difference between the expected and current loss values for a given
// sample point.
[Differentiable]
float forward(SamplerState sampler, Texture2D materialTex, Texture2D normalTex, float2 windowSize, float2 samplePoint, float lossVal)
{
    float diff = targetLoss(sampler, materialTex, normalTex, windowSize, samplePoint, lossVal);
    return diff * diff;
}

// This function is used to visually show the results of learning from mode 5.
float4 renderLearnedLoss(SamplerState sampler, Texture2D materialTex, Texture2D normalTex, Texture2D lossTex, float2 windowSize, int2 pixelCoord)
{
    // Adjust the incoming pixel coordinate to have 0,0 at the bottom left
    // which also places world space -1,-1 at the bottom left for ease of use.
    float2 fragCoord = float2(pixelCoord);
    fragCoord.y = windowSize.y - fragCoord.y;

    // The loss texture has a different memory layout so we use the transposed
    // pixelCoord instead of the fragCoord.
    float2 uv = float2(float(pixelCoord.y) / windowSize.y, float(pixelCoord.x) / windowSize.x);

    // Show the leanred loss value. This texture has a single red channel,
    // instead we render it as white.
    float4 learnedLoss = getTexColor(sampler, lossTex, uv);
    float4 outColor = float4(learnedLoss.x, learnedLoss.x, learnedLoss.x, 1.0);

    return outColor;
}
