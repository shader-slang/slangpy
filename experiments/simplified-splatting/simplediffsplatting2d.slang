// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/*
 * Simplified 2D Gaussian Splatting Example in Slang
 *
 * This example demonstrates the use of Slang's differentiable programming capabilities to implement
 * a 2D Gaussian splatting algorithm that can be trained within the browser using the Slang Playground.
 *
 * This algorithm represents a simplified version of the 3D Gaussian Splatting algorithm detailed in
 * this paper (https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
 * This 2D demonstration does not have the 3D->2D projection step & assumes that the Gaussian blobs
 * are presented in order of depth (higher index = farther away). Further, this implementation does
 * not perform adaptive density control to add or remove blobs.
 *
 * See the `computeDerivativesMain()` kernel and the `splatBlobs()` function for the bulk of the key
 * pieces of the code.
 *
 * Key Slang features used in this example include the autodiff operator `bwd_diff(fn)`, the
 * `[Differentiable]` attribute, and custom derivatives for a few specific components via
 * the `[BackwardDerivative(fn)]` attribute.
 *
 * For a full 3D Gaussian Splatting implementation written in Slang, see this repository:
 * https://github.com/google/slang-gaussian-rasterization
 *
 */

import slangpy;

// ----- Constants and definitions --------
static const float ADAM_ETA = 0.002;
static const float ADAM_BETA_1 = 0.9;
static const float ADAM_BETA_2 = 0.999;
static const float ADAM_EPSILON = 1e-8;

static const uint SIMPLE_BLOBCOUNT = 200;

// Some types to hold state info on the 'blobs' buffer.
// This makes it easy to make sure we're not accidentally using the buffer
// in the wrong state.
//
// The actual data is in the 'blobs' object.
//

struct Blobs
{
    GradInOutTensor<float, 1> blobsBuffer;

    __subscript(uint idx) -> float
    {
        [Differentiable] get { return blobsBuffer[idx]; }
    }
};

/*
 * A utility function to premultiply the color by the alpha value.
 * This is a key part of the alpha blending routine used in the
 * Gaussian splatting algorithm.
 */
[Differentiable]
float4 preMult(float4 pixel)
{
    return float4(pixel.rgb * pixel.a, pixel.a);
}

/*
 * alphaBlend() implements the standard alpha blending algorithm.
 *
 * Takes the current pixel value 'pixel' & blends it with a
 * contribution 'gval' from a new Gaussian.
 */
[Differentiable]
float4 alphaBlend(float4 pixel, float4 gval)
{
    gval = preMult(gval);

    return float4(
        pixel.rgb + gval.rgb * pixel.a,
        pixel.a * (1 - gval.a));
}

[Differentiable]
float2x2 inverse(float2x2 mat)
{
    float2x2 output;

    float det = determinant(mat);
    output[0][0] = mat[1][1] / det;
    output[0][1] = -mat[0][1] / det;
    output[1][0] = -mat[1][0] / det;
    output[1][1] = mat[0][0] / det;

    return output;
}

struct Gaussian2D : IDifferentiable
{
    float2 center;
    float2x2 sigma;
    float3 color;
    float opacity;

    [Differentiable]
    static Gaussian2D load(Blobs blobs, uint idx)
    {
        uint total = Gaussian2D.count(blobs);
        Gaussian2D gaussian;
        gaussian.center = smoothstep(
            float2(0, 0),
            float2(1, 1),
            float2(blobs[total * 0 + idx], blobs[total * 1 + idx]));
        
        // Add a small padding value to avoid singularities or unstable Gaussians.
        gaussian.sigma[0][0] = smoothstep(0.f, 1.f, blobs[total * 2 + idx] * 0.8f) + 0.005f;
        gaussian.sigma[1][1] = smoothstep(0.f, 1.f, blobs[total * 3 + idx] * 0.8f) + 0.005f;

        float aniso = (smoothstep(0.f, 1.f, blobs[total * 4 + idx] * 0.6f) - 0.5f) * 1.65f;
        gaussian.sigma[0][1] = sqrt(gaussian.sigma[0][0] * gaussian.sigma[1][1]) * aniso;
        gaussian.sigma[1][0] = sqrt(gaussian.sigma[0][0] * gaussian.sigma[1][1]) * aniso;

        // Scale the sigma so the blobs aren't too large
        gaussian.sigma *= 0.005;

        gaussian.color = smoothstep(0, 1, float3(
            blobs[total * 5 + idx],
            blobs[total * 6 + idx],
            blobs[total * 7 + idx]) * 0.8f);

        gaussian.opacity = smoothstep(0, 1, blobs[total * 8 + idx] * 0.9f + 0.1f);
        return gaussian;
    }

    // Simple helper method to get the number of elements in the buffer
    static uint count(Blobs blobs)
    {
        return blobs.blobsBuffer.primal.shape[0] / 9;
    }

    /*
     * eval() calculates the color and weight of the Gaussian at a given UV coordinate.
     *
     * The weight of the Gaussian blob at location uv is determined from
     * the exponent of the 2-dimensional Gaussian distribution function, multiplied by the
     * blob's inherent alpha value. The RGB color is stored and returned separately.
     * For more information about calculating Gaussian distributions, see 
     * https://mathworld.wolfram.com/GaussianFunction.html
     */
    [Differentiable]
    float4 eval(float2 uv)
    {
        float2x2 invCov = inverse(sigma);
        float2 diff = uv - center;
        float power = -0.5f * ((diff.x * diff.x * invCov[0][0]) +
                               (diff.y * diff.y * invCov[1][1]) +
                               (diff.x * diff.y * invCov[0][1]) +
                               (diff.y * diff.x * invCov[1][0]));

        float weight = min(.99f, opacity * exp(power));
        return float4(color, weight);
    }
};

/* simpleSplatBlobs() is a naive implementation of the computation of color for a pixel.
 * It will iterate over all of the Gaussians for each pixel, to determine their contributions
 * to the pixel color, so this will become prohibitively slow with a very small number of 
 * blobs, but it reduces the number of steps involved in determining the pixel color.
 */
 [Differentiable]
 float4 simpleSplatBlobs(GradInOutTensor<float, 1> blobsBuffer, uint2 pixelCoord, int2 texSize)
 {
    Blobs blobs = {blobsBuffer};

    float4 result = {0.0, 0.0, 0.0, 1.0};
    float4 blobColor = {0.0, 0.0, 0.0, 0.0};

    for (uint i = 0; i < SIMPLE_BLOBCOUNT; i++)
    {
        Gaussian2D gaussian = Gaussian2D.load(blobs, i);
        blobColor = gaussian.eval(pixelCoord * (1.0/texSize));
        
        result = alphaBlend(result, blobColor);
        if (result.a < 1.f / 255.f)
            continue;
    }

    // Blend with background
    return float4(result.rgb * (1.0 - result.a) + result.a, 1.0);
}

void renderBlobsToTexture(
    RWTexture2D<float4> output,
    GradInOutTensor<float, 1> blobsBuffer,
    uint2 pixelCoord)
{
    uint2 imageSize;
    output.GetDimensions(imageSize.x, imageSize.y);
    output[pixelCoord] = simpleSplatBlobs(blobsBuffer, pixelCoord, imageSize);
}

/*
 * loss() implements the standard L2 loss function to quantify the difference between
 * the rendered image and the target texture.
 */
[Differentiable]
float loss(uint2 pixelCoord, int2 imageSize, Blobs blobs, Texture2D<float4> targetTexture)
{
    int texWidth;
    int texHeight;
    targetTexture.GetDimensions(texWidth, texHeight);
    int2 texSize = int2(texWidth, texHeight);

    // Splat the blobs and calculate the color for this pixel.
    float4 color = simpleSplatBlobs(blobs.blobsBuffer, pixelCoord, imageSize);

    float4 targetColor;
    float weight;
    if (pixelCoord.x >= imageSize.x || pixelCoord.y >= imageSize.y)
    {
        return 0.f;
    }
    else
    {
        targetColor = no_diff targetTexture[pixelCoord];
        return dot(color.rgb - targetColor.rgb, color.rgb - targetColor.rgb);
    }

    return 0.f;
}

// Differentiable function to compute per-pixel loss
// Parameters:
// output:          a 2-dimensional tensor of float4 values, representing the output texture
// pixelCoord:      the coordinates of the output pixel whose loss is being calculated
// blobsBuffer:     a 1-dimensional tensor of floats, containing the Gaussian blobs
[Differentiable]
void perPixelLoss(
    GradInOutTensor<float4, 2> output, 
    uint2 pixelCoord,
    GradInOutTensor<float, 1> blobsBuffer,
    Texture2D<float4> targetTexture)
{
    uint2 imageSize;
    targetTexture.GetDimensions(imageSize.x, imageSize.y);
    output.set(
        {pixelCoord.x, pixelCoord.y},
        loss(pixelCoord, imageSize, {blobsBuffer}, targetTexture));
}

/*
 * clearDerivativesMain() is a kernel that resets the derivative buffer to all 0s
 */
void clearDerivs(uint3 dispatchThreadID, RWNDBuffer<uint, 1> derivBuffer)
{
    derivBuffer[dispatchThreadID.x] = asuint(0.f);
}

/*
 * Output a constant. Useful to quickly clear a buffer to a specific 
 * value with slangpy
 */
void ones(out float4 val)
{
    val = float4(1.f);
}

/*
 * updateBlobsMain() is a kernel that updates the blob parameters using the Adam optimizer.
 *
 * Since all the parameters are laid out in a single float buffer, there is no need to re-interpret
 * the buffer into a struct.
 *
 * The Adam optimization method (https://arxiv.org/abs/1412.6980) is used to process the gradients before
 * applying the update. It acts as a temporal filter on the gradients, and stores per-parameter state that
 * persists across iterations to help stabilize the optimization process.
 *
 */
void adamUpdate(
    inout float val,
    inout float dVal,
    inout float firstMoment,
    inout float secondMoment)
{
    // Read & reset the derivative
    float g_t = dVal;

    float g_t_2 = g_t * g_t;

    //
    // Perform a gradient update using Adam optimizer rules for
    // a smoother optimization.
    //

    float m_t_prev = firstMoment;
    float v_t_prev = secondMoment;
    float m_t = ADAM_BETA_1 * m_t_prev + (1 - ADAM_BETA_1) * g_t;
    float v_t = ADAM_BETA_2 * v_t_prev + (1 - ADAM_BETA_2) * g_t_2;

    firstMoment = m_t;
    secondMoment = v_t;

    float m_t_hat = m_t / (1 - ADAM_BETA_1);
    float v_t_hat = v_t / (1 - ADAM_BETA_2);

    float update = (ADAM_ETA / (sqrt(v_t_hat) + ADAM_EPSILON)) * m_t_hat;

    val -= update;
    dVal = 0.f;
}
