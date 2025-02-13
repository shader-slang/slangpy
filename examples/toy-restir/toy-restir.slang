// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import "slangpy";

// References:
//   "A Gentle Introduction to ReSTIR":
//       Course notes for the SIGGRAPH 2023 ReSTIR course
//       "Gentle Introduction to ReSTIR: Path Reuse in Real-Time"
//       Wyman et al. (2023)
//       https://intro-to-restir.cwyman.org/

// ---------------------------------------------------------------------------

static const float gMotionSpeed = 0.01f; // Wall motion speed.
static const float gConfidenceCap = 50.0f; // Maximum confidence for a reservoir. Best value varies case-by-case.
static const int gReuseRange = 20; // How far in pixels are neighbors selected from for spatial reuse.

// ---------------------------------------------------------------------------

// Standard normal distribution.
float standardNormal(float x) {
    const float c = 1.0f / sqrt(2.0f * float.getPi());
    return c * exp(-0.5 * x * x);
}

// A normal distribution with parameters.
float normal(float x, float mu, float stdev) {
    return standardNormal((x - mu) / stdev) / stdev;
}

// ---------------------------------------------------------------------------

// A stream of pseudorandom numbers.
struct RandomStream {
    uint32_t state;

    __init(uint32_t seed) {
        state = seed;
    }
    
    [mutating]
    uint32_t next() {
        // Xorshift32 from https://en.wikipedia.org/wiki/Xorshift.
	    uint32_t x = state;
	    x ^= x << 13;
	    x ^= x >> 17;
	    x ^= x << 5;
        state = x;
	    return state;
    }

    // Next uniform random number in [0, 1).
    // May round to 1 due to floating point precision.
    [mutating]
    float uniform() {
        return next() / 4294967296.0f;
    }
};

// ---------------------------------------------------------------------------

// A "path" that simulates carrying radiance from the scene to pixels.
struct Path {
    float2 uv;     // Pixel coordinates in [0, 1)^2.
    float t;       // Time of sample.
    uint32_t seed; // Random seed of the path.
    float3 f;      // Amount of light carried by the path.
    
    __init() {
        uv = float2(-1.0f, -1.0f);
        t = 0.0f;
        seed = 0;
        f = float3(0.0f, 0.0f, 0.0f);
    }
};

// A sample that joins a path with its unbiased contribution weight.
// See "A Gentle Introduction to ReSTIR", Section 2.4.
struct Sample {
    Path path;  // The actual path sample.
    float W;    // Unbiased contribution weight (unbiased estimate for 1/PDF)
    float pHat; // Target function value

    __init()
    {
        path = Path();
        W = 0.0f;
        pHat = 0.0f;
    }
};

// Returns amount of light carried by the path.
float3 evaluatePath(Path path) {
    float x = path.uv[0];
    float y = 1.0 - path.uv[1];
    float t = path.t;

    // Recover the random stream for replay.
    RandomStream rng = RandomStream(path.seed);

    // Evaluate component contributions. Simulate path tracing noise in a "toy scene".
    float3 skyColor = float3(0.1f, 0.1f, 0.3f);
    float groundNoise = 0.1f + 0.9f * 2.0f * rng.uniform() * (float)(rng.uniform() < 0.6f) / 0.6f;

    // Ground texture.
    float groundAlpha1 = cos((x + 0.2f * y) * 50.0f);
    float groundAlpha2 = sin((0.7f * y + 0.2f * x) * 50.0f);
    float3 groundColor = float3(0.2f, 0.6f, 0.2f) + float3(0.03f, 0.04f, 0.03f) * groundAlpha1 + float3(0.03f, 0.02f, 0.04f) * groundAlpha2;
    groundColor *= groundNoise;

    // Sun texture.
    float sunBrightness = normal(x, 0.5f, 0.2f) * normal(y, 0.5f, 0.2f) * (2.0f * float(rng.uniform() < 0.5f));
    float3 sunColor = float3(1.0f, 1.0f, 0.0f);

    // Wall texture.
    float brickX = 0.06f * (float)((sin((0.3f - y) * 50.0f)) >= 0.0f);
    float jointFactorY = max(0.0f, 0.9f * (float)(cos((0.3f - y) * 100.0f) > 0.7f));
    float jointFactorX = max(0.0f, 0.9f * (float)(cos((brickX + gMotionSpeed * t - x) * 40.0f) > 0.95f));
    if (y > 0.3f) {
        float jointFactorX2 = max(0.0f, 0.9f * (float)(abs(cos(20.0f * (x - gMotionSpeed * t - 0.5f))) < 0.15f));
        jointFactorX = max(jointFactorX, jointFactorX2);
    }
    float jointFactor = max(jointFactorX, jointFactorY);
    float3 wallColor = (float3(2.0f, 2.0f, 1.0f) - float3(1.1f, 1.1f, 0.1f) * jointFactor) * (float)(rng.uniform() < 0.05f);

    // Combine.
    float3 radiance = float3(0.0f);

    float wallHeight = 0.3f + 0.15f * (float)(cos(20.0f * (x - gMotionSpeed * t - 0.5f)) > 0.0f);
    if (y < wallHeight) return wallColor;

    float groundHeight = 0.618f;
    if (y < groundHeight) return groundColor;

    return skyColor + sunBrightness * sunColor;
}

// Produces a new random sample.
Sample samplePath(int2 pixel, int frame, int2 imageSize, inout RandomStream pathRandom)
{
    Sample sample;
    
    // Randomize subpixel location within the pixel.
    float2 uv = (float2)pixel + float2(pathRandom.uniform(), pathRandom.uniform());
    sample.path.uv = uv / (float2)imageSize;
    sample.path.t = (float)frame;

    // Capture random seed.
    sample.path.seed = pathRandom.state;

    // Evaluate the path. This is normally done during sampling.
    sample.path.f = evaluatePath(sample.path);

    // Contribution weight W = 1/PDF = 1.0 in this toy case.
    // (We use W = 1/PDF since we know the PDF.)
    sample.W = 1.0f;

    // We use target function pHat = mean(f).
    sample.pHat = pHat(sample.path);

    return sample;
}

// Estimate a backward motion vector v for which ideally currentImage[pixel] ≈ priorImage[pixel + v].
int2 backwardMotionVector(int2 pixel, float t, int2 imageSize)
{
    var motion = int2(0, 0);

    float2 uv = float2(pixel) / (float2)imageSize;
    float x = uv[0];
    float y = 1.0f - uv[1];

    float wallHeight = 0.3f + 0.15f * (float)(cos(20.0f * (x - gMotionSpeed * t - 0.5f)) > 0.0f);
    if (y < wallHeight) motion[0] = int(-gMotionSpeed * imageSize[0]);
    
    return motion;
}

// Shifts the path by 'motion' on the image plane.
// See "A Gentle Introduction to ReSTIR", Section 5.1.
//
// Returns the Jacobian determinant, which is set to 0.0 if the shift is undefined.
float shiftPath(Path path, float2 shift, float timeDelta, out Path offsetPath)
{
    Path result;
    float jacobian = 0.0;

    result.uv = path.uv + shift;
    result.seed = path.seed;
    result.t = path.t + timeDelta;
    result.f = float3(0.0, 0.0, 0.0);
    
    bool onScreen = (result.uv[0] >= 0.0 && result.uv[0] < 1.0 && result.uv[1] >= 0.0 && result.uv[1] < 1.0);
    
    if (onScreen)
    {
        result.f = evaluatePath(result);
        jacobian = 1.0; // Will be more complicated in real problems.
    }

    offsetPath = result;
    return jacobian;
}

// ---------------------------------------------------------------------------

// Target function for resampling.
float pHat(Path path)
{
    return (path.f[0] + path.f[1] + path.f[2]) / 3.0f;
}

// Reservoir for Weighted Reservoir Sampling.
// See "A Gentle Introduction to ReSTIR", Sections 3 and 3.1.
struct Reservoir
{
    Sample sample;  // Current sample.
    float wSum;     // Sum of resampling weights.
    float c;        // Confidence weight.

    __init()
    {
        wSum = 0.0f;
        c = 0.0f;
    }

    [mutating]
    void update(Sample x, float resamplingWeight, float uniformRandom)
    {
        wSum += resamplingWeight;
        if (uniformRandom * wSum <= resamplingWeight) // uniformRandom < resamplingWeight / wSum
        {
            sample = x;
        }

        if (wSum > 0.0f)
        {
            sample.W = wSum / sample.pHat;
        }
    }
};


void resetReservoirs(inout Reservoir reservoir) {
    reservoir.wSum = 0.0f;
    reservoir.c = 0.0f;
}

// ---------------------------------------------------------------------------

// A naive and unoptimized generalized resampled importance sampling implementation for exactly one neighbor.
// See "A Gentle Introduction to ReSTIR", Section 5 and Algorithm 5.
struct GRIS
{
    [mutating]
    void prepareCanonicalDomain(float confidence) {
        canonicalConfidence = confidence;
    }

    [mutating]
    void prepareNeighborDomain(float2 offset, float timeDelta, float confidence) {
        neighborOffset = offset;
        neighborTimeDelta = timeDelta;
        neighborConfidence = confidence;
    }

    [mutating]
    void addCanonicalSample(Sample sample, inout RandomStream risRandom)
    {
        float misWeight = 0.0f;

        // MIS weight (generalized balance heuristic).
        // See Equations 5.11 and 5.9 in "A Gentle Introduction to ReSTIR".
        if (sample.pHat > 0.0f && sample.W > 0.0f)
        {
            Path pathInNeighbor;
            float jacobianToNeighbor = shiftPath(sample.path, neighborOffset, neighborTimeDelta, pathInNeighbor);
            pathInNeighbor.f = evaluatePath(pathInNeighbor);

            float pHatFrom_current = sample.pHat;
            float pHatFrom_neighbor = pHat(pathInNeighbor) * jacobianToNeighbor;

            misWeight = (canonicalConfidence * pHatFrom_current) / (canonicalConfidence * pHatFrom_current + neighborConfidence * pHatFrom_neighbor);
        }

        // Resampling weight.
        float w = misWeight * sample.pHat * sample.W;

        // Add to resampling.
        result.update(sample, w, risRandom.uniform());
        result.c += 1.0f;
    }

    [mutating]
    void addNeighborSample(Sample sample, inout RandomStream risRandom)
    {
        Sample sampleInCurrent;

        float misWeight = 0.0f;
        float jacobianFromNeighbor = 0.0f;

        // Evaluate MIS weight (generalized balance heuristic).
        // See Equations 5.11 and 5.9 in "A Gentle Introduction to ReSTIR".
        if (sample.pHat > 0.0f && sample.W > 0.0f)
        {
            jacobianFromNeighbor = shiftPath(sample.path, -neighborOffset, -neighborTimeDelta, sampleInCurrent.path);
            if (jacobianFromNeighbor > 0.0f)
            {
                sampleInCurrent.path.f = evaluatePath(sampleInCurrent.path);
                sampleInCurrent.pHat = pHat(sampleInCurrent.path);

                float pHatFrom_neighbor = sample.pHat / jacobianFromNeighbor;
                float pHatFrom_current = sampleInCurrent.pHat;

                misWeight = (neighborConfidence * pHatFrom_neighbor) / (neighborConfidence * pHatFrom_neighbor + canonicalConfidence * pHatFrom_current);
            }
        }

        // Resampling weight.
        float w = misWeight * sampleInCurrent.pHat * sample.W * jacobianFromNeighbor;

        // Add to resampling.
        result.update(sampleInCurrent, w, risRandom.uniform());
        result.c += neighborConfidence;
    }

    [mutating]
    Reservoir resample()
    {
        result.sample.W = select(result.sample.pHat > 0.0f, result.wSum / result.sample.pHat, 0.0f);
        return result;
    }

    __init()
    {
        neighborOffset = float2(0.0f, 0.0f);
        neighborTimeDelta = 0.0f;
        neighborConfidence =  0.0f;
        canonicalConfidence = 0.f;
    }

    float2 neighborOffset;
    float neighborTimeDelta;
    float neighborConfidence;
    float canonicalConfidence;

    Reservoir result;
};

// ---------------------------------------------------------------------------

void sampleAndMergeInitialCandidate(int2 pixel, int frame, int2 imageSize, int sampleCount, inout Reservoir reservoir, inout RandomStream pathRandom, inout RandomStream risRandom)
{
    Sample sample = samplePath(pixel, frame, imageSize, pathRandom);
    
    float misWeight = 1.0f / (float)sampleCount;
    float resamplingWeight = misWeight * sample.pHat * sample.W;

    reservoir.update(sample, resamplingWeight, risRandom.uniform());
    reservoir.c = 1.0f;
}

void performTemporalReuse(int2 pixel, int frame, inout Reservoir outReservoir, in RWNDBuffer<Reservoir, 2> initialSampleReservoirs, in RWNDBuffer<Reservoir, 2> temporalReservoirs, int2 imageSize,
                         inout RandomStream risRandom)
{
    float t = (float)frame;

    // Get the initial sample's reservoir.
    let currentReservoir = initialSampleReservoirs[pixel];

    // Pixel delta to prior-frame's corresponding pixel.
    var motionVector = backwardMotionVector(pixel, t, imageSize);
    var neighborPixel = pixel + motionVector;

    // Optional (not always ideal): Clamp to the image, but randomize to avoid artifacts.
    if (neighborPixel[0] < 0 || neighborPixel[0] >= imageSize[0] || neighborPixel[1] < 0 || neighborPixel[1] >= imageSize[1])
    {
        neighborPixel += int2(-1 + (risRandom.next() % 3), -1 + (risRandom.next() % 3));
        neighborPixel = clamp(neighborPixel, 0, imageSize - int2(1, 1));
        motionVector = neighborPixel - pixel;
    }

    var shift = float2(motionVector) / float2(imageSize);

    // Disable the temporal input if trying to reuse outside of the image.
    // (Not active due to clamping.)
    Reservoir neighborReservoir;
    if (neighborPixel[0] >= 0 && neighborPixel[0] < imageSize[0] && neighborPixel[1] >= 0 && neighborPixel[1] < imageSize[1])
    {
        neighborReservoir = temporalReservoirs[neighborPixel];
    } else {
        neighborReservoir.sample.W = 0.0f;
    }

    // Resample one from the two.
    GRIS gris;
    gris.prepareCanonicalDomain(currentReservoir.c);
    gris.prepareNeighborDomain(shift, -1.0f, neighborReservoir.c);
    gris.addCanonicalSample(currentReservoir.sample, risRandom);
    gris.addNeighborSample(neighborReservoir.sample, risRandom);

    Reservoir result = gris.resample();
    result.c = min(result.c, gConfidenceCap); // Arbitrary confidence cap.

    outReservoir = result;
}

void performSpatialReuse(int2 pixel, inout Reservoir outReservoir, in RWNDBuffer<Reservoir, 2> reservoirs, int2 imageSize, inout RandomStream risRandom)
{
    // Get current pixel's reservoir.
    let currentReservoir = reservoirs[pixel];

    // Randomize spatial input reservoir.
    var reuseWindowX = int2(max(0, pixel[0] - gReuseRange), min(imageSize[0], pixel[0] + gReuseRange));
    var reuseWindowY = int2(max(0, pixel[1] - gReuseRange), min(imageSize[1], pixel[1] + gReuseRange));

    // Randomize the reuse pixel from a box around the current pixel.
    // Improved pixel selection based on e.g., object similarity would significantly improve results.
    var reusePixel = int2((int)(reuseWindowX[0] + (float)(reuseWindowX[1] - reuseWindowX[0]) * risRandom.uniform()),
                          (int)(reuseWindowY[0] + (float)(reuseWindowY[1] - reuseWindowY[0]) * risRandom.uniform()));
    var pixelDelta = reusePixel - pixel;
    var shift = float2(pixelDelta) / float2(imageSize);
    
    Reservoir neighborReservoir = reservoirs[reusePixel];

    // Resample one from the two.
    GRIS gris;
    gris.prepareCanonicalDomain(currentReservoir.c);
    gris.prepareNeighborDomain(shift, 0.0f, neighborReservoir.c);
    gris.addCanonicalSample(currentReservoir.sample, risRandom);
    gris.addNeighborSample(neighborReservoir.sample, risRandom);

    Reservoir result = gris.resample();
    result.c = min(result.c, gConfidenceCap); // Arbitrary confidence cap.

    outReservoir = result;
}

void evaluate(in Reservoir reservoir, inout float4 color)
{
    float3 radiance = reservoir.sample.path.f * reservoir.sample.W;
    color = float4(radiance, 1.0);
}

void copyToOutput(int2 pixel, int2 offset, Texture2D<float4> input, RWTexture2D<float4> output)
{
    output[pixel + offset] = input[pixel];
}

// ---------------------------------------------------------------------------

