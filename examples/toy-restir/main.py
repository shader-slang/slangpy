# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# A toy renderer with ReSTIR (reservoir-based spatiotemporal sample reuse).
#
# This example uses ReSTIR to render the view of an animated 2D toy scene. Artificial noise is introduced
# to the path contributions to model path tracing noise. Samples are reused between pixels and frames with
# ReSTIR for lower-noise integration of the path tracing noise and the pixel footprint.
#
# The sample reuse between pixels and frames is based on generalized resampled importance sampling (GRIS)
# with the generalized balance heuristic; this is thoroughly documented in the "Gentle Introduction to ReSTIR"
# SIGGRAPH course (see below). Temporal reuse is made fully unbiased by using the prior-frame target
# function in the multiple importance sampling (MIS) in resampling. Temporal reuse follows integer
# motion vectors, which is not ideal, but works fine for low-frequency content. The implementation is
# theoretically proper, but lacks improvements like multiple spatial neighbors or G-buffer based sample
# rejection in spatial reuse, which could make reuse much more robust in practice. A proper application
# would further apply a high-quality denoiser to the ReSTIR output.
#
# For context and additional information, please see the ReSTIR course "A Gentle Introduction to ReSTIR:
# Path Reuse in Real-Time" (Wyman et al. 2023), https://intro-to-restir.cwyman.org/, especially the course notes.
#
# This application outputs the ReSTIR frames to 'tev'; first 1spp frames and then ReSTIR frames.

import slangpy as spy
import sgl
import numpy as np
import pathlib
from app import App

# Size of the image.
imageWidth = 1024
imageHeight = 1024
imageSize = sgl.int2(imageWidth, imageHeight)

# Create windows app with space for 2 images.
app = App("Toy ReSTIR", imageWidth * 2, imageHeight)

# Number of initial candidates per pixel for ReSTIR.
initialCandidateCount = 1

# Number of frames to render.
frameCount = 400

# Create the device.
device = app.device

# Load the Slang code.
module = spy.Module.load_from_file(device, "toy-restir.slang")


# Initialize the random generators.
np.random.seed(0)
pathRandom = spy.NDBuffer(device, dtype=module.RandomStream, shape=(imageHeight, imageWidth))
risRandom = spy.NDBuffer(device, dtype=module.RandomStream, shape=(imageHeight, imageWidth))
risRandom.copy_from_numpy(np.random.randint(2**32, dtype=np.uint32, size=(imageHeight, imageWidth)))

# Create reservoirs.
initialOutput = spy.NDBuffer(device, dtype=module.Reservoir, shape=(imageHeight, imageWidth))
temporalOutput = spy.NDBuffer(device, dtype=module.Reservoir, shape=(imageHeight, imageWidth))
spatialOutput = spy.NDBuffer(device, dtype=module.Reservoir, shape=(imageHeight, imageWidth))

# Prepare the image.
tex = device.create_texture(
    width=imageWidth,
    height=imageHeight,
    format=sgl.Format.rgba32_float,
    usage=sgl.TextureUsage.shader_resource | sgl.TextureUsage.unordered_access
)


# First render without ReSTIR.
for frameIndex in range(frameCount):
    if not app.process_events():
        exit(0)

    # Populate the reservoirs with initial candidates.
    module.resetReservoirs(initialOutput)
    for i in range(initialCandidateCount):
        module.RandomStream(spy.wang_hash(
            seed=frameIndex * imageWidth * imageHeight), _result=pathRandom)
        module.sampleAndMergeInitialCandidate(
            spy.call_id(), frameIndex, imageSize, initialCandidateCount, initialOutput, pathRandom, risRandom)

    # Render.
    module.evaluate(initialOutput, tex)

    # Copy to app output window texture.
    module.copyToOutput(spy.grid((imageHeight, imageWidth)), sgl.int2(0, 0), tex, app.output)
    app.present()


# Then render with ReSTIR.
for frameIndex in range(frameCount):
    if not app.process_events():
        exit(0)

    # Populate the reservoirs with initial candidates.
    module.resetReservoirs(initialOutput)
    for i in range(initialCandidateCount):
        module.RandomStream(spy.wang_hash(
            seed=frameIndex * imageWidth * imageHeight), _result=pathRandom)
        module.sampleAndMergeInitialCandidate(
            spy.call_id(), frameIndex, imageSize, initialCandidateCount, initialOutput, pathRandom, risRandom)

    # Temporal reuse.
    module.performTemporalReuse(spy.call_id(), frameIndex, temporalOutput,
                                initialOutput, spatialOutput, imageSize, risRandom)

    # Spatial reuse.
    module.performSpatialReuse(spy.call_id(), spatialOutput, temporalOutput, imageSize, risRandom)

    # Render.
    module.evaluate(spatialOutput, tex)

    # Copy to app output window texture.
    module.copyToOutput(spy.grid((imageHeight, imageWidth)),
                        sgl.int2(imageWidth, 0), tex, app.output)
    app.present()

# Keep window processing events until user closes it.
while app.process_events():
    app.present()
