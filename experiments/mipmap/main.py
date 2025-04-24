# SPDX-License-Identifier: Apache-2.0

from app import App
import slangpy as spy
import numpy as np
import sgl
import sys
import time
from slangpy.types import call_id


# The purpose of this example is to render pixels using input material and
# normal map textures, and calculate per-pixel lighting.
#
# This can be expected to give correct results at full res, but when using
# lower res texture inputs (mipmap levels) the result will be somewhat
# incorrect, represented as per-pixel L2 loss values, which tell us how
# the inputs need to change to give the correct values.
#
# This app has 5 modes, set by the mode global variable in main.py:
# 1. Evaluate pixel color using full res textures. This simply reads full res
#    texture data and evaluates a simple BRDF (in this case phone).
# 2. Evaluate a lower res pixel color by using mode (1) to evaluate 4 adjacent
#    values and take the average. This is similar to using lower res mipmap
#    levels but should give a more accurate result.
# 3. Evaluate a lower res pixel by using mode (1) with lower res texture
#    inputs, ie, lower res mipmap levels. This should be less accurate
#    than mode 2.
# 4. Show the difference between mode (1) and (3), ie the per-pixel
#    L2 loss values between full res and low res inputs.
# 5. Train per-pixel loss values to reach the expected values from (4).
#    This is shown by rendering the learned loss values as white pixels,
#    which should match the values from mode (4).
mode = 1


# Simple function to read a texture from a file and exit if an error occurs.
def createTextureFromFile(device, filepath):
    try:
        loader = sgl.TextureLoader(device)
        texture = loader.load_texture(filepath)
        return texture

    except Exception as e:
        print(f"\nError loading the texture: {e}")
        sys.exit(1)


# Alternatively, create a texture from a list of floating point values.
def createTextureFromList(values, width, height):
    # Convert to numpy array if not already.
    data = np.array(values, dtype=np.float32)

    # Create the texture.
    texture = app.device.create_texture(
        width=width,
        height=height,
        format=sgl.Format.r32_float,
        usage=sgl.TextureUsage.shader_resource | sgl.TextureUsage.unordered_access,
        data=data
    )
    return texture



# Create the app and load the sample shader.
app = App()
module = spy.Module.load_from_file(app.device, "mipmapping.slang")

# Get the app's window size.
windowSize = sgl.float2(app._window.width, app._window.height)

# Material and normal map textures used for this example.
materialTexture = createTextureFromFile(app.device, "diffuse.jpg")
normalTexture = createTextureFromFile(app.device, "normal.jpg")

# We need a sampler for the above textures. We want to do the filtering and
# mipmapping work ourselves, so this uses a simple point sampler to read data
# from the texture.
pointSampler = app.device.create_sampler(
    min_filter=sgl.TextureFilteringMode.point,
    mag_filter=sgl.TextureFilteringMode.point,
)

# We could sample the texture data randomly, but for this example we instead
# use one sample for each pixel, encompassing the whole window size.
sampleSize = int(windowSize.x) * int(windowSize.y)

# A higher learning rate will converge more quickly but with a potentially
# less exact result.
learningRate = 0.01

# We should stop the learning at some point, as there are very diminishing
# returns after a number of iterations. For this example 300 should be enough.
iteration = 300

# Sample positions, and the tensor for them.
# These can be random, and an example of how to init these with random points
# is given, but instead for this example we sample each pixel.
samplePoints = np.random.randn(sampleSize, 2).astype(np.float32)
i = 0
for x in range(int(windowSize.x)):
    for y in range(int(windowSize.y)):
        samplePoints[i] = [x, y]
        i += 1
# A tensor is then used for handling these sample points in the Slang shader.
samplePointsTensor = spy.Tensor.from_numpy(app.device, samplePoints)

# This example will learn the L2 loss values for each pixel, these will
# slowly converge on the correct values as the example learns.
lossVals = np.random.randn(sampleSize).astype(np.float32)
for i in range(sampleSize):
    lossVals[i] = 0.0
# Again we use a tensor for these, initialising the gradients to zero.
lossValTensor = spy.Tensor.from_numpy(app.device, lossVals).with_grads(zero = True)

# This tensor is to store the result of the learning, which requires gradients.
forwardResult = spy.Tensor.from_numpy(app.device, np.zeros((sampleSize, ), dtype=np.float32)).with_grads()

# Simple numpy array used for copying data out of the tensors.
allOnes = np.ones((sampleSize, 1), dtype=np.float32)


# This function does the learning, using the previously created arrays of
# data and tensors.
#
# We first call the forward function in the Slang shader, passing the textures
# and the previously created tensors. After this, we backpropagate any
# differentiable gradients from the forwards function, which is needed for
# learning. Then we adjust the resulting data based on the gradients and
# previously defined learning rate.
def findMachingLoss(iter):
    # Here we call the forward function defined in the Slang shader. This runs
    # on multiple threads. The sample and loss tensors previously created allow
    # passing data from arrays to the associated threads, as the function
    # definition only takes a single sample and loss value.
    module.forward(pointSampler, materialTexture, normalTexture, windowSize,
        samplePointsTensor, lossValTensor, _result = forwardResult)

    # Copy out the gradients.
    forwardResult.grad.storage.copy_from_numpy(allOnes)

    # Perform the backwards propagation of the gradients.
    module.forward.bwds(pointSampler, materialTexture, normalTexture,
        windowSize, samplePoints, lossValTensor, _result = forwardResult)

    # Get the resulting data and gradients, and adjust the values based on
    # the previously defined learning rate.
    lossResult = lossValTensor.to_numpy()
    lossGrad = lossValTensor.grad.to_numpy()
    lossResult = lossResult - learningRate * lossGrad

    # Copy the data back into the tensor for the next iteration of learning,
    # and clear the previously calculated gradients.
    lossValTensor.storage.copy_from_numpy(lossResult)
    lossValTensor.grad.clear()

    # Periodically print some data to confirm that the learning is
    # actually happening.
    if iter % 50 == 0:
        resultArray = forwardResult.to_numpy()
        loss = np.linalg.norm(resultArray) / sampleSize
        print("Iteration: {}, Loss: {}".format(iter, loss))
        print("Loss val {}".format(lossValTensor.to_numpy()))

    # TODO: No garbage collection?
    #app.device.run_garbage_collection()
    return forwardResult


# Run the app, using the previously defined mode. Only mode 5 will perform
# actual training.
iter = 0
while app.process_events():
    # Only perform training and show the learned result in mode 5, otherwise
    # pick a mipmapping mode.
    if mode == 5:
        if iter < iteration:
            forwardResultTensor = findMachingLoss(iter)
            iter += 1

        # Populate a texture with the current loss values, so we can visually
        # show the results.
        lossValList = lossValTensor.to_numpy().tolist()
        lossTexture = createTextureFromList(lossValList, int(windowSize.x), int(windowSize.y))
        module.renderLearnedLoss(pointSampler, materialTexture, normalTexture, lossTexture,
            windowSize, call_id(), _result = app.output)
    else:
        module.renderMipmapMode(pointSampler, materialTexture, normalTexture, windowSize,
            mode, call_id(), _result = app.output)
    app.present()
