# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy
import pathlib
import torch
import numpy as np
import sgl
from app import App
from slangpy.types import call_id
import time


def computeRef(input, parameter):
    target = np.linalg.norm(input) - 1.0
    pred = np.linalg.norm(input - parameter[:3]) - parameter[3]
    diff = target - pred

    grad = np.zeros((4, ), dtype=np.float32)
    grad[:3] = -2 * diff * (parameter[:3] - input) / (pred + parameter[3])
    grad[3] = 2 * diff
    return grad

# Create an SGL device with the local folder for slangpy includes
#  device = spy.create_device(include_paths=[
#      pathlib.Path(__file__).parent.absolute(),
#  ])


app = App()
module = spy.Module.load_from_file(app.device, "example.slang")

samplesSize = 32
learningRate = 0.1
iteration = 1500

samplePoint = np.random.randn(samplesSize, 3).astype(np.float32)
allOnes = np.ones((samplesSize, 1), dtype=np.float32)
input = spy.Tensor.numpy(app.device, samplePoint)

# Create a tensor
paramArr = np.random.randn(4).astype(np.float32) * 3
paramArr[3] = np.abs(paramArr[3])
params = spy.Tensor.numpy(app.device, paramArr).with_grads(zero=True)
forwardResult = spy.Tensor.numpy(app.device, np.zeros(
    (samplesSize, ), dtype=np.float32)).with_grads()
print(params.to_numpy())


def findMachingSDF(iter):
    #  for i in range(iteration):
    module.forward(samplePoint=input, sdf_params=params, _result=forwardResult)

    forwardResult.grad.storage.copy_from_numpy(allOnes)

    module.forward.bwds(samplePoint, params, _result=forwardResult)

    paramArray = params.to_numpy()
    gradArray = params.grad.to_numpy()
    paramArray = paramArray - learningRate * gradArray / samplesSize
    paramArray[3] = np.fmax(paramArray[3], 0.0)  # Clamp the radius to be positive

    params.storage.copy_from_numpy(paramArray)
    params.grad.clear()

    if iter % 50 == 0:
        resultArray = forwardResult.to_numpy()
        loss = np.linalg.norm(resultArray) / samplesSize
        print("Iteration: {}, Loss: {}".format(iter, loss))
        print("parameter {}".format(params.to_numpy()))

    app.device.run_garbage_collection()
    return forwardResult


iter = 0
while app.process_events():
    if (iter < iteration):
        forwardResult = findMachingSDF(iter)
        iter += 1
    pylist = params.to_numpy().tolist()
    windowSize = sgl.float2(app._window.width, app._window.height)
    module.RunRayMarch(windowSize, call_id(), pylist, _result=app.output)
    time.sleep(0.005)  # Sleep for 10ms to see the evolution of the SDF
    app.present()
