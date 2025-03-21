# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from slangpy.backend import Device, DeviceType, TextureLoader
from slangpy import Module
from slangpy.types import NDBuffer
import numpy as np
import math
import time

# Import neural network library
import neuralnetworks as nn

from app import App

import slangpy as spy
from slangpy.reflection import SlangType


def training_main():
    # Create vulkan device and add include paths of neural network library
    device = spy.create_device(DeviceType.vulkan, False, include_paths=nn.slang_include_paths())

    resolution = 512
    app = App(device, "Neural Texture", width=resolution, height=resolution)
    device = app.device

    # If the device supports cooperative vector, run MLP at half precision and in coopvec mode
    if "cooperative-vector" in device.features:
        print("Cooperative vector enabled!")
        mlp_input = nn.ArrayKind.coopvec
        mlp_precision = nn.Real.half
    else:
        print("Device does not support cooperative vector. Sample will run, but it will be slow")
        mlp_input = nn.ArrayKind.array
        mlp_precision = nn.Real.float

    # Set up model architecture
    model = nn.ModelChain(
        nn.Convert.to_array(),
        nn.FrequencyEncoding(6),
        nn.Convert.to_precision(mlp_precision),
        nn.Convert.to_array_kind(mlp_input),
        nn.LinearLayer(nn.Auto, 64),
        nn.LeakyReLU(),
        nn.LinearLayer(nn.Auto, 64),
        nn.LeakyReLU(),
        nn.LinearLayer(nn.Auto, 64),
        nn.LeakyReLU(),
        nn.LinearLayer(nn.Auto, 3),
        nn.Convert.to_vector(),
        nn.Convert.to_precision(nn.Real.float),
        nn.Exp(),
    )

    # Create optimizer
    optim = nn.AdamOptimizer(learning_rate=0.0005)

    # If the MLP runs in half precision, wrap adam in a FullPrecisionOptimizer to
    # optimize a of the parameters in floating point and avoid repeated rounding
    # errors of 16bit float during training.
    grad_scale = 1.0
    if mlp_precision == nn.Real.half:
        optim = nn.FullPrecisionOptimizer(optim, gradient_scale=128.0)
        # Scale up gradients in half precision mode to avoid round-off errors
        grad_scale = 128.0

    # Load slang module containing our eval/training code and initialize the model and optimizer
    module = Module.load_from_file(device, "NeuralTexture.slang")
    model.initialize(module, module.float2)
    optim.initialize(module, model.parameters())

    # Optimize in increments of 256x256 samples per batch
    batch_shape = (256, 256)
    loss_scale = grad_scale / math.prod(batch_shape)
    num_batches_per_epoch = 10

    # Initialize a separate random number generator per batch entry
    # by calling the RNG constructor over a grid of batch_shape seeds
    seeds = np.random.get_bit_generator().random_raw(batch_shape).astype(np.uint32)
    rng = module.RNG(seeds)

    # Load target texture to be learned and create a uv_grid of 0...1 values for
    # evaluating the trained model as we train
    loader = TextureLoader(device)
    target_tex = loader.load_texture("bernie.jpg", {"load_as_normalized": True})
    sampler = device.create_sampler(min_lod=0, max_lod=0)
    uv_grid = create_uv_grid(device, resolution)

    timer = Timer()
    cmd = device.create_command_buffer()

    while app.process_events():
        timer.start()

        # For a bit of extra performance, don't call module.trainTexture(....) directly,
        # but collect multiple calls into a command buffer and dispatch them as a big
        # block. This avoids some launch overhead
        cmd.open()
        for i in range(num_batches_per_epoch):
            # Backpropagate network loss
            module.trainTexture.append_to(cmd, model, rng, target_tex, sampler, loss_scale)
            optim.step(cmd)
        cmd.close()

        id = device.submit_command_buffer(cmd)
        # Stall and wait, then garbage collect for a good interactive experience.
        # Will slow things down a lot though - headless training will run faster.
        # The interactive perf is not representative of the actual training cost
        device.wait_command_buffer(id)
        device.run_garbage_collection()

        # Display some useful info as we go
        msamples = (num_batches_per_epoch * math.prod(batch_shape)) * 1e-6
        print(f"Throughput: {timer.frequency() * msamples:.2f} MSamples/s "
              f"Epoch time: {timer.elapsed() * 1e3:.1f}ms")

        # Evaluate the neural network over the uv_grid and write the results to the
        # output texture
        module.evalModel(model, uv_grid, _result=app.output)

        app.present()
        timer.stop()


class ToGrayscale(nn.IModel):
    """Example of extending the library with a custom component. See README.md for more information."""

    def __init__(self, weights: nn.AutoSettable[list[float]] = nn.Auto, dtype: nn.AutoSettable[nn.Real] = nn.Auto, width: nn.AutoSettable[int] = nn.Auto):
        super().__init__()
        self._weights = weights
        self._dtype = dtype
        self._width = width

    def model_init(self, module: Module, input_type: SlangType):
        input_array = nn.RealArray.from_slangtype(input_type)
        self.width = nn.resolve_auto(self._width, input_array.length)
        self.dtype = nn.resolve_auto(self._dtype, input_array.dtype)

        if self._weights is nn.Auto:
            # No weights supplied? -> Generate weights for a simple average
            self.weights = []
            for i in range(self.width):
                self.weights.append(1.0 / self.width)
        else:
            # Weights supplied? Double check they agree with the resolved width
            if len(self._weights) != self.width:
                self.model_error(
                    f"Expected {self.width} weights; received {len(self._weights)} instead")
            self.weights = self._weights

    @property
    def type_name(self) -> str:
        return f"ToGrayscale<{self.dtype}, {self.width}>"

    def get_this(self):
        return {
            "channelWeights": self.weights,
            "_type": self.type_name
        }


def create_uv_grid(device: Device, resolution: int):
    span = np.linspace(0, 1, resolution, dtype=np.float32)
    uvs_np = np.stack(np.broadcast_arrays(span[None, :], span[:, None]), axis=2)
    uvs = NDBuffer(device, 'float2', shape=(resolution, resolution))
    uvs.copy_from_numpy(uvs_np)
    return uvs


class Timer:
    def __init__(self, history: int = 16):
        super().__init__()
        self.index = 0
        self.begin = None
        self.times = [0.0] * history
        self.history = history

    def start(self):
        self.begin = time.time()

    def stop(self):
        if self.begin is None:
            return

        t = time.time()
        elapsed = t - self.begin
        self.begin = t

        self.times[self.index % self.history] = elapsed
        self.index += 1

        return self.elapsed()

    def elapsed(self):
        l = min(self.index, self.history)
        return 0 if l == 0 else sum(self.times[:l]) / l

    def frequency(self):
        e = self.elapsed()
        return 0 if e == 0 else 1.0 / e


if __name__ == "__main__":
    training_main()
