# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from slangpy.backend import DataType, Device, DeviceType, TextureLoader
from slangpy import Module
from slangpy.types import NDBuffer
import numpy as np
import math
import time

from network import ModuleChain, LinearLayer, FrequencyEncoding
from network import Activation, NoneAct, ReLUAct, LeakyReLUAct, ELUAct, SwishAct, TanhAct, SigmoidAct, ExpAct

from app import App


def training_main():
    resolution = 512
    app = App("Neural Texture", device_type=DeviceType.vulkan, width=resolution, height=resolution)
    device = app.device

    model = ModuleChain(
        FrequencyEncoding(2, 5),
        LinearLayer(20, 64),
        LeakyReLUAct(64),
        LinearLayer(64, 64),
        LeakyReLUAct(64),
        LinearLayer(64, 64),
        LeakyReLUAct(64),
        LinearLayer(64, 3),
        SigmoidAct(3)
    )
    model.initialize(device)

    module = Module.load_from_file(device, "NeuralTexture.slang")

    optimizers = [module.AdamOptimizer(p) for p in model.parameters()]

    batch_shape = (256, 256)
    learning_rate = 0.001
    grad_scale = 128.0
    loss_scale = grad_scale / math.prod(batch_shape)
    num_batches_per_epoch = 10

    seeds = np.random.get_bit_generator().random_raw(batch_shape).astype(np.uint32)
    rng = module.RNG(seeds)

    loader = TextureLoader(device)
    target_tex = loader.load_texture("bernie.jpg", {"load_as_normalized": True})
    sampler = device.create_sampler(min_lod=0, max_lod=0)
    uv_grid = create_uv_grid(device, resolution)

    timer = Timer()
    cmd = device.create_command_buffer()

    while app.process_events():
        timer.start()

        # Prefetch functions so we don't do module lookups in a tight loop
        train = module.trainTexture
        step = module.AdamOptimizer.step

        cmd.open()
        for i in range(num_batches_per_epoch):
            train.append_to(cmd, model, rng, target_tex, sampler, loss_scale)
            for params, optim in zip(model.parameters(), optimizers):
                step.append_to(cmd, optim, params, params.grad_out, learning_rate, grad_scale)
        cmd.close()

        id = device.submit_command_buffer(cmd)
        # Stall and wait, then garbage collect for a good interactive experience.
        # Will slow things down a lot though - headless training will run faster.
        device.wait_command_buffer(id)
        device.run_garbage_collection()

        msamples = (num_batches_per_epoch * math.prod(batch_shape)) * 1e-6
        print(f"Throughput: {timer.frequency() * msamples:.2f} MSamples/s "
              f"Epoch time: {timer.elapsed() * 1e3:.1f}ms")

        module.evalModel(model, uv_grid, _result=app.output)

        app.present()
        timer.stop()


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
