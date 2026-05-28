# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy
import numpy as np

IMAGE_COUNT = 1024
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024


def write_test():
    print("writing images")
    t = spy.Timer()
    for i in range(IMAGE_COUNT):
        bmp = spy.Bitmap(np.random.randint(0, 255, (IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=np.uint8))
        bmp.write_async(f"test{i}.png")
        # bmp = spy.Bitmap(np.random.rand(IMAGE_WIDTH, IMAGE_HEIGHT, 3, dtype=np.uint8))
        # bmp.convert(component_type=spy.Bitmap.ComponentType.uint8, srgb_gamma=True).write_async(f"test{i}.exr")

    print("waiting")
    spy.thread.wait_for_tasks()
    print("done")
    print(t.elapsed_s())


def read_test_serial():
    print("reading images (serial)")
    t = spy.Timer()
    bmps = []
    for i in range(IMAGE_COUNT):
        bmps.append(spy.Bitmap(f"test{i}.png"))
    print(f"done ({len(bmps)} images)")
    print(t.elapsed_s())


def read_test_parallel():
    print("reading images (parallel)")
    t = spy.Timer()
    paths = list([f"test{i}.png" for i in range(IMAGE_COUNT)])
    bmps = spy.Bitmap.read_multiple(paths)
    print(f"done ({len(bmps)} images)")
    print(t.elapsed_s())


# write_test()
# read_test_serial()
read_test_parallel()
