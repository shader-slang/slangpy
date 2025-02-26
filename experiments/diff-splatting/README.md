# 2D Differentiable Gaussian Splatting

## About

This example demonstrates the use of Slang and SlangPy to implement a 2D Gaussian splatting algorithm. 

This algorithm represents a simplified version of the 3D Gaussian Splatting algorithm detailed in this paper (https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). This 2D demonstration does not have the 3D->2D projection step & assumes that the Gaussian blobs are presented in order of depth (higher index = farther away). Further, this implementation does not perform adaptive density control to add or remove blobs. 

See the `computeDerivativesMain()` kernel and the `splatBlobs()` function for the bulk of the key pieces of the code. This sample uses SlangPy (see `main.py`) to easily load and dispatch the kernels. SlangPy handles the pipeline setup, buffer allocation, buffer copies, and other boilerplate tasks to make it easy to prototype high-performance differentiable code.

For a full 3D Gaussian Splatting implementation written in Slang, see this repository: https://github.com/google/slang-gaussian-rasterization

### Workaround for 'compressing' a 2D group size into a 1D group
This sample uses a workaround for SlangPy's fixed group size of `(32, 1, 1)`. The rasterizer uses a fixed `8x4` 2D tile. We use numpy commands to construct an aray of dispatch indices such that the right threads receive the right 2D thread index. `calcCompressedDispatchIDs()` in `main.py` holds the logic for this workaround. 

When SlangPy is updated with the functionality to specify group sizes, this workaround will be removed.

## How to Use

### Installation

First, install slangpy and the tev viewer: 

- **SlangPy** python package: `pip install slangpy`. See SlangPy's [docs](https://slangpy.shader-slang.org/en/latest/installation.html) for a full list of requirements.
- **Tev** viewer: Download from [releases](https://github.com/Tom94/tev/releases/tag/v1.29). See [tev's github](https://github.com/Tom94/tev) for more information.  

Then install the example's requirements, from within the sample folder:

`pip install -r requirements.txt`

### Optional: Setup via Conda
For simpler setup, use an anaconda/miniconda installation (See [Conda's user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) for more).

Ensure that your environment is using **Python 3.10**. 
If you are using conda, you can create a new environment with **python 3.10** and **slangpy** both installed using the following command: 
`conda create -n slangpy-env python=3.10 slangpy`. Then switch to this new environment with `conda activate slangpy-env`.

### Running the Sample
- Open the **Tev** viewer and keep it running in the background.
- From the sample folder, run `python main.py` from a terminal.

You should see a stream of images in Tev as the training progresses:

![](./example-image.png)