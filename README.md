# SlangPy

SlangPy is a library designed to make calling GPU code written in Slang extremely simple and easy.
It's core features are:
- Make it quick and simple to call Slang functions on the GPU from Python
- Eradicate the boilerplate and bugs associated with writing compute kernels
- Full auto-diff support using Slang's auto-diff features
- Optional PyTorch support out of the box

It is built upon 2 core NVidia technologies:
- Slang shading language
- Slang graphics library (SGL)

## Installation

As SlangPy is coupled with a specific version of SGL, we recommend the use of a virtual environment,
typically VEnv on Linux or Anaconda on Windows. Whether in a global or virtual environment, basic
installation is:

```
pip install slangpy
```

To download the repo and run locally, checkout either the `main` or `stable` branch, 

```
git clone https://github.com/shader-slang/slangpy.git
git checkout stable
pip install -r ./requirements.txt
pip install .
```

Note that if using the `main` branch, you may need to clone and build the latest revision of SGL rather than the package installed with pip.





