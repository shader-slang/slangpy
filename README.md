# SlangPy

SlangPy is a library designed to make calling GPU code written in Slang extremely simple and easy.
It's core objectives are to:
- Make it quick and simple to call Slang functions on the GPU from Python
- Eradicate the boilerplate and bugs associated with writing compute kernels
- Grant easy access to Slang's auto-diff features
- Provide optional PyTorch support out of the box

It is built upon 2 core NVidia technologies:
- [Slang shading language](https://shader-slang.com/): A modern, platform agnostic shading language with full auto-diff support.
- [Slang graphics library (SGL)](https://github.com/shader-slang/sgl): A powerful Python extension providing a thin wrapper around the graphics layer.

By bringing these 2 technologies together with a simple and flexible Python library, calling GPU code from Python is as simple and easy as calling a function.

## Documentation

For more detailed information and examples, see [the Documentation here](https://slangpy.readthedocs.io/).

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

To enable PyTorch integration, simply pip install pytorch as usual and it will be detected automatically by SlangPy.
