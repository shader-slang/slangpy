Installation
============

As SlangPy is coupled with a specific version of SGL, we recommend the use of a virtual environment,
typically VEnv on Linux or Anaconda on Windows. Whether in a global or virtual environment, basic
installation is:

::

   pip install slangpy

To download the repo and run locally, checkout either the ``main`` or ``stable`` branch, 

::

   git clone https://github.com/shader-slang/slangpy.git
   git checkout stable
   pip install -r ./requirements.txt
   pip install .

Note that if using the ``main`` branch, you may need to clone and build the latest revision of SGL rather than the package installed with pip.

To enable PyTorch integration, simply pip install pytorch as usual and it will be detected automatically by SlangPy.
