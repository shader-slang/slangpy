# Developing with SlangPy

## Setup

```
#Create conda environment if needed
conda create -n "slangpy" python=3.9

#Install SGL unless building locally
pip install --upgrade nv-sgl

#Clone
git clone https://gitlab-master.nvidia.com/ccummings/slangpy.git
cd slangpy

#Install as local, editable package
pip install --editable .

#Install developer extras
pip install -r requirements-dev.txt

#Install precommit hooks
pre-commit install

#Test precommit
pre-commit run --all-files

#Run unit tests
pytest
```

## Tests

If opened in VS Code, the default setup will detect and register tests in the VS Code testing tools. To run manually:

```
pytest
```

To debug a test, simply run the corresponding test file

## Adding new tests

`slangpy/tests/test_sgl.py` is a very basic test example. Note it:
- Use parameterization to create a test that runs once per device type
- Includes an `__main__` handler at the bottom to allow the file to be debugged
