# Contributing / Developing

## Setup

```
#Create conda environment if needed
conda create -n "kernelfunctions" python=3.9

#Clone
git clone https://gitlab-master.nvidia.com/ccummings/kernelfunctions.git
cd kernelfunctions

#Install as local, editable package
pip install --editable .

#Install developer extras
pip install -r requirements.txt

#Install precommit hooks
pre-commit install

#Test precommit
pre-commit run --all-files
```

## Tests

If opened in VS Code, the default setup will detect and register tests in the VS Code testing tools. To run manually:

```
pytest tests
```

To debug a test, simply run the corresponding test file

## Adding new tests

`tests/test_sgl.py` is a very basic test example. Note it:
- Updates the path before importing helpers, so test can be run from any directory
- Use parameterization to create a test that runs once per device type
- Includes an `__main__` handler at the bottom to allow the file to be debugged
