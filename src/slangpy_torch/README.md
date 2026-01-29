# slangpy-torch

Minimal PyTorch native extension providing fast (~28ns) tensor metadata access
from native code without Python API overhead (~350ns).

## Prerequisites

- **Python 3.9+**
- **PyTorch 2.0+** installed
- **C++ compiler**

### Windows

Install [Visual Studio 2019 or 2022](https://visualstudio.microsoft.com/) with the "Desktop development with C++" workload.

### Linux

```bash
# Ubuntu/Debian
sudo apt-get install build-essential
```

## Install

This extension **must** be installed with `--no-build-isolation` to ensure ABI compatibility
with your installed PyTorch version:

```bash
pip install slangpy-torch --no-build-isolation
```

### Verify Installation

```python
import torch # Before slangpy_torch
import slangpy_torch
print(slangpy_torch.get_api_ptr())  # Should print a non-zero integer
```

### Troubleshooting

- **"torch not found"**: Ensure PyTorch is installed first (`pip install torch`)
- **Windows linker errors**: Run from a "Developer Command Prompt for VS" or ensure MSVC is in PATH

## Usage from native code (slangpy_ext)

```cpp
#include "tensor_bridge_api.h"

// At init time:
auto bridge = nb::module_::import_("slangpy_torch");
auto api = reinterpret_cast<const TensorBridgeAPI*>(
    nb::cast<uintptr_t>(bridge.attr("get_api_ptr")())
);

// In hot path (~28ns):
TensorBridgeInfo info;
api->extract(handle.ptr(), &info);
// Use info.data_ptr, info.shape, info.strides, etc.
```
