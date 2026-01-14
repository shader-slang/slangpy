# slangpy-torch

Minimal PyTorch native extension providing fast (~28ns) tensor metadata access
from native code without Python API overhead (~350ns).

## Install

```bash
pip install slangpy-torch
```

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
