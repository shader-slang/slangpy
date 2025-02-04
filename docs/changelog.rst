Changelog
---------

**Version 0.19.1**
- Update SGL -> 0.12.2
- Fix major issue with texture transposes

**Version 0.19.0**
- Add experimental grid type

**Version 0.18.2**
- Update SGL -> 0.12.1
- Rename from_numpy to buffer_from_numpy

**Version 0.18.1**
- Fix Python 3.9 typing

**Version 0.18.0**

- Long file temp filenames fix 
- Temp fix for resolution of types that involve generics in multiple files 
- Support passing 1D NDBuffer to structured buffer 
- Fix native buffer not being passed to bindings 
- Missing slang field check 
- Avoid synthesizing store methods for none-written nested types

**Version 0.17.0**

- Update to latest `nv-sgl` with CoopVec support
- Native tensor implementation
- Linux crash fix

**Version 0.16.0**

- Native texture and structured buffer implementations
- Native function dispatches
- Lots of bug fixes

**Version 0.15.2**

- Correctly package slang files in wheel

**Version 0.15.0**

- Native buffer takes full reflection layout
- Add uniforms + cursor api to native buffer
- Update required version of `nv-sgl` to `0.9.0`

**Version 0.14.0**

- Update required version of `nv-sgl` to `0.8.0`
- Substantial native + python optimizations

**Version 0.13.0**

- Update required version of `nv-sgl` to `0.7.0`
- Native SlangPy backend re-enabled 
- Conversion of NDBuffer to native code 
- PyTorch integration refactor

**Version 0.12.0**

- Update required version of `nv-sgl` to `0.6.2`
- Re-enable broken Vulkan tests

**Version 0.12.0**

- Update required version of `nv-sgl` to `0.6.1`

**Version 0.10.0**

- Initial test release
