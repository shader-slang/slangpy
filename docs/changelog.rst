.. _changelog:

.. cpp:namespace:: sgl

Changelog
=========

SlangPy uses a `semantic versioning <http://semver.org>`__ policy for its API.

Version 0.37.0 (October 15, 2025)
-------

- Update to Slang version 2025.19 with latest shader compilation improvements and bug fixes.
  (PR `#572 <https://github.com/shader-slang/slangpy/pull/572>`__, PR `#560 <https://github.com/shader-slang/slangpy/pull/560>`__)
- Update slang-rhi submodule to latest version with improved stability and bug fixes.
  (PR `#569 <https://github.com/shader-slang/slangpy/pull/569>`__, PR `#550 <https://github.com/shader-slang/slangpy/pull/550>`__, PR `#541 <https://github.com/shader-slang/slangpy/pull/541>`__)
- Add persistent shader cache implementation based on LMDB for improved compilation performance and caching across sessions.
  (PR `#561 <https://github.com/shader-slang/slangpy/pull/561>`__, PR `#555 <https://github.com/shader-slang/slangpy/pull/555>`__)
- Implement string printing support in shaders for improved debugging capabilities.
  (PR `#566 <https://github.com/shader-slang/slangpy/pull/566>`__)
- Add support for calling interface parameters with implementing types.
  (PR `#562 <https://github.com/shader-slang/slangpy/pull/562>`__)
- Add nanothread library and improve threading support.
  (PR `#563 <https://github.com/shader-slang/slangpy/pull/563>`__)
- Fix import determinism to ensure consistent code generation for shader cache compatibility.
  (PR `#565 <https://github.com/shader-slang/slangpy/pull/565>`__)
- Fix texture loader for CUDA and improve platform compatibility.
  (PR `#545 <https://github.com/shader-slang/slangpy/pull/545>`__, PR `#552 <https://github.com/shader-slang/slangpy/pull/552>`__)
- Fix compute blit functionality and various bug fixes.
  (PR `#503 <https://github.com/shader-slang/slangpy/pull/503>`__, PR `#546 <https://github.com/shader-slang/slangpy/pull/546>`__, PR `#554 <https://github.com/shader-slang/slangpy/pull/554>`__, PR `#553 <https://github.com/shader-slang/slangpy/pull/553>`__)

Version 0.36.0 (September 30, 2025)
-------

- Update to Slang version 2025.18 with latest shader compilation improvements and bug fixes.
- Update slang-rhi submodule to latest version with improved dependency handling.
  (PR `#533 <https://github.com/shader-slang/slangpy/pull/533>`__)

Version 0.35.0 (September 18, 2025)
-------

- Add initial support for ray tracing pipelines, enabling hardware-accelerated ray tracing workflows.
  (PR `#502 <https://github.com/shader-slang/slangpy/pull/502>`__)
- Update to latest Slang version (2025.17) with improved shader compilation and platform support.
  (PR `#507 <https://github.com/shader-slang/slangpy/pull/507>`__)
- Add helper function to create homogeneous 4x4 transformation matrices from 3x4 matrices.
  (PR `#506 <https://github.com/shader-slang/slangpy/pull/506>`__)
- Add new ``load_from_file`` and ``load_from_numpy`` functions for improved data loading workflows.
  (PR `#513 <https://github.com/shader-slang/slangpy/pull/513>`__)
- Fix hot reload functionality for built-in reflection data to ensure proper shader recompilation.
  (PR `#514 <https://github.com/shader-slang/slangpy/pull/514>`__)
- Fix memory stream loading issues and improve data loading reliability.
  (PR `#513 <https://github.com/shader-slang/slangpy/pull/513>`__)
- Rename getter methods throughout the API to follow consistent coding conventions.
  (PR `#505 <https://github.com/shader-slang/slangpy/pull/505>`__)

Version 0.34.0 (September 9, 2025)
-------

- Add ``Device.report_heaps()`` method to query internal memory heap status and allocation information.
- Update to latest Slang version (2025.16.0) with improved CUDA and Metal support.
  (PR ```#493 <https://github.com/shader-slang/slangpy/pull/493>```__)
- Add GPU clock locking for consistent benchmark results and implement trimmed mean calculation for more accurate performance measurements.
  (PR ```#484 <https://github.com/shader-slang/slangpy/pull/484>```__, PR ```#480 <https://github.com/shader-slang/slangpy/pull/480>```__, PR ```#472 <https://github.com/shader-slang/slangpy/pull/472>```__)
- Support passing call data as entry point parameters on CUDA for improved performance.
  (PR ```#481 <https://github.com/shader-slang/slangpy/pull/481>```__)
- Fix multiple memory leaks related to Python object references and improve resource cleanup.
  (PR ```#488 <https://github.com/shader-slang/slangpy/pull/488>```__)
- Add benchmark comparison and delta reporting functionality with GPU information in reports.
  (PR ```#471 <https://github.com/shader-slang/slangpy/pull/471>```__, PR ```#456 <https://github.com/shader-slang/slangpy/pull/456>```__)
- Rename ```command_buffer``` to ```command_encoder``` for API consistency.
  (PR ```#487 <https://github.com/shader-slang/slangpy/pull/487>```__)
- Add ```PassEncoder::write_timestamp``` and timestamp support in ```ComputeKernel::dispatch```.
  (PR ```#473 <https://github.com/shader-slang/slangpy/pull/473>```__)
- Optimize ```write_from_numpy``` performance with faster copy options.
  (PR ```#455 <https://github.com/shader-slang/slangpy/pull/455>```__)
- Fix PyTorch examples and improve integration.
  (PR ```#459 <https://github.com/shader-slang/slangpy/pull/459>```__)
- Add support for platform-specific test isolation via environment variables.
  (PR ```#478 <https://github.com/shader-slang/slangpy/pull/478>```__)
- Fix module linking for layout when using ```link``` modules.
  (PR ```#449 <https://github.com/shader-slang/slangpy/pull/449>```__)
- Add string conversion functions for slangpy types and improve debugging capabilities.
  (PR ```#463 <https://github.com/shader-slang/slangpy/pull/463>```__, PR ```#464 <https://github.com/shader-slang/slangpy/pull/464>```__)

Version 0.33.1 (August 25, 2025)
----------------------------

- Include the missing Slang binary file into the package.
  (PR `#445 <https://github.com/shader-slang/slangpy/pull/445>`__)
- Introduce benchmark plugin and testing infrastructure with MongoDB integration for automated performance tracking.
  (PR `#452 <https://github.com/shader-slang/slangpy/pull/452>`__)
- Add support for bindless storage buffers in GPU abstraction layer.
  (PR `#421 <https://github.com/shader-slang/slangpy/pull/421>`__).
- Fix ``copy_from_torch()`` for CUDA devices and resolve PyTorch integration issues.
  (PR `#391 <https://github.com/shader-slang/slangpy/pull/391>`__).
- Introduce unified ``slangpy.testing`` module consolidating all testing utilities and pytest plugin system.
  (PR `#448 <https://github.com/shader-slang/slangpy/pull/448>`__).
- Force release all slang-rhi resources during shutdown to prevent memory leaks and segfaults on Linux.
  (PR `#426 <https://github.com/shader-slang/slangpy/pull/426>`__).
- Rename ``DeviceResource`` to ``DeviceChild`` for consistency with slang-rhi.
  (PR `#425 <https://github.com/shader-slang/slangpy/pull/425>`__).
- Enable more tests across platforms: Linux, CUDA, and Metal support improvements.
  (PR `#429 <https://github.com/shader-slang/slangpy/pull/429>`__).
- Fix race condition in hot reload test and improve shader change detection.
  (PR `#433 <https://github.com/shader-slang/slangpy/pull/433>`__).
- Force unroll small fixed size loops and globally disable warning 30856 for better compilation.
  (PR `#437 <https://github.com/shader-slang/slangpy/pull/437>`__).

Version 0.33.0 (August 12, 2025)
----------------------------

- Update to slang version 2025.14.3.
  (PR `#409 <https://github.com/shader-slang/slangpy/pull/409>`__).
- Fix tensor alignment issue when copying data to GPU tensors with vector element types.
  Metal platform now handles vector alignment correctly to match other platforms.
  (PR `#418 <https://github.com/shader-slang/slangpy/pull/418>`__).
- Update samples.
  (PR `#413 <https://github.com/shader-slang/slangpy/pull/413>`__).

Version 0.32.0 (August 8, 2025)
----------------------------

- Update to slang version 2025.14.
- Improve CUDA support.
- Improve Metal support.
- Improve PyTorch support.
  (PR `#362 <https://github.com/shader-slang/slangpy/pull/362>`__).
- Add support for pointers.
  (PR `#323 <https://github.com/shader-slang/slangpy/pull/323>`__, PR `#326 <https://github.com/shader-slang/slangpy/pull/326>`__).
- Add ``SGL_SLANG_DEBUG_INFO`` cmake variable to enable downloading Slang debug info (enabled by default).
  (PR `#296 <https://github.com/shader-slang/slangpy/pull/296>`__).
- Add ``sgl::CommandEncoder::generate_mips()`` (``slangpy.CommandEncoder.generate_mips()``) to generate mipmaps for textures.
  (PR `#293 <https://github.com/shader-slang/slangpy/pull/293>`__).
- Add optional ``_append_to`` argument to slangpy call functions to append commands to an existing command encoder.
  (PR `#287 <https://github.com/shader-slang/slangpy/pull/287>`__).
- Allow creating ``Bitmap`` from non-contiguous arrays.
  (PR `#282 <https://github.com/shader-slang/slangpy/pull/282>`__).

Version 0.31.0 (June 5, 2025)
----------------------------

- Update to slang version 2025.10.1.
- Add support for vectorizing against Python lists.
- Make ``NDBuffer`` and ``Tensor`` ``empty`` / ``zeros`` APIs consistent.
- Added ``load_from_image`` for ``NDBuffer`` and ``Tensor``.
- Fix typings for ``float2x3``, ``float3x2``, ``float4x2`` and ``float4x3``.

Version 0.30.0 (May 27, 2025)
----------------------------

- Update `slang-rhi` to latest version.
  Improve CUDA error reporting.
  Improve debug marker support and add `WinPixEventRuntime`.
  Fix resource lifetime tracking for entry point arguments.
  (PR `#236 <https://github.com/shader-slang/slangpy/pull/236>`__).

Version 0.29.0 (May 22, 2025)
----------------------------

- Update `slang-rhi` to latest version. Make enum infos constexpr.
  (PR `#234 <https://github.com/shader-slang/slangpy/pull/234>`__).
- Fix ``sgl::Feature`` (``slangpy.Feature``) to include missing value.
  (PR `#233 <https://github.com/shader-slang/slangpy/pull/233>`__).
- Fix registered matrix types in ``PYTHON_TYPES``.
  (PR `#232 <https://github.com/shader-slang/slangpy/pull/232>`__).

Version 0.28.0 (May 21, 2025)
----------------------------

- Load PyTorch module lazily to avoid overhead when PyTorch is not used.
  (PR `#184 <https://github.com/shader-slang/slangpy/pull/184>`__).
- Improve warning when tev image viewer is not running.
  (PR `#216 <https://github.com/shader-slang/slangpy/pull/216>`__).
- Report correct LUID in ``sgl::DeviceInfo::adapter_luid`` (``slangpy.DeviceInfo.adapter_luid``).
  (PR `#215 <https://github.com/shader-slang/slangpy/pull/215>`__).


Version 0.27.0 (May 9, 2025)
----------------------------

- Package and distribute pytest tests. Fix deploying ``.pyi`` files in wheels + other minor fixes.
  (PR `#197 <https://github.com/shader-slang/slangpy/pull/197>`__).
- Introduce basic support for bindless textures and samplers. Currently only supported on D3D12.
  Add ``sgl::Feature::bindless`` (``slangpy.Feature.bindless``) to detect bindless support.
  Add ``sgl::DescriptorHandle`` (``slangpy.DescriptorHandle``) to represent bindless descriptor handles.
  Add ``sgl::Sampler::descriptor_handle()`` (``slangpy.Sampler.descriptor_handle``) to get the descriptor handle for a sampler.
  Add ``sgl::Texture::descriptor_handle_ro()`` (``slangpy.Texture.descriptor_handle_ro``) to get the read-only descriptor handle for a texture.
  Add ``sgl::Texture::descriptor_handle_rw()`` (``slangpy.Texture.descriptor_handle_rw``) to get the read-write descriptor handle for a texture.
  (PR `#196 <https://github.com/shader-slang/slangpy/pull/196>`__).
- Rename ``sgl::Struct`` to ``sgl::DataStruct`` to match ``slangpy.DataStruct``.
  Rename ``sgl::StructConverter`` to ``sgl::DataStructConverter``
  and ``slangpy.StructConverter`` to ``slangpy.DataStructConverter``.
  (PR `#185 <https://github.com/shader-slang/slangpy/pull/185>`__).


Version 0.26.0
----------------------------

- Port samples to use new combined SlangPy/SGL API
- CUDA and Metal fixes
- Initial deployment of wheels for macOS


Version 0.25.0
----------------------------

- Fix deploying slangpy shader files


Version 0.24.0
----------------------------

- Merge SGL (https://github.com/shader-slang/sgl) into SlangPy.

Version 0.23.0
----------------------------

- Require SGL v0.15.0
- Refactor of NDBuffer and Tensor to share some underlying type
- NDBuffer and Tensor support indexing

Version 0.22.0
----------------------------

- Requre new SGL v0.14.0 with switch to Slang-RHI

Version 0.21.1
----------------------------

- Fix to numpy version requirement
- Fixes to examples
- Add neural network example
- Require SGL v0.13.1

Version 0.21.0
----------------------------

- Full Jupyter notebook support
- Lots of fixes for edge-case hot reload crashes
- Significantly more robust wang hash and rand float generators
- Direct return of structs from scalar calls
- Add diff splatting sample
- Fix for rare issue involving lookup order of generic functions vs generic types
- Require SGL v0.13.0

Version 0.20.1
----------------------------

- Fix scalar wang-hash arg types

Version 0.20.0
----------------------------

- Add SDF example
- Transpose vector coordinates

Version 0.19.5
----------------------------

- Documentation for generators
- Extra fixes for grid

Version 0.19.4
----------------------------

- Fix grid issue

Version 0.19.3
----------------------------

- Update SGL -> 0.12.4
- Significant improvements to generator types
- Support textures as output type

Version 0.19.2
----------------------------

- Update SGL -> 0.12.3
- Better error messages during generation
- Fix corrupt error tables
- Restore detailed error information during dispatch

Version 0.19.1
----------------------------

- Update SGL -> 0.12.2
- Fix major issue with texture transposes

Version 0.19.0
----------------------------

- Add experimental grid type

Version 0.18.2
----------------------------

- Update SGL -> 0.12.1
- Rename from_numpy to buffer_from_numpy

Version 0.18.1
----------------------------

- Fix Python 3.9 typing

Version 0.18.0
----------------------------

- Long file temp filenames fix
- Temp fix for resolution of types that involve generics in multiple files
- Support passing 1D NDBuffer to structured buffer
- Fix native buffer not being passed to bindings
- Missing slang field check
- Avoid synthesizing store methods for none-written nested types

Version 0.17.0
----------------------------

- Update to latest `nv-sgl` with CoopVec support
- Native tensor implementation
- Linux crash fix

Version 0.16.0
----------------------------

- Native texture and structured buffer implementations
- Native function dispatches
- Lots of bug fixes

Version 0.15.2
----------------------------

- Correctly package slang files in wheel

Version 0.15.0
----------------------------

- Native buffer takes full reflection layout
- Add uniforms + cursor api to native buffer
- Update required version of `nv-sgl` to `0.9.0`

Version 0.14.0
----------------------------

- Update required version of `nv-sgl` to `0.8.0`
- Substantial native + python optimizations

Version 0.13.0
----------------------------

- Update required version of `nv-sgl` to `0.7.0`
- Native SlangPy backend re-enabled
- Conversion of NDBuffer to native code
- PyTorch integration refactor

Version 0.12.0
----------------------------

- Update required version of `nv-sgl` to `0.6.2`
- Re-enable broken Vulkan tests

Version 0.12.0
----------------------------

- Update required version of `nv-sgl` to `0.6.1`

Version 0.10.0
----------------------------

- Initial test release
