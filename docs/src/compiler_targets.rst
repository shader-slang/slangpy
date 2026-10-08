Compiler profiles
=================

Use ``SlangCompilerOptions.profile`` to select a compilation target. Leave it
unset (``None``) to select automatically from the device and compiler support.

.. list-table:: Profile examples
    :header-rows: 1

    * - Backend
      - Profiles
    * - D3D12
      - ``sm_6_6``, ``sm_6_9``
    * - Vulkan
      - ``spirv_1_3``, ``spirv_1_6``
    * - Metal
      - ``metallib_2_4``
    * - CUDA
      - ``compute_75``, ``compute_86``, ``compute_120``
    * - CPU, WebGPU
      - ``None`` only

.. code-block:: python

    import slangpy as spy

    device = spy.Device(type=spy.DeviceType.vulkan)
    session = device.create_slang_session({"profile": "spirv_1_3"})

Choosing a profile
------------------

Automatic selection uses device capabilities recognized by Slang. An explicit
profile uses its own requirements, plus backend and runtime integrations; optional
device features do not silently raise the target. Unknown or incompatible
profiles fail at session creation. Native D3D12, SPIR-V, and Metal profile versions
are checked against the device's reported version when available.

Prefer ``sm_*`` on D3D12 for sessions containing multiple shader stages.
Vulkan also accepts Slang's DX/GLSL mappings, such as ``sm_6_6`` and ``glsl_460``;
use ``spirv_*`` to select a SPIR-V version directly.

Native Slang profiles are compiler assumptions, not strict feature limits:
shader requirements can raise the generated target. Set
``warnings_as_errors=["41012"]`` to reject upgrades reported by Slang. Requirement
inference has gaps, so this is not a complete compatibility check.
``Device.capabilities`` describes the device; sessions do not expose capability
selection or a resolved capability list.

CUDA profiles
-------------

SlangPy supplies synthetic CUDA profiles because Slang has no native equivalents.
Names use ``compute_<major * 10 + minor>``: ``compute_120`` means compute capability
12.0. Architecture suffixes such as ``compute_90a`` are unsupported.

With ``profile=None``, SlangPy queries the NVRTC library selected by Slang and
chooses its highest supported architecture that does not exceed the device's
compute capability. An explicit profile must be supported by both; it is never
rounded or clamped. Failure to query NVRTC or find a compatible architecture is
an error.

The profile sets NVRTC's exact ``--gpu-architecture`` argument and the CUDA
capabilities Slang recognizes up to that target. A shader requiring a higher
architecture fails compilation. OptiX cooperative-vector support is included only
for targets 9.0 or later when the device supports the integration.

The ``SGL_MAX_CUDA_COMPUTE_CAPABILITY`` environment variable limits the maximum
CUDA compute capability (e.g. ``90`` for 9.0). Automatic selection respects this
limit; explicit profiles above it fail. Unset or empty values disable the limit.

D3D12 integration
-----------------

SlangPy preserves NVAPI-based shader execution reordering (SER) where supported,
including at shader model 6.9 and later. Changing the profile does not select
native HLSL SER, whose Slang APIs differ.

At shader model 6.7 and later, SlangPy adds DXC's ``-disable-payload-qualifiers``
because Slang can omit required annotations on separately compiled miss/hit
shaders. To enable qualifiers, supply the annotations and set
``downstream_args=["-enable-payload-qualifiers"]`` on the session. An explicit
enable or disable option takes precedence.

Session options
---------------

New sessions start with fresh compiler options; they do not inherit custom options
from the device's default session. ``session.desc`` returns a copy of the requested
settings. Resolved targets participate in shader caching, and hot reload preserves
the session settings.

``downstream_args`` passes raw arguments to DXC or NVRTC. Callers must keep them
compatible with the profile: overriding the downstream target does not update
Slang's compiler assumptions.

Migrating from shader models
----------------------------

``SlangCompilerOptions.shader_model``, ``ShaderModel``, and
``Device.supported_shader_model`` have been removed. Replace
``shader_model=ShaderModel.sm_6_6`` with ``profile="sm_6_6"`` on D3D12. For Vulkan,
prefer a SPIR-V profile, or retain Slang's mapping with ``profile="sm_6_6"``.

The deprecated ``__SHADER_TARGET_MAJOR`` and ``__SHADER_TARGET_MINOR`` macros
describe the D3D12 profile and are zero elsewhere. Prefer backend defines such as
``__TARGET_D3D12__`` and ``__TARGET_VULKAN__`` for backend-specific source.
