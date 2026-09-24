Compiler target selection
=========================

Use ``SlangCompilerOptions.profile`` to select a backend compilation version:
``sm_6_6`` for D3D12, ``spirv_1_6`` for Vulkan, or ``cuda_sm_7_0`` for CUDA.
An optional ``capabilities`` list and ``capability_overrides`` dictionary provide
advanced control over the inputs passed to Slang. These settings are compiler
requests and assumptions, not verified limits on the generated code.

Every session uses the device's detected compiler inputs by default. Set a
profile, replace the input list, or override individual inputs when needed.
Explicit sessions start from fresh compiler options; they do not inherit custom
options from the device's default session.

Selecting a profile
-------------------

For example, on a CUDA device supporting compute capability 7.0 or newer:

.. code-block:: python

    session = device.create_slang_session({
        "profile": "cuda_sm_7_0",
    })

A CUDA profile is SlangPy shorthand for a recognized numeric CUDA capability.
SlangPy forwards that capability to Slang, which selects the NVRTC architecture;
it does not pass a native Slang profile or append an NVRTC architecture flag.
Tiers absent from the loaded Slang registry and architecture suffixes are rejected
rather than approximated. For example, Slang 2026.18.2 does not recognize
``cuda_sm_7_5`` even on a device reporting compute capability 7.5.

To use device inputs, leave the compiler options unset:

.. code-block:: python

    session = device.create_slang_session()
    print(session.target_info.capabilities)
    print(session.target_info.ignored_capabilities)

Advanced capability inputs
--------------------------

An explicit list replaces the detected device inputs. With ``capabilities=None``,
the compiler starts from the device inputs instead. For example,
``capabilities=["cuda_sm_7_0"]`` forwards only that optional input.

The corresponding baseline names are ``hlsl`` for D3D12, ``spirv`` for Vulkan,
``cuda`` for CUDA, ``metal`` for Metal, ``wgsl`` for WebGPU, and ``cpp`` for CPU.

``capabilities=[]`` supplies no optional device inputs. The mandatory backend
baseline and any profile requirements remain. Overrides apply after selecting
the base: ``True`` adds an input and ``False`` removes it. Removing a lower
version does not remove higher versions or features implying it. Adding a lower
version does not lower a higher version already selected.

With empty inputs and no explicit profile, the compatibility baseline is
``sm_6_0`` on D3D12 and ``spirv_1_0`` on Vulkan. Other backends receive no profile;
their compiler/toolkit defaults still apply.

Unknown explicit names fail at session creation. Detected names unknown to the
loaded compiler appear in ``session.target_info.ignored_capabilities``. Removing
an unknown detected input is allowed; misspelled removal names are rejected.
Equivalent numeric shader-model aliases on D3D12 and CUDA aliases on CUDA are
normalized. SPIR-V public aliases and raw atoms stay distinct: ``spirv_1_6`` adds
features beyond the raw ``_spirv_1_6`` version.

Profiles and conflicts
----------------------

An omitted profile is selected automatically when the backend needs one. An
explicit native profile preserves Slang's profile semantics; CUDA shorthand
supplies the corresponding capability. For example:

.. code-block:: python

    session = d3d_device.create_slang_session({"profile": "sm_6_6"})

Higher inherited numeric version inputs are removed and reported. A higher
explicit numeric version, including a ``True`` override of a detected version,
raises an error instead. The same rule applies to native SPIR-V, Metal, and CUDA
selectors. Direct version requests above the detected device version also fail.
General feature dependencies are left to Slang and downstream compilers; retained
features may raise output requirements above the selected profile. SlangPy keeps
one narrow feature adapter: native D3D12 SER requires shader model 6.9, so its
markers raise the automatic D3D profile and conflict with a lower explicit profile.

Native D3D profiles, SPIR-V profiles on Vulkan, and Metal profiles are supported.
Stage-specific D3D profiles retain their stage restriction; prefer ``sm_*`` for
sessions containing multiple shader stages. Vulkan also accepts Slang's DX/GLSL
profile mappings when an explicit capability list is supplied. In that advanced
mode, Slang defines the additive mapping; SlangPy does not infer a native version
ceiling. CPU and WebGPU require ``profile=None``.

An explicit SPIR-V profile includes its feature bundle even when the capability
list is empty. A removal override cannot subtract those profile requirements.
For CUDA, the selected profile similarly supplies its capability even after a
removal override. For raw version/feature control, omit the profile and supply an authoritative
list; automatic Vulkan selection uses a minimal compatibility profile.

For native D3D12 SER on a device with the required API support, remove NVAPI
and select the native implementation:

.. code-block:: python

    session = d3d_device.create_slang_session({
        "capability_overrides": {
            "hlsl_nvapi": False,
            "ser_hlsl_native": True,
        },
    })

To select NVAPI SER on an NVAPI-enabled device, use for example
``profile="sm_6_6", capabilities=["hlsl_nvapi"]``. With device inputs instead,
remove any retained native SER markers explicitly when selecting a lower profile.
Selecting ``hlsl_nvapi`` requires an NVAPI-enabled device. Native SER is
not implied merely by a shader-model profile. The compiler may warn about the
other implementation due to a known SER requirement-inference issue.

Observing the resolution
------------------------

``session.target_info`` returns a read-only snapshot containing the output target,
requested and resolved profile, base inputs, forwarded
inputs and their origins, removals, ignored detections, generated SlangPy
downstream arguments, compatibility notes, and session digest. Collections are
copies; modifying them does not change the session. ``session.desc`` also returns
a copy. Reloading preserves the requested options and resolves them again.

``requested_profile`` preserves the caller's selector. ``profile`` is the actual
native profile passed to Slang and remains ``None`` for CUDA, including an explicit
CUDA selector. ``profile_automatic`` indicates that the caller omitted the option.
A capability synthesized for a CUDA profile has origin ``"profile"``; an existing
equivalent input retains its ``"device"``, ``"explicit"``, or ``"override"`` origin.

The report lists inputs actually passed to Slang, not the full set of implied
capabilities or the final generated architecture. CUDA architecture can also
depend on shader code and the installed toolkit. Missing exact CUDA tiers are
rejected rather than approximated for explicit requests. Architecture/profile
flags in downstream arguments conflict with the new policy and are rejected at
both session creation and linking.

CUDA compilation behavior
-------------------------

All CUDA selections are compiler assumptions, regardless of whether they came
from a profile, explicit list, override, or device detection. Shader requirements
and the downstream toolkit can raise the emitted architecture. For example,
Slang 2026.18.2 with NVRTC 12.2 can emit ``sm_60`` for half-precision code requested
at ``cuda_sm_5_0``, or ``sm_50`` for a request below the toolkit minimum.

Explicit selections use ordinary RHI compilation, runtime specialization, deferred
pipelines, persistent caches, and hot reload. Shader, downstream compiler, and
driver failures still surface through the normal paths. Session creation does not
certify toolkit support or driver acceptance of the generated PTX ISA version.

This intentionally replaces the earlier exact-CUDA contract: SlangPy no longer
compiles eagerly at link time to inspect PTX, requires full specialization at
link time, or compares cached code with freshly generated output. There is no
additional guarantee against downstream toolkit changes or stale cached artifacts.
The target report describes session inputs, not per-program output.

Capability-upgrade warnings (41012) are visible by default. Compilation remains
permissive: use ``warnings_as_errors=["41012"]`` to reject those upgrades where
appropriate, or ``disable_warnings=["41012"]`` to suppress the warning for an
affected workload. SER inference and incomplete device reporting can produce
warnings for valid programs. Profiles, removal overrides, and restrictive
capability checks do not establish a universal semantic capability ceiling.

Migrating from shader models
----------------------------

This is a breaking API change. ``SlangCompilerOptions.shader_model``,
``ShaderModel``, ``Device.supported_shader_model``, and the transitional
``SlangTargetInfo.legacy`` field have been removed.

* For ordinary compilation, omit target options to use the device's inputs.
* Replace a D3D12 ``shader_model=ShaderModel.sm_6_6`` setting with
  ``profile="sm_6_6"``. Profile reconciliation rejects higher explicit numeric models and retained
  native SER markers requiring SM 6.9; remove those inputs or select a suitable profile.
* For Vulkan, prefer native SPIR-V capabilities or profiles. An old HLSL model
  has no universal SPIR-V version equivalent. To request Slang's corresponding
  cross-family profile mapping, use ``profile="sm_6_6", capabilities=[]``;
  this requests its profile bundle and does not reproduce every legacy default.
* For CUDA, use ``profile="cuda_sm_7_0"`` or other recognized numeric CUDA tiers. Do not pass an NVRTC
  architecture flag alongside the compiler's generated flag.
* Inspect ``Device.capabilities`` for reported inputs and
  ``session.target_info`` for the selected profile and forwarded inputs.
  Neither is a complete capability-implication query.

The legacy ``__SHADER_TARGET_MAJOR`` and ``__SHADER_TARGET_MINOR`` shader macros
are deprecated compatibility defines required by NVAPI headers. They describe
the selected D3D12 profile and are zero on other backends. Prefer backend-specific
capability requirements and the existing ``__TARGET_D3D12__``,
``__TARGET_VULKAN__``, ``__TARGET_CUDA__``, etc. defines when backend-specific
source is needed. The old SM 6.7-to-6.6 default cap
is gone; D3D12 now derives its automatic profile from detected compiler inputs.

For D3D12 profiles 6.7 and later, SlangPy currently adds DXC's
``-disable-payload-qualifiers`` compatibility option because Slang can omit
required ray-payload annotations for separately compiled miss/hit shaders. This
preserves the selected profile; ``target_info.generated_downstream_args`` and
``notes`` disclose the workaround. To use payload qualifiers, set
``downstream_args=["-enable-payload-qualifiers"]`` on the session and supply the
required annotations. A conflicting link-only enable option is rejected.
