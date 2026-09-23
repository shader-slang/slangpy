Compiler target selection
=========================

``SlangCompilerOptions`` supports an optional ``profile``, an optional
``capabilities`` list, and a ``capability_overrides`` dictionary. These settings
control the Slang session used to compile shaders for a device.

Every session uses the device's detected compiler inputs by default. Set a
profile, replace the input list, or override individual inputs when needed.
Explicit sessions start from fresh compiler options; they do not inherit custom
options from the device's default session.

Selecting inputs
----------------

An explicit list replaces the detected device inputs. With ``capabilities=None``,
the compiler starts from the device inputs instead. For example, on a CUDA
device supporting compute capability 7.0 or newer:

.. code-block:: python

    session = device.create_slang_session({
        "capabilities": ["cuda_sm_7_0"],
    })

To use device inputs, leave the compiler options unset:

.. code-block:: python

    session = device.create_slang_session()
    print(session.target_info.capabilities)
    print(session.target_info.ignored_capabilities)

The corresponding baseline names are ``hlsl`` for D3D12, ``spirv`` for Vulkan,
``metal`` for Metal, ``wgsl`` for WebGPU, and ``cpp`` for CPU.

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
explicit profile preserves Slang's profile semantics. For example:

.. code-block:: python

    session = d3d_device.create_slang_session({"profile": "sm_6_6"})

Higher inherited shader-model inputs are removed and reported. A higher explicit
version or a known feature requiring a higher model raises an error instead.
The same rule applies to native SPIR-V and Metal profiles. Known conflicting
features are reported even when inherited; SlangPy does not silently delete
those features. Checks cover documented version families and selected feature
dependencies, not Slang's entire capability graph.

Native D3D profiles, SPIR-V profiles on Vulkan, and Metal profiles are supported.
Stage-specific D3D profiles retain their stage restriction; prefer ``sm_*`` for
sessions containing multiple shader stages. Vulkan also accepts Slang's DX/GLSL
profile mappings when an explicit capability list is supplied. In that advanced
mode, Slang defines the additive mapping; SlangPy does not infer a native version
ceiling. CUDA, CPU, and WebGPU require ``profile=None``.

An explicit SPIR-V profile includes its feature bundle even when the capability
list is empty. A removal override cannot subtract those profile requirements.
For raw version/feature control, omit the profile and supply an authoritative
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

Selecting ``hlsl_nvapi`` instead requires an NVAPI-enabled device. Native SER is
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

The report lists inputs actually passed to Slang, not the full set of implied
capabilities or the final generated architecture. CUDA architecture can also
depend on shader code and the installed toolkit. Missing exact CUDA tiers are
rejected rather than approximated for explicit requests. Architecture/profile
flags in downstream arguments conflict with the new policy and are rejected at
both session creation and linking.

Exact CUDA architecture requests
--------------------------------

When the highest selected numeric CUDA capability comes from an explicit list
or a ``True`` override, linking compiles every entry point through the loaded
Slang/NVRTC toolchain and checks its PTX ``.target``. For example,
``capabilities=["cuda_sm_5_0"]`` fails when half-precision code emits ``sm_60``.
An architecture changed by the toolkit minimum also fails. The error includes
the requested architecture, actual target, and PTX version. Compilation failures
retain the downstream diagnostics. Session creation alone does not certify that
the toolkit accepts the target.

This validation runs again on reload and before any RHI shader-cache lookup.
Cached code for an exact request must match the validated output; a mismatch
becomes a cache miss, so an older toolchain's artifact cannot bypass the check.
It currently makes compilation eager at link time, including for deferred
pipelines, and can add compilation work even when a persistent cache contains
the shader. Exact requests require programs to be fully specialized when linked;
runtime interface specialization is rejected because SlangPy cannot inspect its
eventual output through the current RHI API. Specialize before linking or use
device-derived inputs for such programs.

An inherited highest CUDA version remains a compiler assumption. Adding a lower
version does not turn it into an exact request or remove inherited higher
versions. Device-derived and empty selections allow deferred compilation. The resolution report explains this distinction; it is a
session snapshot, so it does not contain per-program PTX results.

Matching PTX architecture does not establish a semantic capability ceiling or
guarantee driver support for the PTX ISA version. The CUDA driver still validates
the generated code when creating the pipeline. CUDA architecture suffixes and
tiers absent from Slang remain unsupported explicit inputs.

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
  ``profile="sm_6_6"``. Profile reconciliation can reject a retained feature that
  requires a higher model; remove that input explicitly or provide a capability list.
* For Vulkan, prefer native SPIR-V capabilities or profiles. An old HLSL model
  has no universal SPIR-V version equivalent. To request Slang's corresponding
  cross-family profile mapping, use ``profile="sm_6_6", capabilities=[]``;
  this requests its profile bundle and does not reproduce every legacy default.
* For CUDA, use recognized ``cuda_sm_*`` capabilities. Do not pass an NVRTC
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
