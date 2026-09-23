Compiler target selection
=========================

``SlangCompilerOptions`` supports an optional ``profile``, an optional
``capabilities`` list, and a ``capability_overrides`` dictionary. These settings
control the Slang session used to compile shaders for a device.

During the transition from ``shader_model``, the new policy is opt-in: set a
profile, supply a capability list (including an empty one), or use a nonempty
override dictionary. Leaving all three at their defaults retains the legacy
policy. A non-default ``shader_model`` cannot be combined with the new options.

Selecting inputs
----------------

An explicit list replaces the detected device inputs. With ``capabilities=None``,
the new policy starts from the device inputs instead. For example, on a CUDA
device supporting compute capability 7.0 or newer:

.. code-block:: python

    session = device.create_slang_session({
        "capabilities": ["cuda_sm_7_0"],
    })

To use CUDA device inputs during the transition, a neutral override activates
the new policy:

.. code-block:: python

    session = device.create_slang_session({
        "capability_overrides": {"cuda": True},
    })

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

On the new path, the predefined ``__SHADER_TARGET_MAJOR`` and
``__SHADER_TARGET_MINOR`` macros describe the resolved D3D profile. Both are zero
on other backends. Their legacy values remain unchanged on the legacy path.

Observing the resolution
------------------------

``session.target_info`` returns a read-only snapshot containing the output target,
legacy-policy indicator, requested and resolved profile, base inputs, forwarded
inputs and their origins, removals, ignored detections, generated SlangPy
downstream arguments, compatibility notes, and session digest. Collections are
copies; modifying them does not change the session. ``session.desc`` also returns
a copy. Reloading preserves the requested options and resolves them again.

The report lists inputs actually passed to Slang, not the full set of implied
capabilities or the final generated architecture. CUDA architecture can also
depend on shader code and the installed toolkit. Missing exact CUDA tiers are
rejected rather than approximated for explicit requests. Complete toolkit and
emitted-architecture validation remains future work. Architecture/profile flags
in downstream arguments conflict with the new policy and are rejected at both
session creation and linking.

The existing warning policy remains in effect during this transition. Neither
profiles, removal overrides, nor restrictive capability checks establish a
universal hardware or compiler capability ceiling.
