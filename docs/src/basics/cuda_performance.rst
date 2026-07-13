CUDA Performance Notes
======================

SlangPy runs the same Slang source across every backend (D3D12, Vulkan, Metal, and CUDA), so the
performance characteristics of your kernels depend on how the Slang compiler lowers that source for
each target. This page documents a **current Slang code-generation limitation** — one that is
expected to be resolved upstream — which can make otherwise identical code run several times slower on
the CUDA backend than on Vulkan or D3D12.

Multi-component swizzle reads in hot loops
------------------------------------------

On the **CUDA backend**, a hot loop that reads a multi-component swizzle (``.rgb`` / ``.xyz``) from
an expensive expression can run roughly 3x slower than the mathematically identical code that reads a
whole ``float4`` or accumulates in individual scalars. Vulkan and D3D12 show no such penalty — the
difference is in how each backend's emitter lowers the swizzle.

The cause is in the Slang compiler, not in SlangPy (SlangPy is transparent to vector types and does
not rewrite them). Slang's C-family emitter — shared by the **CUDA and CPU/C++ targets** — lowers a
multi-component swizzle read to a per-component brace initializer, e.g.
``float3{ base.x, base.y, base.z }``. This **re-evaluates the base expression once per component**.
When the base is a folded texture or buffer load, that load is emitted three times, so a single
``.rgb`` read becomes three fetches (verified fetch counts for the reported kernels: a whole
``float4`` read = 1 fetch; the ``.rgb`` read inside the loop = 3 fetches, matching the observed
~2.9x). The penalty tracks the **swizzle**, not the ``float3`` type or its 12-byte layout.

Backends whose emitters read the base only once do not have this problem: SPIR-V (Vulkan) uses a
single ``OpVectorShuffle`` and HLSL (D3D12) emits a native ``.xyz`` accessor. That is why the
slowdown is asymmetric across backends.

This is a Slang compiler limitation, not intended behaviour, and the fix belongs in the code
generator: the emitter should evaluate the swizzle base once — as SPIR-V and HLSL already do —
instead of re-emitting it per component. It is tracked upstream as `shader-slang/slang#12073
<https://github.com/shader-slang/slang/issues/12073>`_. Until that fix lands, the workarounds below
are the recommended mitigation; they should become unnecessary once the compiler is fixed.

Recognising the pattern
-----------------------

Treat this as a *pattern class*, not a single exact snippet. The trigger is **a multi-component
swizzle whose base is a non-trivial, folded expression** — a texture/buffer fetch, a helper-function
call, or something like ``saturate(...)`` — read in a hot path. It is *not* the ``float3`` type
itself: a swizzle of a cheap, register-resident local is fine, which is exactly why binding the value
to a named local before swizzling removes the penalty (and why a minimal reproduction that keeps the
value in a small local may not exhibit it, while the same operation over a helper result in a larger
kernel does). If a CUDA kernel is unexpectedly slow and reads ``.rgb`` / ``.xyz`` from a fetch or a
helper result in an inner loop, suspect this first.

Workarounds
-----------

All of these keep the math identical while avoiding a folded expression sitting directly under a
multi-component swizzle:

**Accumulate in float4 or in named scalars, not by swizzling the fetch.**

.. code-block::

    // Slow: the .rgb swizzle re-reads tex[coord(i)] once per component (3 fetches per iteration).
    float3 sum3 = float3(0);
    for (int i = 0; i < N; ++i)
        sum3 += tex[coord(i)].rgb;

    // Fast: read the whole texel once per iteration (the .a lane is simply unused) ...
    float4 sum4 = float4(0);
    for (int i = 0; i < N; ++i)
        sum4 += tex[coord(i)];

    // ... or bind the fetch to a local first, then read its components (one fetch per iteration).
    float sr = 0, sg = 0, sb = 0;
    for (int i = 0; i < N; ++i)
    {
        float4 s = tex[coord(i)];
        sr += s.r; sg += s.g; sb += s.b;
    }

**Read the full float4 into a local and index its components, instead of swizzling the fetch.**

.. code-block::

    // Prefer binding the fetch to a local, then reading .r / .g / .b (one fetch) ...
    float4 s = tex[q];
    float lum = 0.299 * s.r + 0.587 * s.g + 0.114 * s.b;

    // ... over swizzling the fetch directly, which re-reads tex[q] once per component.
    float3 rgb = tex[q].rgb;   // avoid in inner loops

**Write results via per-lane assignment rather than a constructor swizzle.**

.. code-block::

    // Slow: the .rgb swizzle re-evaluates shade(uv) once per component (3 calls).
    dst[tid] = float4(shade(uv).rgb, a);

    // Fast: evaluate shade(uv) once into a local, overwrite the differing lane, and store.
    float4 v = shade(uv);
    v.a = a;
    dst[tid] = v;

Confirming the emitted code
---------------------------

If you want to inspect the generated CUDA C++ for one of these patterns, note that
``SLANGPY_PRINT_GENERATED_SHADERS=1`` dumps the *Slang wrapper* SlangPy generates, **not** the CUDA
C++ that NVRTC ultimately compiles. To see the actual emission, compile a standalone Slang snippet
with ``slangc -target cuda`` (or ``-target cpp`` — it shares the same C-family emitter) and look at
how each ``.rgb`` / ``.xyz`` read is expanded: a folded base under the swizzle is emitted once per
component, so you can count how many times an expensive load appears in the output.
