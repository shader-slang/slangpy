.. _index_representation:

Index Representation Conventions
=================================

SlangPy supports two different coordinate conventions for indexing multi-dimensional data: **array coordinates** (following ML conventions) and **vector coordinates** (following graphics conventions). Understanding these conventions is essential when working with buffers, tensors, generators, and multi-dimensional indexing operations.

This dual convention system exists because of a fundamental conflict between how machine learning and graphics communities think about data layout.
SlangPy follows the ML (NumPy/PyTorch) convention by default, where the **rightmost dimension has the smallest stride**.
However, this creates a problem when dealing with classic graphics primitives like textures, where graphics APIs and shader languages are typically designed around the **'x' component representing the smallest stride**.
For example, a texture that is 512 pixels wide and 128 pixels high would be represented in ML conventions as shape ``[128, 512]`` with strides ``[512, 1]``, but graphics APIs typically expect the x component (width) to have the smallest stride.

Array Coordinates (ML Convention)
----------------------------------

Array coordinates follow the convention used by NumPy, PyTorch, and similar libraries:

- **Rightmost dimension has smallest stride** (row-major storage)
- **Indexing**: ``array[dim0, dim1, dim2, ...]`` where the rightmost index varies fastest
- **Memory layout**: Elements are stored so consecutive memory locations correspond to incrementing the rightmost coordinate

Vector Coordinates (Graphics Convention)
-----------------------------------------

Vector coordinates follow the convention used in shader development and computer graphics:

- **x component has smallest stride** (transposed from array convention)
- **Indexing**: ``vector(x, y, z, ...)`` where the x component varies fastest
- **Coordinate mapping**: Designed to align with how graphics APIs and textures naturally work

This convention ensures that ``vector.x`` represents the fastest-varying dimension (like horizontal position in textures), matching how graphics developers often intuitively think about coordinates.

Why Both Conventions Exist
---------------------------

This dual convention design solves a fundamental mismatch between ML and graphics mental models:

**The Core Problem**
Graphics developers are mentally trained to think of the 'x' component of a vector as representing the smallest stride. In a texture, the x coordinate is the pixel in a row, and the y coordinate is the row in the overall image.
But ML conventions would represent a 512×128 texture as shape ``[128, 512]`` with strides ``[512, 1]`` – exactly backwards from graphics expectations.

**Graphics Developer Expectations**
Graphics developers expect vector coordinates where x represents the smallest stride. This mental model comes from working with spatial coordinates where x represents horizontal position (the fastest-varying dimension in typical memory layouts).

**ML Developer Expectations**
Machine learning developers are accustomed to NumPy-style indexing where the rightmost dimension has the smallest stride. Forcing them to use transposed coordinates would create unnecessary confusion and errors in ML workflows.

**Solution: Support Both**
SlangPy supports both conventions as a necessary compromise. While having two coordinate systems adds complexity, the alternative—forcing either community to abandon their established mental models—would create even greater confusion and adoption barriers.
In practice, developers typically work primarily with either vectors or arrays, so the dual system generally works well despite the conceptual overhead.

Example: call_id Generator
---------------------------

The ``call_id`` generator provides a concrete example of how these conventions manifest in practice. Consider two Slang functions, one that takes a vector parameter and one that takes an array parameter:

.. code-block::

    // Vector parameter (graphics convention)
    int2 myfunc_vector(int2 value) {
        return value;
    }

    // Array parameter (ML convention)
    int2 myfunc_array(int[2] value) {
        return int2(value[0], value[1]);
    }

When we dispatch the same ``call_id`` generator to both functions over a 4×4 grid:

.. code-block:: python

    # Vector version: x component represents rightmost dimension
    res_vector = np.zeros((4, 4, 2), dtype=np.int32)
    module.myfunc_vector(spy.call_id(), _result=res_vector)

    # Array version: rightmost dimension has smallest stride
    res_array = np.zeros((4, 4, 2), dtype=np.int32)
    module.myfunc_array(spy.call_id(), _result=res_array)

The outputs will demonstrate the coordinate convention difference:

.. code-block:: python

    # Vector result (graphics convention):
    # [ [ [0,0], [1,0], [2,0], [3,0] ],
    #   [ [0,1], [1,1], [2,1], [3,1] ], ... ]

    # Array result (ML convention):
    # [ [ [0,0], [0,1], [0,2], [0,3] ],
    #   [ [1,0], [1,1], [1,2], [1,3] ], ... ]

Notice how the vector version fills the x component (first element) fastest, while the array version fills the rightmost index (second element) fastest.
For position [row=0, col=1], the vector gets ``[1,0]`` but the array gets ``[0,1]`` - the coordinates are transposed.

Example: Texture2D Sampling
---------------------------

The graphics convention is more intuitive when working with textures. Consider this Slang function that samples a texture:

.. code-block::

    float4 sample_texture(Texture2D<float4> tex, SamplerState sampler, float2 uv) {
        return tex.SampleLevel(sampler, uv, 0);
    }

When graphics developers work with textures, they naturally think in terms of (x, y) coordinates where:

- **x represents horizontal position** (column, width dimension)
- **y represents vertical position** (row, height dimension)
- **x should be the fastest-varying dimension** for memory efficiency

For a 512×128 texture (512 wide, 128 tall), the vector convention aligns perfectly:

.. code-block:: python

    # Vector coordinates: (x, y) = (column, row)
    uv_coords = np.array([
        [0.0, 0.0],    # Top-left: column 0, row 0
        [1.0, 0.0],    # Top-right: column 511, row 0
        [0.5, 0.5]     # Center: column 255, row 64
    ], dtype=np.float32)

    # This feels natural to graphics developers
    result = module.sample_texture(texture, sampler, uv_coords, _result='numpy')

With array coordinates, the same texture would require thinking about it as shape ``[128, 512]`` (height first, width second), which conflicts with how graphics APIs and developers typically conceptualize texture dimensions.
The vector convention preserves the intuitive ``(width, height)`` mental model that graphics developers expect.

Summary
-------

SlangPy's dual coordinate system represents a necessary compromise between ML and graphics conventions.
While supporting both array coordinates (ML convention) and vector coordinates (graphics convention) adds conceptual complexity, it avoids the greater headache of forcing either community to unlearn their established mental models.
Most developers work primarily with one convention or the other, making the system manageable in practice.
