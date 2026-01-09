// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Reproduce for incorrect report that function can't be specialized


/*
Reproduce for incorrect report that function can't be specialized. When
- Enabling full new slangpy tensor
- Or use the minimal repro below.

Slang reports the following error:

E c :\sw\slangpy\slangpy\tests\slangpy_tests\test_existential_bug .slang(59) :
error 33180 : specializing 'load' with an existential type is not allowed .
All generic arguments must be statically resolvable at compile time.
E val.x = texture.load(texel.y, texel.x, 0);

The issue will still reproduce if the ability to specify element type
is removed entirely, however this makes it incompatible with the core slangpy
code so I've left it in.

The minimal example has been simplified from the SlangPy tensor so:
- We only extend the concrete Tensor<float,3> instead of Tensor<T,3>
- Load does not return a value or access any internal data.
- No subscripts are used.
- Load involves no generic types.
*/

// Enable this to use the slangpy tensors, or disable to use minimal repro below
#define USE_SLANGPY_TENSORx

#ifdef USE_SLANGPY_TENSOR
import slangpy;
#else

public interface ITensor<T, let D : int>
{

}

public struct Tensor<T, let D : int> : ITensor<T, D>
{
    public StructuredBuffer<T> _data;
    public uint[D] _shape;
    public uint[D] _strides;
    public uint _offset;
}

// Extend the interface to add the 3D load, in this case specifically to ITensors
// with float element type and 3 dimensions.
//
// In practice, in the SlangPy tensor this would be a generic extension on Tensor<T, 3>,
// however this is not necessary to reproduce the bug.
public extension<TensorType : ITensor<float, 3>> TensorType
{
    public void load(int i0, int i1, int i2)
    {
    }
}

#endif

void fetch_texture(ITensor<float, 3> texture)
{
    texture.load(0, 0, 0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
public void build_importance_map(
    uint3 tid: SV_DispatchThreadID,
    uniform Tensor<float, 3> texture
)
{
    fetch_texture(texture);
}
