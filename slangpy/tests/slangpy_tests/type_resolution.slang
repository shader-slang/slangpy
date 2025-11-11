// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy;

// Basic scalar function types, should generally support:
// - scalar arg (any python number)
// - vector/matrix arg (correct vector type)
// - containers such as NDBuffer/Tensor of any dimensionality
float func_float(float a)
{
    return a;
}
int func_int(int a)
{
    return a;
}
half func_half(half a)
{
    return a;
}
int8_t func_int8(int8_t a)
{
    return a;
}
int64_t func_int64(int64_t a)
{
    return a;
}

// Fully generic function, generally ambiguous as a type such
// as NDBuffer<float> could be vectorized as NDBuffer<float> or float.
// Only pure scalars can typically disambiguate.
T func_generic<T>(T a)
{
    return a;
}

// Correctly constrained generic function allows disambiguation
// via specializeWithArgTypes. i.e. NDBuffer<int> can only map to
// T: __BuiltinIntegerType by vectorizing to int.
T func_genericint<T: __BuiltinIntegerType>(T a)
{
    return a;
}

// Same principle for float
T func_genericfloat<T : __BuiltinFloatingPointType>(T a)
{
    return a;
}

// General vector types
// - vector of correct type
// - matrix of correct type (vectorizes across rows)
// - containers such as NDBuffer/Tensor of the vector element type
// - containers such as NDBuffer/Tensor of the correct scalar type (eg NDBuffer<float> of shape (N,3) can load float3)
float3 func_float3(float3 a)
{
    return a;
}
vector<float,3> func_vector_float3(vector<float,3> a)
{
    return a;
}
int3 func_int3(int3 a)
{
    return a;
}
vector<int,3> func_vector_int3(vector<int,3> a)
{
    return a;
}
half3 func_half3(half3 a)
{
    return a;
}
vector<half,3> func_vector_half3(vector<half,3> a)
{
    return a;
}


// Vector with fixed dimensionality but generic type. Pretty much always resolvable as
// there is no current vector use case in Python in which the Python side element type
// is unknown.
// - vector
// - matrix
// - containers such as NDBuffer/Tensor of vector type
// - containers such as NDBuffer/Tensor of scalar type (eg NDBuffer<float> of shape (N,3) can load float3)
vector<T,3> func_vector3_generic<T>(vector<T,3> a)
{
    return a;
}

// Vector with generic dimensionality and generic type. The NDBuffer of scalars can not
// be resolved in this case, as the last dimension's size can't be inferred at compile time.
vector<T,N> func_vectorN_generic<T,let N: int>(vector<T,N> a)
{
    return a;
}

// Vector with generic dimensionality and known type. The NDBuffer of scalars can not
// be resolved in this case, as the last dimension's size can't be inferred at compile time.
vector<float, N> func_floatN_generic<let N : int>(vector<float, N> a)
{
    return a;
}

// StructuredBuffer and RWStructuredBuffer types. These can be passed
// a slangpy Buffer, or an NDBuffer/Tensor of the correct element type.
// - spy.Buffer
// - spy.NDBuffer<float> of any shape
// - spy.Tensor<float> of any shape
void func_float_structuredbuffer(StructuredBuffer<float> a)
{
    return;
}
void func_float_rwstructuredbuffer(RWStructuredBuffer<float> a)
{
    return;
}
void func_int_structuredbuffer(StructuredBuffer<int> a)
{
    return;
}
void func_half_structuredbuffer(StructuredBuffer<half> a)
{
    return;
}

// Generic structured buffer is valid when passed a typed slangpy
// container such as NDBuffer/Tensor
// - spy.NDBuffer<T> of any shape
// - spy.Tensor<T> of any shape
void func_generic_structuredbuffer<T>(StructuredBuffer<T> a)
{
    return;
}

// Pointer types obey effectively the same rules as structure buffers.
// in both types and generic situations
// - spy.Buffer
// - spy.NDBuffer<T> of any shape
// - spy.Tensor<T> of any shape
void func_float_ptr(Ptr<float> a)
{
    return;
}
void func_float_cptr(float* a)
{
    return;
}
void func_generic_ptr<T>(Ptr<T> a)
{
    return;
}

// ByteAddressBuffer and RWByteAddressBuffer types. Obey same basic
// rules as structure buffer without any generic element typing.
// - spy.Buffer
// - spy.NDBuffer of any shape
// - spy.Tensor of any shape
void func_bytebuffer(ByteAddressBuffer a)
{
    return;
}
void func_rwbytebuffer(RWByteAddressBuffer a)
{
    return;
}
