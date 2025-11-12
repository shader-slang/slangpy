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

// Matrix types
float3x3 func_float3x3(float3x3 a)
{
    return a;
}
float4x4 func_float4x4(float4x4 a)
{
    return a;
}
float3x4 func_float3x4(float3x4 a)
{
    return a;
}
matrix<float,3,3> func_matrix_float3x3(matrix<float,3,3> a)
{
    return a;
}
int3x3 func_int3x3(int3x3 a)
{
    return a;
}
matrix<T,R,C> func_matrix_generic<T, let R: int, let C: int>(matrix<T,R,C> a)
{
    return a;
}
matrix<T, 3, 3> func_matrix_generic33<T>(matrix<T, 3, 3> a)
{
    return a;
}
matrix<float,R,C> func_matrix_floatRC_generic<let R: int, let C: int>(matrix<float,R,C> a)
{
    return a;
}
matrix<float,R,3> func_matrix_floatR3_generic<let R: int>(matrix<float,R,3> a)
{
    return a;
}
matrix<float,3,C> func_matrix_float3C_generic<let C: int>(matrix<float,3,C> a)
{
    return a;
}

// 1D arrays of known type/size. Can be passed
// - python list/tuple of correct type and length
// - spy.NDBuffer/Tensor of ArrayType
// - spy.NDBuffer/Tensor of ElementType
float[4] func_float_array(float[4] a)
{
    return a;
}
Array<float,8> func_float_array2(Array<float,8> a)
{
    return a;
}

// Unsized arrays of known type.
void func_float_unsized_array(float[] a)
{
    return;
}

// Generic arrays. These get more complex as T itself
// could be an array type.
void func_generic_array<T, let N: int>(T[N] a)
{
    return;
}
void func_generic_type_array<T>(T[4] a)
{
    return;
}
void func_generic_constrained_type_array<T: __BuiltinFloatingPointType>(T[4] a)
{
    return;
}
void func_generic_length_array<let N: int>(float[N] a)
{
    return;
}
void func_generic_unsized_array<T>(T[] a)
{
    return;
}

float [8][5] func_float_array2d(float [8][5] a)
{
    return a;
}
float [8][5] func_float_array2d_full(Array<Array<float,8>, 5> a)
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
