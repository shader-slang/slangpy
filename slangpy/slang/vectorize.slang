// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

//--------------------------------------------------------------------------------------------
// This file contains all the interfaces and extensions for the experimental slang based
// vectorization system, that can be enabled with slangpy.set_experimental_vectorization(True).
//--------------------------------------------------------------------------------------------

// Represents an unknown type
public struct Unknown: IDifferentiable { }

// This to describe how they can be vectorized from a marshall type
// to a type argument for a slang function.
public struct Vectorizer<From, To> {

}

public interface INoneDefaultFloatingPointType {}
extension half : INoneDefaultFloatingPointType {}
extension double : INoneDefaultFloatingPointType {}

public interface INoneDefaultIntegerType {}
extension int8_t : INoneDefaultIntegerType {}
extension int16_t : INoneDefaultIntegerType {}
extension int64_t : INoneDefaultIntegerType {}
extension uint8_t : INoneDefaultIntegerType {}
extension uint16_t : INoneDefaultIntegerType {}
extension uint32_t : INoneDefaultIntegerType {}
extension uint64_t : INoneDefaultIntegerType {}
extension bool : INoneDefaultIntegerType {}

//--------------------------------------------------------------------------------------------
// These defines a set of interfaces that a marshall can inherit from to add some common
// behaviours, such as the ability to vectorize to a specific target type.
//--------------------------------------------------------------------------------------------

// By inheriting from IVectorizeTo, a Vectorizer is automatically defined with the
// correct type arguments. Due to slang bug with multiple interface extensions, we
// define N versions of the interface.
public interface IVectorizeTo<TargetType, let Dimensionality : int> {}

// Add extension to vectorizer that defines the VectorType for anything
// that implements IVectorizeTo
public extension<TargetType, let Dimensionality : int, From : IVectorizeTo<TargetType, Dimensionality>> Vectorizer<From, TargetType> {
    typealias VectorType = TargetType;
}

// IVectorizeTo for a single specific type, typically for types that map directly to themselves (eg structured buffer)
public interface IVectorizeToSelf<TargetType> {}
public extension<TargetType, From : IVectorizeToSelf<TargetType>> Vectorizer<From, TargetType> {
    typealias VectorType = TargetType;
}

// IVectorizeTo for scalars, that handles the target scalar being 'unknown'
public interface IVectorizeToScalar<TargetType> {}
public extension<TargetType, From : IVectorizeToScalar<TargetType>> Vectorizer<From, TargetType> {
    typealias VectorType = TargetType;
}
public extension<TargetType, From : IVectorizeToScalar<TargetType>> Vectorizer<From, Unknown> {
    typealias VectorType = TargetType;
}

// IVectorizeTo for scalars that can translate:
// - python floats -> additional slang float types
// - python float -> slang int types
// - python int -> additional slang int types
// - python int -> any slang float types
// - python bool -> slang int
// - python int -> any pointer
public interface IVectorizeToScalarConversions<TargetType> {}
public extension<From : IVectorizeToScalarConversions<float>, TargetType : INoneDefaultFloatingPointType> Vectorizer<From, TargetType> {
    typealias VectorType = TargetType;
}
public extension<From : IVectorizeToScalarConversions<float>, TargetType : __BuiltinIntegerType> Vectorizer<From, TargetType> {
    typealias VectorType = TargetType;
}
public extension<From : IVectorizeToScalarConversions<int>, TargetType : INoneDefaultIntegerType> Vectorizer<From, TargetType> {
    typealias VectorType = TargetType;
}
public extension<From : IVectorizeToScalarConversions<int>, TargetType : __BuiltinFloatingPointType> Vectorizer<From, TargetType> {
    typealias VectorType = TargetType;
}
public extension<From : IVectorizeToScalarConversions<bool>, TargetType : __BuiltinIntegerType> Vectorizer<From, TargetType> {
    typealias VectorType = TargetType;
}
public extension<From : IVectorizeToScalarConversions<int>, TargetPtrType> Vectorizer<From, Ptr<TargetPtrType>> {
    typealias VectorType = Ptr<TargetPtrType>;
}


// IVectorizeTo for vectors, that handles the target vector type 'unknown' and/or dimensionality being unknown (0)
public interface IVectorizeToVector<TargetVectorElement, let TargetVectorDim : int> {}
public extension<TargetVectorElement, let TargetVectorDim : int, From : IVectorizeToVector<TargetVectorElement, TargetVectorDim>> Vectorizer<From, vector<TargetVectorElement, TargetVectorDim>> {
    typealias VectorType = vector<TargetVectorElement, TargetVectorDim>;
}
public extension< TargetVectorElement, let TargetVectorDim : int, From : IVectorizeToVector<TargetVectorElement, TargetVectorDim>> Vectorizer<From, vector<Unknown, TargetVectorDim>> {
    typealias VectorType = vector<TargetVectorElement, TargetVectorDim>;
}
public extension< TargetVectorElement, let TargetVectorDim : int, From : IVectorizeToVector<TargetVectorElement, TargetVectorDim>> Vectorizer<From, vector<TargetVectorElement, 0>> {
    typealias VectorType = vector<TargetVectorElement, TargetVectorDim>;
}
public extension< TargetVectorElement, let TargetVectorDim : int, From : IVectorizeToVector<TargetVectorElement, TargetVectorDim>> Vectorizer<From, vector<Unknown, 0>> {
    typealias VectorType = vector<TargetVectorElement, TargetVectorDim>;
}

// IVectorizeTo for vectors in which the size is not known at compile time, that handles the target vector type 'unknown' and the dimensionality is always unknown
public interface IVectorizeToUnsizedVector<TargetVectorElement> {}
public extension< TargetVectorElement, let TargetVectorDim : int, From : IVectorizeToUnsizedVector<TargetVectorElement>> Vectorizer<From, vector<TargetVectorElement, TargetVectorDim>> {
    typealias VectorType = vector<TargetVectorElement, TargetVectorDim>;
}
public extension< TargetVectorElement, let TargetVectorDim : int, From : IVectorizeToUnsizedVector<TargetVectorElement>> Vectorizer<From, vector<Unknown, TargetVectorDim>> {
    typealias VectorType = vector<TargetVectorElement, TargetVectorDim>;
}

// IVectorizeTo for vectors that specifically only handles the generic case in which type and/or dimensionality is unknown
public interface IVectorizeToGenericVector<TargetVectorElement, let TargetVectorDim : int> {}
public extension< TargetVectorElement, let TargetVectorDim : int, From : IVectorizeToGenericVector<TargetVectorElement, TargetVectorDim>> Vectorizer<From, vector<Unknown, TargetVectorDim>> {
    typealias VectorType = vector<TargetVectorElement, TargetVectorDim>;
}
public extension< TargetVectorElement, let TargetVectorDim : int, From : IVectorizeToGenericVector<TargetVectorElement, TargetVectorDim>> Vectorizer<From, vector<TargetVectorElement, 0>> {
    typealias VectorType = vector<TargetVectorElement, TargetVectorDim>;
}
public extension< TargetVectorElement, let TargetVectorDim : int, From : IVectorizeToGenericVector<TargetVectorElement, TargetVectorDim>> Vectorizer<From, vector<Unknown, 0>> {
    typealias VectorType = vector<TargetVectorElement, TargetVectorDim>;
}


// IVectorizeTo for matrices (matrix<T,R,C>), that handles the target matrix type 'unknown' and/or rows/cols being unknown (0)
public interface IVectorizeToMatrix<TargetElementType, let TargetRows : int, let TargetCols : int> {}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<TargetElementType, TargetRows, TargetCols>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<Unknown, TargetRows, TargetCols>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<TargetElementType, 0, TargetCols>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<Unknown, 0, TargetCols>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<TargetElementType, TargetRows, 0>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<Unknown, TargetRows, 0>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<TargetElementType, 0, 0>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<Unknown, 0, 0>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}

// IVectorizeTo for matrices in which the size is not known at compile time, that handles the target matrix type 'unknown' and the rows/cols are always unknown
public interface IVectorizeToUnsizedMatrix<TargetElementType> {}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToUnsizedMatrix<TargetElementType>> Vectorizer<From, matrix<TargetElementType, TargetRows, TargetCols>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToUnsizedMatrix<TargetElementType>> Vectorizer<From, matrix<Unknown, TargetRows, TargetCols>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}

// IVectorizeTo for matrices that specifically only handles the generic case in which type and/or rows/cols are unknown
public interface IVectorizeToGenericMatrix<TargetElementType, let TargetRows : int, let TargetCols : int> {}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToGenericMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<Unknown, TargetRows, TargetCols>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToGenericMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<TargetElementType, 0, TargetCols>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToGenericMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<Unknown, 0, TargetCols>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToGenericMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<TargetElementType, TargetRows, 0>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToGenericMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<Unknown, TargetRows, 0>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToGenericMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<TargetElementType, 0, 0>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToGenericMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<Unknown, 0, 0>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}

// IVectorizeTo for arrays (Array<T,N>), that handles the target array type 'unknown' and/or dimensionality being unknown (0)
public interface IVectorizeToArray<TargetElementType, let TargetDim : int> {}
public extension<TargetElementType, let TargetDim : int, From : IVectorizeToArray<TargetElementType, TargetDim>> Vectorizer<From, Array<TargetElementType, TargetDim>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}
public extension<TargetElementType, let TargetDim : int, From : IVectorizeToArray<TargetElementType, TargetDim>> Vectorizer<From, Array<Unknown, TargetDim>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}
public extension<TargetElementType, let TargetDim : int, From : IVectorizeToArray<TargetElementType, TargetDim>> Vectorizer<From, Array<TargetElementType,  0>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}
public extension<TargetElementType, let TargetDim : int, From : IVectorizeToArray<TargetElementType, TargetDim>> Vectorizer<From, Array<Unknown,  0>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}

// IVectorizeTo for arrays that are python floats, but need to support half/float/double or any int type
public extension< TargetElementType : INoneDefaultFloatingPointType, let TargetDim : int, From : IVectorizeToArray<float,TargetDim>> Vectorizer<From, Array<TargetElementType, TargetDim>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}
public extension< TargetElementType : INoneDefaultFloatingPointType, let TargetDim : int, From : IVectorizeToArray<float,TargetDim>> Vectorizer<From, Array<TargetElementType, 0>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}
public extension< TargetElementType : __BuiltinIntegerType, let TargetDim : int, From : IVectorizeToArray<float, TargetDim>> Vectorizer<From, Array<TargetElementType, TargetDim>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}
public extension< TargetElementType : __BuiltinIntegerType, let TargetDim : int, From : IVectorizeToArray<float, TargetDim>> Vectorizer<From, Array<TargetElementType, 0>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}

// IVectorizeTo for arrays that are python ints, but need to support all the other int types or a float type
public extension<TargetElementType : INoneDefaultIntegerType, let TargetDim : int, From : IVectorizeToArray<int,TargetDim>> Vectorizer<From, Array<TargetElementType, TargetDim>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}
public extension< TargetElementType : __BuiltinFloatingPointType, let TargetDim : int, From : IVectorizeToArray<int, TargetDim>> Vectorizer<From, Array<TargetElementType, TargetDim>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}
public extension<TargetElementType : INoneDefaultIntegerType, let TargetDim : int, From : IVectorizeToArray<int,TargetDim>> Vectorizer<From, Array<TargetElementType, 0>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}
public extension<TargetElementType : __BuiltinFloatingPointType, let TargetDim : int, From : IVectorizeToArray<int, TargetDim>> Vectorizer<From, Array<TargetElementType, 0>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}

// IVectorizeTo for StructuredBuffer<T>, where T can be unknown in either the marshall or the target
public interface IVectorizeToStructuredBuffer<TargetElementType> {}
public extension<TargetElementType, From : IVectorizeToStructuredBuffer<TargetElementType>> Vectorizer<From, StructuredBuffer<TargetElementType>> {
    typealias VectorType = StructuredBuffer<TargetElementType>;
}
public extension<TargetElementType, From : IVectorizeToStructuredBuffer<TargetElementType>> Vectorizer<From, StructuredBuffer<Unknown>> {
    typealias VectorType = StructuredBuffer<TargetElementType>;
}
public extension<TargetElementType, From : IVectorizeToStructuredBuffer<Unknown>> Vectorizer<From, StructuredBuffer<TargetElementType>> {
    typealias VectorType = StructuredBuffer<TargetElementType>;
}


// IVectorizeTo for RWStructuredBuffer<T>, where T can be unknown in either the marshall or the target
public interface IVectorizeToRWStructuredBuffer<TargetElementType> {}
public extension<TargetElementType, From : IVectorizeToRWStructuredBuffer<TargetElementType>> Vectorizer<From, RWStructuredBuffer<TargetElementType>> {
    typealias VectorType = RWStructuredBuffer<TargetElementType>;
}
public extension<TargetElementType, From : IVectorizeToRWStructuredBuffer<TargetElementType>> Vectorizer<From, RWStructuredBuffer<Unknown>> {
    typealias VectorType = RWStructuredBuffer<TargetElementType>;
}
public extension<TargetElementType, From : IVectorizeToRWStructuredBuffer<Unknown>> Vectorizer<From, RWStructuredBuffer<TargetElementType>> {
    typealias VectorType = RWStructuredBuffer<TargetElementType>;
}

// IVectorizeTo for Ptr<T>, where T can be unknown in either the marshall or the target
public interface IVectorizeToPointer<TargetElementType> {}
public extension<TargetElementType, From : IVectorizeToPointer<TargetElementType>> Vectorizer<From, Ptr<TargetElementType>> {
    typealias VectorType = Ptr<TargetElementType>;
}
public extension<TargetElementType, From : IVectorizeToPointer<TargetElementType>> Vectorizer<From, Ptr<Unknown>> {
    typealias VectorType = Ptr<TargetElementType>;
}
public extension<TargetElementType, From : IVectorizeToPointer<Unknown>> Vectorizer<From, Ptr<TargetElementType>> {
    typealias VectorType = Ptr<TargetElementType>;
}


// IVectorizeTo for ByteAddressBuffer
public interface IVectorizeToByteAddressBuffer {}
public extension<From : IVectorizeToByteAddressBuffer> Vectorizer<From, ByteAddressBuffer> {
    typealias VectorType = ByteAddressBuffer;
}

// IVectorizeTo for RWByteAddressBuffer
public interface IVectorizeToRWByteAddressBuffer {}
public extension<From : IVectorizeToRWByteAddressBuffer> Vectorizer<From, RWByteAddressBuffer> {
    typealias VectorType = RWByteAddressBuffer;
}

//--------------------------------------------------------------------------------------------
// Bespoke vectorizers for specific marshall types
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
// NDBufferMarshall<T,N,RW> represents a slangpy NDBuffer or Tensor marshall type.
//--------------------------------------------------------------------------------------------

struct NDBufferMarshall<T, let N : int, let RW : int>
    : IVectorizeToScalar<T>
   , IVectorizeToPointer<T>
{

}

// Add vector/matrix vectorizers for NDBufferMarshall for integer, float and bool types
extension<T : __BuiltinIntegerType, let N : int, let RW : int> NDBufferMarshall<T, N, RW>
    : IVectorizeToUnsizedVector<T>
    , IVectorizeToUnsizedMatrix<T>
{}
extension<T : __BuiltinFloatingPointType, let N : int, let RW : int> NDBufferMarshall<T, N, RW>
    : IVectorizeToUnsizedVector<T>
    , IVectorizeToUnsizedMatrix<T>
{}
extension<let N : int, let RW : int> NDBufferMarshall<bool, N, RW>
    : IVectorizeToUnsizedVector<bool>
    , IVectorizeToUnsizedMatrix<bool>
{}

// Add support for generic vector/matrix types if the underlying element type is vector/matrix
extension<VectorT, let VectorDim : int, let N : int, let RW : int> NDBufferMarshall<vector<VectorT, VectorDim>, N, RW>
    : IVectorizeToGenericVector<VectorT, VectorDim>
{}
extension<MatrixT, let Rows : int, let Cols : int, let N : int, let RW : int> NDBufferMarshall<matrix<MatrixT, Rows, Cols>, N, RW>
    : IVectorizeToGenericMatrix<MatrixT, Rows, Cols>
{}

// Specifically add support for typed arrays. The case of generic typed arrays must be handled python
// side as it is too ambigous to resolve with slang extension rules, due to the fact that a generic type,
// T, could itself be an array type.
extension<T, let N : int, let RW : int, let R : int> Vectorizer<NDBufferMarshall<T, N, RW>, T[R]> {
    typealias VectorType = T[R];
}
extension<T, let N : int, let RW : int, let R : int, let C : int> Vectorizer<NDBufferMarshall<T, N, RW>, T[R][C]> {
    typealias VectorType = T[R][C];
}
extension<T, let N : int, let RW : int, let R : int> Vectorizer<NDBufferMarshall<T[R], N, RW>, T[0]> {
    typealias VectorType = T[R];
}
extension<T, let N : int, let RW : int, let R : int, let C : int> Vectorizer<NDBufferMarshall<T[R][C], N, RW>, T[R][0]> {
    typealias VectorType = T[R][C];
}
extension<T, let N : int, let RW : int, let R : int, let C : int> Vectorizer<NDBufferMarshall<T[R][C], N, RW>, T[0][C]> {
    typealias VectorType = T[R][C];
}
extension<T, let N : int, let RW : int, let R : int, let C : int> Vectorizer<NDBufferMarshall<T[R][C], N, RW>, T[0][0]> {
    typealias VectorType = T[R][C];
}

// NDBufferMarshall<T,N,RW> can vectorize to NDBuffer<T,N> or NDBuffer<Unknown,N>
extension<T, let N : int, let RW : int> Vectorizer<NDBufferMarshall<T, N, RW>, NDBuffer<T, N>> {
    typealias VectorType = NDBuffer<T, N>;
}
extension<T, let N : int, let RW : int> Vectorizer<NDBufferMarshall<T, N, RW>, NDBuffer<Unknown, N>> {
    typealias VectorType = NDBuffer<T, N>;
}

// NDBufferMarshall<T,N,true> can vectorize to RWNDBuffer<T,N> or RWNDBuffer<Unknown,N>
extension<T, let N : int> Vectorizer<NDBufferMarshall<T, N, 1>, RWNDBuffer<T, N>> {
    typealias VectorType = RWNDBuffer<T, N>;
}
extension<T, let N : int> Vectorizer<NDBufferMarshall<T, N, 1>, RWNDBuffer<Unknown, N>> {
    typealias VectorType = RWNDBuffer<T, N>;
}

#ifndef __TARGET_CUDA__
// NDBufferMarshall to its internal storage types.
// Storage is always a pointer on cuda, which is already handled
extension<T, let N : int, let RW : int> Vectorizer<NDBufferMarshall<T, N, RW>, StorageTraits<T>.BufferType> {
    typealias VectorType = StorageTraits<T>.BufferType;
}
extension<T, let N : int, let RW : int> Vectorizer<NDBufferMarshall<T, N, RW>, StorageTraits<Unknown>.BufferType> {
    typealias VectorType = StorageTraits<T>.BufferType;
}
extension<T, let N : int, let RW : int> Vectorizer<NDBufferMarshall<T, N, RW>, ByteAddressBuffer> {
    typealias VectorType = ByteAddressBuffer;
}
extension<T, let N : int> Vectorizer<NDBufferMarshall<T, N, 1>, StorageTraits<T>.RWBufferType> {
    typealias VectorType = StorageTraits<T>.RWBufferType;
}
extension<T, let N : int> Vectorizer<NDBufferMarshall<T, N, 1>, StorageTraits<Unknown>.RWBufferType> {
    typealias VectorType = StorageTraits<T>.RWBufferType;
}
extension<T, let N : int> Vectorizer<NDBufferMarshall<T, N, 1>, RWByteAddressBuffer> {
    typealias VectorType = RWByteAddressBuffer;
}
#endif

// Buffer of uint64s can vectorize to pointer of known type
extension<T, let N : int, let RW : int> Vectorizer<NDBufferMarshall<uint64_t, N, RW>, Ptr<T>> {
    typealias VectorType = Ptr<T>;
}

void _ndbuffer_test()
{
    Vectorizer<NDBufferMarshall<float[4], 1, 1>, float[4]>.VectorType vt5;
    Vectorizer<NDBufferMarshall<int, 1, 1>, vector<Unknown, 1>>.VectorType vt9;
    Vectorizer<NDBufferMarshall<float[4], 1, 1>, float [4][2]>.VectorType vt10;
    Vectorizer<NDBufferMarshall<float[4], 1, 1>, float [4][4]>.VectorType vt11;
    Vectorizer<NDBufferMarshall<float[4], 1, 1>, float [4][4][4]>.VectorType vt12;
    Vectorizer<NDBufferMarshall<float, 1, 1>, float [4][4]>.VectorType vt13;
    Vectorizer<NDBufferMarshall<float[2], 1, 1>, float[0]>.VectorType vt14;
    Vectorizer<NDBufferMarshall<float [2][3], 1, 1>, float [2][3]>.VectorType vt15;
#ifndef __TARGET_CUDA__
    Vectorizer<NDBufferMarshall<float, 1, 0>, StructuredBuffer<float>>.VectorType vt17;
    // Vectorizer<NDBufferMarshall<float, 1, 0>, RWStructuredBuffer<float>>.VectorType vt18;
    Vectorizer<NDBufferMarshall<float, 1, 1>, StructuredBuffer<float>>.VectorType vt19;
    Vectorizer<NDBufferMarshall<float, 1, 1>, RWStructuredBuffer<float>>.VectorType vt20;
#endif
    Vectorizer<NDBufferMarshall<float, 1, 0>, Ptr<float>>.VectorType vt21;
    Vectorizer<NDBufferMarshall<float, 1, 1>, Ptr<float>>.VectorType vt22;
    Vectorizer<NDBufferMarshall<float, 1, 0>, Ptr<Unknown>>.VectorType vt23;
    Vectorizer<NDBufferMarshall<float, 1, 1>, Ptr<Unknown>>.VectorType vt24;
}
