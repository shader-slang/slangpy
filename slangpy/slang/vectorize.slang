// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

// Represents an unknown type
public struct Unknown { }

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

// IVectorizeTo for arrays in which the size is not known at compile time, that handles the target array type 'unknown' and the dimensionality is always unknown
public interface IVectorizeToUnsizedArray<TargetElementType> {}
public extension<TargetElementType, let TargetDim : int, From : IVectorizeToUnsizedArray<TargetElementType>> Vectorizer<From, Array<TargetElementType, TargetDim>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}
public extension<TargetElementType, let TargetDim : int, From : IVectorizeToUnsizedArray<TargetElementType>> Vectorizer<From, Array<Unknown, TargetDim>> {
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

// A test marshall that can vectorize to a vector<int,1> when Dimensionality is 0,
// and to int when Dimensionality is 1.
struct TestMarshallType1
    : IVectorizeToVector<int, 1>
    , IVectorizeToScalar<int>
{
    // ...
}

// Exactly the same, but with the interfaces the other way around
struct TestMarshallType2
    : IVectorizeToScalar<int>
    , IVectorizeToVector<int, 1>
{
    // ...
}

void test()
{
    // Compiles fine - 'x1' is vector<int,1>
    Vectorizer<TestMarshallType1, vector<int, 0>> v1;
    v1.VectorType x1;

    // Doesn't compile - 'v2.VectorType' is undefined
    Vectorizer<TestMarshallType1, int> v2;
    v2.VectorType x2;

    // Doesn't compile - 'v3.VectorType' is undefined
    Vectorizer<TestMarshallType2, vector<int, 1>> v3;
    v3.VectorType x3;

    // Compiles fine - 'x3' is int
    Vectorizer<TestMarshallType2, int> v4;
    v4.VectorType x4;

    // Compiles fine - 'x3' is int
    Vectorizer<TestMarshallType2, Unknown> v5;
    v5.VectorType x5;
}
