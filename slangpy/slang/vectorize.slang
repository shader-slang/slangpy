// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

// Represents an unknown type
public struct Unknown { }

// This to describe how they can be vectorized from a marshall type
// to a type argument for a slang function.
public struct Vectorizer<From, To> {

}

// By inheriting from IVectorizeTo, a Vectorizer is automatically defined with the
// correct type arguments. Due to slang bug with multiple interface extensions, we
// define N versions of the interface.
public interface IVectorizeTo<TargetType, let Dimensionality : int> {}

// Add extension to vectorizer that defines the VectorType for anything
// that implements IVectorizeTo
public extension<TargetType, let Dimensionality : int, From : IVectorizeTo<TargetType, Dimensionality>> Vectorizer<From, TargetType> {
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

// IVectorizeTo for vectors, that handles the target vector type 'unknown' and/or dimensionality being unknown (-1)
public interface IVectorizeToVector<TargetVectorElement, let TargetVectorDim : int> {}
public extension<TargetVectorElement, let TargetVectorDim : int, From : IVectorizeToVector<TargetVectorElement, TargetVectorDim>> Vectorizer<From, vector<TargetVectorElement, TargetVectorDim>> {
    typealias VectorType = vector<TargetVectorElement, TargetVectorDim>;
}
public extension< TargetVectorElement, let TargetVectorDim : int, From : IVectorizeToVector<TargetVectorElement, TargetVectorDim>> Vectorizer<From, vector<Unknown, TargetVectorDim>> {
    typealias VectorType = vector<TargetVectorElement, TargetVectorDim>;
}
public extension< TargetVectorElement, let TargetVectorDim : int, From : IVectorizeToVector<TargetVectorElement, TargetVectorDim>> Vectorizer<From, vector<TargetVectorElement, -1>> {
    typealias VectorType = vector<TargetVectorElement, TargetVectorDim>;
}
public extension< TargetVectorElement, let TargetVectorDim : int, From : IVectorizeToVector<TargetVectorElement, TargetVectorDim>> Vectorizer<From, vector<Unknown, -1>> {
    typealias VectorType = vector<TargetVectorElement, TargetVectorDim>;
}

// IVectorizeTo for matrices (matrix<T,R,C>), that handles the target matrix type 'unknown' and/or rows/cols being unknown (-1)
public interface IVectorizeToMatrix<TargetElementType, let TargetRows : int, let TargetCols : int> {}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<TargetElementType, TargetRows, TargetCols>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<Unknown, TargetRows, TargetCols>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<TargetElementType, -1, TargetCols>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<Unknown, -1, TargetCols>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<TargetElementType, TargetRows, -1>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<Unknown, TargetRows, -1>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<TargetElementType, -1, -1>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}
public extension<TargetElementType, let TargetRows : int, let TargetCols : int, From : IVectorizeToMatrix<TargetElementType, TargetRows, TargetCols>> Vectorizer<From, matrix<Unknown, -1, -1>> {
    typealias VectorType = matrix<TargetElementType, TargetRows, TargetCols>;
}

// IVectorizeTo for arrays (Array<T,N>), that handles the target array type 'unknown' and/or dimensionality being unknown (-1)
public interface IVectorizeToArray<TargetElementType, let TargetDim : int> {}
public extension<TargetElementType, let TargetDim : int, From : IVectorizeToArray<TargetElementType, TargetDim>> Vectorizer<From, Array<TargetElementType, TargetDim>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}
public extension<TargetElementType, let TargetDim : int, From : IVectorizeToArray<TargetElementType, TargetDim>> Vectorizer<From, Array<Unknown, TargetDim>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}
public extension<TargetElementType, let TargetDim : int, From : IVectorizeToArray<TargetElementType, TargetDim>> Vectorizer<From, Array<TargetElementType,  -1>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
}
public extension<TargetElementType, let TargetDim : int, From : IVectorizeToArray<TargetElementType, TargetDim>> Vectorizer<From, Array<Unknown,  -1>> {
    typealias VectorType = Array<TargetElementType, TargetDim>;
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
    Vectorizer<TestMarshallType1, vector<int, -1>> v1;
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
