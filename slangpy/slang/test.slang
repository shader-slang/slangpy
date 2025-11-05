// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module test;


// This to describe how they can be vectorized from a marshall type
// to a type argument for a slang function.
struct Vectorizer<From, To> {

}

// By inheriting from IVectorizeTo, a Vectorizer is automatically defined with the
// correct type arguments.
interface IVectorizeTo<TargetType, let Dimensionality : int> {}
interface IVectorizeTo2<TargetType, let Dimensionality : int> {}

// Add extension to vectorizer that defines the VectorType for anything
// that implements IVectorizeTo
extension<TargetType, let Dimensionality : int, From : IVectorizeTo<TargetType, Dimensionality>>
Vectorizer<From, TargetType> {
    typealias VectorType = TargetType;
}
extension<TargetType, let Dimensionality : int, From : IVectorizeTo2<TargetType, Dimensionality>>
    Vectorizer<From, TargetType> {
    typealias VectorType = TargetType;
}

// A test marshall that can vectorize to a vector<int,1> when Dimensionality is 0,
// and to int when Dimensionality is 1.
struct TestMarshallType1
    : IVectorizeTo<vector<int,1>, 0>
    , IVectorizeTo2<int, 1>
{
    // ...
}

// Exactly the same, but with the interfaces the other way around
struct TestMarshallType2
    : IVectorizeTo<int, 1>
    , IVectorizeTo2<vector<int, 1>, 0>
{
    // ...
}

void test()
{
    // Compiles fine - 'x1' is vector<int,1>
    Vectorizer<TestMarshallType1, vector<int,1>> v1;
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
}
