// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

import ArrayModule;

struct DiffCoopVec<T : IReal, int N> : IDifferentiable, IArray<T>, IArithmetic
{
    typealias Differential = DiffCoopVec<T, N>;

    static const CoopVecComponentType ComponentType =
        (T is half) ? CoopVecComponentType.Float16 :
        (T is float) ? CoopVecComponentType.Float32 :
        CoopVecComponentType.Float64;

    CoopVec<T, N> cv;

    [BackwardDifferentiable] __init() { this = fill(T(0.0f)); }
    [BackwardDifferentiable] __init(T x) { this = fill(x); }
    [BackwardDifferentiable] __init<S : IReal>(S x) { this = fill(T(x.toFloat())); }
    [BackwardDifferentiable] __init(This x) { cv = x.cv; } // TODO: Derivative?
    [BackwardDifferentiable] __init<S : IReal>(DiffCoopVec<S, N> other) { cv.copyFrom(other.cv); } // TODO: Derivative?
    __init<S : IReal>(no_diff CoopVec<S, N> other) { cv.copyFrom(other); }

    int getCount()
    {
        return N;
    }

    __subscript(int index) -> T
    {
        [BackwardDifferentiable] get { return indexRead(this, index); }
        [BackwardDifferentiable] set { indexWrite(this, index, newValue); }
    }

    bool equals(This other) { return cv.equals(other.cv); }
    bool lessThan(This other) { return cv.lessThan(other.cv); }
    bool lessThanOrEquals(This other) { return cv.lessThanOrEquals(other.cv); }
    [BackwardDifferentiable] This add(This other) { return add(this, other); }
    [BackwardDifferentiable] This sub(This other) { return sub(this, other); }
    [BackwardDifferentiable] This mul(This other) { return mul(this, other); }
    [BackwardDifferentiable] This div(This other) { return div(this, other); }
    [BackwardDifferentiable] This neg() { return neg(this); }
    This mod(This other) { return This(cv.mod(other.cv)); }

    [BackwardDifferentiable] T[N] toArray() { return toArray(this); }
    [BackwardDifferentiable] vector<T, N> toVector() { return toVector(this); }

    static Differential dzero() { return Differential(T(0.0f)); }
    static Differential dmul(T factor, Differential d) { return This(factor * d.cv); }
    static Differential dadd(Differential a, Differential b) { return a + b; }

    [BackwardDerivative(fill_bwd)]
    static This fill(T x) { return This(CoopVec<T, N>(x)); }
    [BackwardDerivative(indexRead_bwd)]
    static T indexRead(This x, int i) { return x.cv[i]; }
    [BackwardDerivative(indexWrite_bwd)]
    static void indexWrite(inout This x, int i, T value) { x.cv[i] = value; }
    [BackwardDerivative(toArray_bwd)]
    static T[N] toArray(This x)
    {
        T result[N];
        for (int i = 0; i < N; ++i)
            result[i] = x.cv[i];
        return result;
    }
    [BackwardDerivative(toVector_bwd)]
    static vector<T, N> toVector(This x)
    {
        vector<T, N> result;
        for (int i = 0; i < N; ++i)
            result[i] = x.cv[i];
        return result;
    }
    [BackwardDerivative(fromArray_bwd)]
    static This fromArray(T x[N])
    {
        CoopVec<T, N> cv;
        for (int i = 0; i < N; ++i)
            cv[i] = x[i];
        return This(cv);
    }
    [BackwardDerivative(fromVector_bwd)]
    static This fromVector(vector<T, N> x)
    {
        CoopVec<T, N> cv;
        for (int i = 0; i < N; ++i)
            cv[i] = x[i];
        return This(cv);
    }
    [BackwardDerivative(add_bwd)] static This add(This a, This b) { return This(a.cv.add(b.cv)); }
    [BackwardDerivative(sub_bwd)] static This sub(This a, This b) { return This(a.cv.sub(b.cv)); }
    [BackwardDerivative(mul_bwd)] static This mul(This a, This b) { return This(a.cv.mul(b.cv)); }
    [BackwardDerivative(div_bwd)] static This div(This a, This b) { return This(a.cv.div(b.cv)); }
    [BackwardDerivative(neg_bwd)] static This neg(This x) { return This(x.cv.neg()); }

    static void fill_bwd(inout DifferentialPair<T> x, Differential grad)
    {
        T dx = T(0.0f);
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
            dx += grad[i];

        x = diffPair(x.p, __slang_noop_cast<DifferentialPair<T>.DifferentialElementType>(dx));
    }
    static void indexRead_bwd(inout DifferentialPair<This> x, int i, T.Differential grad)
    {
        Differential d = dzero();
        indexWrite(d, i, __slang_noop_cast<T>(grad));
        x = diffPair(x.p, d);
    }
    static void indexWrite_bwd(inout DifferentialPair<This> x, int i, inout DifferentialPair<T> value)
    {
        let grad = __slang_noop_cast<T.Differential>(indexRead(x.d, i));
        value = diffPair(value.p, grad);
    }
    static void toArray_bwd(inout DifferentialPair<This> x, T.Differential[N] grad)
    {
        Differential dx;
        for (int i = 0; i < N; ++i)
            dx.cv[i] = __slang_noop_cast<T>(grad[i]);
        x = diffPair(x.p, dx);
    }
    static void toVector_bwd(inout DifferentialPair<This> x, vector<T, N> grad)
    {
        Differential dx;
        for (int i = 0; i < N; ++i)
            dx.cv[i] = grad[i];
        x = diffPair(x.p, dx);
    }
    static void fromArray_bwd(inout DifferentialPair<T[N]> x, This grad)
    {
        T dx[N];
        for (int i = 0; i < N; ++i)
            dx[i] = grad.cv[i];
        x = diffPair(x.p, __slang_noop_cast<DifferentialPair<T[N]>.DifferentialElementType>(dx));
    }
    static void fromVector_bwd(inout DifferentialPair<vector<T, N>> x, This grad)
    {
        vector<T, N> dx;
        for (int i = 0; i < N; ++i)
            dx[i] = grad.cv[i];
        x = diffPair(x.p, __slang_noop_cast<DifferentialPair<vector<T, N>>.DifferentialElementType>(dx));
    }
    static void add_bwd(inout DifferentialPair<This> a, inout DifferentialPair<This> b, Differential grad)
    {
        a = diffPair(a.p, grad);
        b = diffPair(b.p, grad);
    }
    static void sub_bwd(inout DifferentialPair<This> a, inout DifferentialPair<This> b, Differential grad)
    {
        a = diffPair(a.p, grad);
        b = diffPair(b.p, -grad);
    }
    static void mul_bwd(inout DifferentialPair<This> a, inout DifferentialPair<This> b, Differential grad)
    {
        a = diffPair(a.p, b.p * grad);
        b = diffPair(b.p, a.p * grad);
    }
    static void div_bwd(inout DifferentialPair<This> a, inout DifferentialPair<This> b, Differential grad)
    {
        a = diffPair(a.p, grad / b.p);
        b = diffPair(b.p, (-a.p * grad) / (b.p * b.p));
    }
    static void neg_bwd(inout DifferentialPair<This> x, Differential grad)
    {
        x = diffPair(x.p, -grad);
    }
}

DiffCoopVec<S, N> operator *<T : IReal, S : IReal, int N>(DiffCoopVec<S, N> lhs, const T rhs) { return DiffCoopVec<S, N>(lhs.cv * S(rhs.toFloat())); }
DiffCoopVec<S, N> operator +<T : IReal, S : IReal, int N>(DiffCoopVec<S, N> lhs, const T rhs) { return lhs + DiffCoopVec<S, N>(rhs); }
DiffCoopVec<S, N> operator -<T : IReal, S : IReal, int N>(DiffCoopVec<S, N> lhs, const T rhs) { return lhs - DiffCoopVec<S, N>(rhs); }
DiffCoopVec<S, N> operator /<T : IReal, S : IReal, int N>(DiffCoopVec<S, N> lhs, const T rhs) { return lhs / DiffCoopVec<S, N>(rhs); }
DiffCoopVec<S, N> operator *<T : IReal, S : IReal, int N>(const T lhs, DiffCoopVec<S, N> rhs) { return DiffCoopVec<S, N>(S(lhs.toFloat()) * rhs.cv); }
DiffCoopVec<S, N> operator +<T : IReal, S : IReal, int N>(const T lhs, DiffCoopVec<S, N> rhs) { return DiffCoopVec<S, N>(lhs) + rhs; }
DiffCoopVec<S, N> operator -<T : IReal, S : IReal, int N>(const T lhs, DiffCoopVec<S, N> rhs) { return DiffCoopVec<S, N>(lhs) - rhs; }
DiffCoopVec<S, N> operator /<T : IReal, S : IReal, int N>(const T lhs, DiffCoopVec<S, N> rhs) { return DiffCoopVec<S, N>(lhs) / rhs; }

[BackwardDifferentiable]
DiffCoopVec<T, N> exp<T : IReal, int N>(DiffCoopVec<T, N> x)
{
    return DiffCoopVec<T, N>(exp(x.cv));
}
[BackwardDerivativeOf(exp)]
void exp_bwd<T : IReal, int N>(inout DifferentialPair<DiffCoopVec<T, N>> x, DiffCoopVec<T, N> grad)
{
    x = diffPair(x.p, grad * exp(x.p));
}

[BackwardDifferentiable]
DiffCoopVec<T, N> log<T : IReal, int N>(DiffCoopVec<T, N> x)
{
    return DiffCoopVec<T, N>(log(x.cv));
}
[BackwardDerivativeOf(log)]
void log_bwd<T : IReal, int N>(inout DifferentialPair<DiffCoopVec<T, N>> x, DiffCoopVec<T, N> grad)
{
    x = diffPair(x.p, grad / x.p);
}

[BackwardDifferentiable]
DiffCoopVec<T, N> tanh<T : IReal, int N>(DiffCoopVec<T, N> x)
{
    return DiffCoopVec<T, N>(tanh(x.cv));
}
[BackwardDerivativeOf(tanh)]
void tanh_bwd<T : IReal, int N>(inout DifferentialPair<DiffCoopVec<T, N>> x, DiffCoopVec<T, N> grad)
{
    let y = tanh(x.p);
    x = diffPair(x.p, (1.0f - y * y) * grad);
}

[BackwardDifferentiable]
DiffCoopVec<T, N> atan<T : IReal, int N>(DiffCoopVec<T, N> x)
{
    return DiffCoopVec<T, N>(atan(x.cv));
}
[BackwardDerivativeOf(atan)]
void atan_bwd<T : IReal, int N>(inout DifferentialPair<DiffCoopVec<T, N>> x, DiffCoopVec<T, N> grad)
{
    x = diffPair(x.p, grad / (x.p * x.p + 1.0f));
}

[BackwardDifferentiable]
DiffCoopVec<T, N> max<T : IReal, int N>(DiffCoopVec<T, N> x, DiffCoopVec<T, N> y)
{
    return DiffCoopVec<T, N>(max(x.cv, y.cv));
}
[BackwardDerivativeOf(max)]
void max_bwd<T : IReal, int N>(inout DifferentialPair<DiffCoopVec<T, N>> x, inout DifferentialPair<DiffCoopVec<T, N>> y, DiffCoopVec<T, N> grad)
{
    DiffCoopVec<T, N> gradX, gradY;
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
    {
        if (x.p[i] > y.p[i])
            gradX[i] = grad[i];
        else
            gradY[i] = grad[i];
    }
    x = diffPair(x.p, gradX);
    y = diffPair(y.p, gradY);
}

[BackwardDifferentiable]
DiffCoopVec<T, N> min<T : IReal, int N>(DiffCoopVec<T, N> x, DiffCoopVec<T, N> y)
{
    return DiffCoopVec<T, N>(min(x.cv, y.cv));
}
[BackwardDerivativeOf(min)]
void min_bwd<T : IReal, int N>(inout DifferentialPair<DiffCoopVec<T, N>> x, inout DifferentialPair<DiffCoopVec<T, N>> y, DiffCoopVec<T, N> grad)
{
    DiffCoopVec<T, N> gradX, gradY;
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
    {
        if (x.p[i] > y.p[i])
            gradY[i] = grad[i];
        else
            gradX[i] = grad[i];
    }
    x = diffPair(x.p, gradX);
    y = diffPair(y.p, gradY);
}
