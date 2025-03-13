// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
implementing NeuralNetworks;

__include IReal;

// This class wraps around a CoopVec to make it differentiable. This is a temporary
// workaround until the Slang core library supplies a differentiable CoopVec.
public struct DiffCoopVec<T : IReal, int N> : IDifferentiable, IArray<T>, IArithmetic
{
    public typealias Differential = DiffCoopVec<T, N>;

    static const CoopVecComponentType ComponentType =
        (T is half) ? CoopVecComponentType.Float16 :
        (T is float) ? CoopVecComponentType.Float32 :
        CoopVecComponentType.Float64;

    CoopVec<T, N> cv;

    [BackwardDifferentiable] public __init() { this = fill(T(0.0f)); }
    [BackwardDifferentiable] public __init(T x) { this = fill(x); }
    [BackwardDifferentiable] public __init<S : IReal>(S x) { this = fill(x); }
    [BackwardDifferentiable] public __init(This x) { this = x; }
    [BackwardDifferentiable] public __init<S : IReal>(DiffCoopVec<S, N> x) { cv = CoopVec<T, N>(x.cv); }
    public __init(no_diff CoopVec<T, N> x) { cv = x; }

    public int getCount()
    {
        return N;
    }

    public __subscript(int index) -> T
    {
        [BackwardDifferentiable] get { return indexRead(this, index); }
        [BackwardDifferentiable] set { indexWrite(this, index, newValue); }
    }

    public bool equals(This other) { return cv.equals(other.cv); }
    public bool lessThan(This other) { return cv.lessThan(other.cv); }
    public bool lessThanOrEquals(This other) { return cv.lessThanOrEquals(other.cv); }
    [BackwardDifferentiable] public This add(This other) { return add(this, other); }
    [BackwardDifferentiable] public This sub(This other) { return sub(this, other); }
    [BackwardDifferentiable] public This mul(This other) { return mul(this, other); }
    [BackwardDifferentiable] public This div(This other) { return div(this, other); }
    [BackwardDifferentiable] public This neg() { return neg(this); }
    public This mod(This other) { return This(cv.mod(other.cv)); }

    [BackwardDifferentiable] public T[N] toArray() { return toArray(this); }
    [BackwardDifferentiable] public vector<T, N> toVector() { return toVector(this); }

    public static Differential dzero() { return Differential(T(0.0f)); }
    public static Differential dadd(Differential a, Differential b) { return a + b; }
    public static Differential dmul<S : __BuiltinRealType>(S factor, Differential d) { return This(__realCast<T>(factor) * d.cv); }

    [BackwardDerivative(fill_bwd)]
    public static This fill<S : IReal>(S x) { return This(CoopVec<T, N>(T(x.toFloat()))); }
    [BackwardDerivative(cast_bwd)]
    public static This cast<S : IReal>(DiffCoopVec<S, N> x) { return This(CoopVec<S, N>(x.cv)); }
    [BackwardDerivative(indexRead_bwd)]
    public static T indexRead(This x, int i) { return x.cv[i]; }
    [BackwardDerivative(indexWrite_bwd)]
    public static void indexWrite(inout This x, int i, T value) { x.cv[i] = value; }
    [BackwardDerivative(toArray_bwd)]
    public static T[N] toArray(This x)
    {
        T result[N];
        for (int i = 0; i < N; ++i)
            result[i] = x.cv[i];
        return result;
    }
    [BackwardDerivative(toVector_bwd)]
    public static vector<T, N> toVector(This x)
    {
        vector<T, N> result;
        for (int i = 0; i < N; ++i)
            result[i] = x.cv[i];
        return result;
    }
    [BackwardDerivative(fromArray_bwd)]
    public static This fromArray(T x[N])
    {
        CoopVec<T, N> cv;
        for (int i = 0; i < N; ++i)
            cv[i] = x[i];
        return This(cv);
    }
    [BackwardDerivative(fromVector_bwd)]
    public static This fromVector(vector<T, N> x)
    {
        CoopVec<T, N> cv;
        for (int i = 0; i < N; ++i)
            cv[i] = x[i];
        return This(cv);
    }
    [BackwardDerivative(add_bwd)] public static This add(This a, This b) { return This(a.cv.add(b.cv)); }
    [BackwardDerivative(sub_bwd)] public static This sub(This a, This b) { return This(a.cv.sub(b.cv)); }
    [BackwardDerivative(mul_bwd)] public static This mul(This a, This b) { return This(a.cv.mul(b.cv)); }
    [BackwardDerivative(div_bwd)] public static This div(This a, This b) { return This(a.cv.div(b.cv)); }
    [BackwardDerivative(neg_bwd)] public static This neg(This x) { return This(x.cv.neg()); }

    public static void fill_bwd<S : IReal>(inout DifferentialPair<S> x, Differential grad)
    {
        T dx = T(0.0f);
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
            dx += grad[i];

        x = diffPair(x.p, __slang_noop_cast<DifferentialPair<S>.DifferentialElementType>(S(dx.toFloat())));
    }
    public static void cast_bwd<S : IReal>(inout DifferentialPair<DiffCoopVec<S, N>> x, Differential grad)
    {
        x = diffPair(x.p, DiffCoopVec<S, N>(CoopVec<S, N>(grad.cv)));
    }
    public static void indexRead_bwd(inout DifferentialPair<This> x, int i, T.Differential grad)
    {
        Differential d = dzero();
        indexWrite(d, i, __slang_noop_cast<T>(grad));
        x = diffPair(x.p, d);
    }
    public static void indexWrite_bwd(inout DifferentialPair<This> x, int i, inout DifferentialPair<T> value)
    {
        let grad = __slang_noop_cast<T.Differential>(indexRead(x.d, i));
        value = diffPair(value.p, grad);
    }
    public static void toArray_bwd(inout DifferentialPair<This> x, T.Differential[N] grad)
    {
        Differential dx;
        for (int i = 0; i < N; ++i)
            dx.cv[i] = __slang_noop_cast<T>(grad[i]);
        x = diffPair(x.p, dx);
    }
    public static void toVector_bwd(inout DifferentialPair<This> x, vector<T, N> grad)
    {
        Differential dx;
        for (int i = 0; i < N; ++i)
            dx.cv[i] = grad[i];
        x = diffPair(x.p, dx);
    }
    public static void fromArray_bwd(inout DifferentialPair<T[N]> x, This grad)
    {
        T dx[N];
        for (int i = 0; i < N; ++i)
            dx[i] = grad.cv[i];
        x = diffPair(x.p, __slang_noop_cast<DifferentialPair<T[N]>.DifferentialElementType>(dx));
    }
    public static void fromVector_bwd(inout DifferentialPair<vector<T, N>> x, This grad)
    {
        vector<T, N> dx;
        for (int i = 0; i < N; ++i)
            dx[i] = grad.cv[i];
        x = diffPair(x.p, __slang_noop_cast<DifferentialPair<vector<T, N>>.DifferentialElementType>(dx));
    }
    public static void add_bwd(inout DifferentialPair<This> a, inout DifferentialPair<This> b, Differential grad)
    {
        a = diffPair(a.p, grad);
        b = diffPair(b.p, grad);
    }
    public static void sub_bwd(inout DifferentialPair<This> a, inout DifferentialPair<This> b, Differential grad)
    {
        a = diffPair(a.p, grad);
        b = diffPair(b.p, -grad);
    }
    public static void mul_bwd(inout DifferentialPair<This> a, inout DifferentialPair<This> b, Differential grad)
    {
        a = diffPair(a.p, b.p * grad);
        b = diffPair(b.p, a.p * grad);
    }
    public static void div_bwd(inout DifferentialPair<This> a, inout DifferentialPair<This> b, Differential grad)
    {
        a = diffPair(a.p, grad / b.p);
        b = diffPair(b.p, (-a.p * grad) / (b.p * b.p));
    }
    public static void neg_bwd(inout DifferentialPair<This> x, Differential grad)
    {
        x = diffPair(x.p, -grad);
    }
}

[BackwardDifferentiable] public DiffCoopVec<S, N> operator +<T : IReal, S : IReal, int N>(DiffCoopVec<S, N> lhs, const T rhs) { return lhs + DiffCoopVec<S, N>(rhs); }
[BackwardDifferentiable] public DiffCoopVec<S, N> operator -<T : IReal, S : IReal, int N>(DiffCoopVec<S, N> lhs, const T rhs) { return lhs - DiffCoopVec<S, N>(rhs); }
[BackwardDifferentiable] public DiffCoopVec<S, N> operator /<T : IReal, S : IReal, int N>(DiffCoopVec<S, N> lhs, const T rhs) { return lhs / DiffCoopVec<S, N>(rhs); }
[BackwardDifferentiable] public DiffCoopVec<S, N> operator +<T : IReal, S : IReal, int N>(const T lhs, DiffCoopVec<S, N> rhs) { return DiffCoopVec<S, N>(lhs) + rhs; }
[BackwardDifferentiable] public DiffCoopVec<S, N> operator -<T : IReal, S : IReal, int N>(const T lhs, DiffCoopVec<S, N> rhs) { return DiffCoopVec<S, N>(lhs) - rhs; }
[BackwardDifferentiable] public DiffCoopVec<S, N> operator /<T : IReal, S : IReal, int N>(const T lhs, DiffCoopVec<S, N> rhs) { return DiffCoopVec<S, N>(lhs) / rhs; }
[BackwardDerivative(scalarMultiplyR_bwd)] public DiffCoopVec<S, N> operator *<T : IReal, S : IReal, int N>(DiffCoopVec<S, N> lhs, const T rhs) { return DiffCoopVec<S, N>(lhs.cv * S(rhs.toFloat())); }
[BackwardDerivative(scalarMultiplyL_bwd)] public DiffCoopVec<S, N> operator *<T : IReal, S : IReal, int N>(const T lhs, DiffCoopVec<S, N> rhs) { return DiffCoopVec<S, N>(S(lhs.toFloat()) * rhs.cv); }
public void scalarMultiplyR_bwd<T : IReal, S : IReal, int N>(inout DifferentialPair<DiffCoopVec<S, N>> lhs, inout DifferentialPair<T> rhs, DiffCoopVec<S, N> grad)
{
    lhs = diffPair(lhs.p, grad * rhs.p);
    DiffCoopVec<S, N>::fill_bwd(rhs, grad * lhs.p);
}
public void scalarMultiplyL_bwd<T : IReal, S : IReal, int N>(inout DifferentialPair<T> lhs, inout DifferentialPair<DiffCoopVec<S, N>> rhs, DiffCoopVec<S, N> grad)
{
    scalarMultiplyR_bwd(rhs, lhs, grad);
}

[BackwardDerivative(exp_bwd)]
public DiffCoopVec<T, N> exp<T : IReal, int N>(DiffCoopVec<T, N> x)
{
    return DiffCoopVec<T, N>(exp(x.cv));
}
public void exp_bwd<T : IReal, int N>(inout DifferentialPair<DiffCoopVec<T, N>> x, DiffCoopVec<T, N> grad)
{
    x = diffPair(x.p, grad * exp(x.p));
}

[BackwardDerivative(log_bwd)]
public DiffCoopVec<T, N> log<T : IReal, int N>(DiffCoopVec<T, N> x)
{
    return DiffCoopVec<T, N>(log(x.cv));
}
public void log_bwd<T : IReal, int N>(inout DifferentialPair<DiffCoopVec<T, N>> x, DiffCoopVec<T, N> grad)
{
    x = diffPair(x.p, grad / x.p);
}

[BackwardDerivative(tanh_bwd)]
public DiffCoopVec<T, N> tanh<T : IReal, int N>(DiffCoopVec<T, N> x)
{
    return DiffCoopVec<T, N>(tanh(x.cv));
}
public void tanh_bwd<T : IReal, int N>(inout DifferentialPair<DiffCoopVec<T, N>> x, DiffCoopVec<T, N> grad)
{
    let y = tanh(x.p);
    x = diffPair(x.p, (1.0f - y * y) * grad);
}

[BackwardDerivative(atan_bwd)]
public DiffCoopVec<T, N> atan<T : IReal, int N>(DiffCoopVec<T, N> x)
{
    return DiffCoopVec<T, N>(atan(x.cv));
}
public void atan_bwd<T : IReal, int N>(inout DifferentialPair<DiffCoopVec<T, N>> x, DiffCoopVec<T, N> grad)
{
    x = diffPair(x.p, grad / (x.p * x.p + 1.0f));
}

[BackwardDerivative(max_bwd)]
public DiffCoopVec<T, N> max<T : IReal, int N>(DiffCoopVec<T, N> x, DiffCoopVec<T, N> y)
{
    return DiffCoopVec<T, N>(max(x.cv, y.cv));
}
public void max_bwd<T : IReal, int N>(inout DifferentialPair<DiffCoopVec<T, N>> x, inout DifferentialPair<DiffCoopVec<T, N>> y, DiffCoopVec<T, N> grad)
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

[BackwardDerivative(min_bwd)]
public DiffCoopVec<T, N> min<T : IReal, int N>(DiffCoopVec<T, N> x, DiffCoopVec<T, N> y)
{
    return DiffCoopVec<T, N>(min(x.cv, y.cv));
}
public void min_bwd<T : IReal, int N>(inout DifferentialPair<DiffCoopVec<T, N>> x, inout DifferentialPair<DiffCoopVec<T, N>> y, DiffCoopVec<T, N> grad)
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
