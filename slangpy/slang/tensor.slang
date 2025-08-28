// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

public interface ITensor<T : IDifferentiable, let D : int>
{
    [Differentiable] public T get(int idx[D]);

    [Differentiable] public T getv(vector<int,D> idx);


    public property shape : uint[D];
}
public interface IRWTensor<T : IDifferentiable, let D : int> : ITensor<T, D>
{
    [Differentiable] public void set(int idx[D], T value);

    [Differentiable] public void setv(vector<int,D> idx, T value);

}

namespace impl
{
    // Helper function to turn a variadic list into an array of statically known size
    // Slang seems to crash with variadic constructors, so this helper is needed for now
    int[D] makeIndex<let D : int, each T : IInteger>(expand each T indices)
    {
        // We can't currently specify a type constraint that we expect exactly D Ts, so
        // static_assert is needed
        static_assert(countof(T) == D, "Number of indices does not match Tensor dimensionality");
        int[D] idxVec;
        int i = 0;
        expand idxVec[i++] = (each indices).toInt();
        return idxVec;
    }
    // Helper for turning integer array-likes into indices
    int[D] makeIndex<let D : int, T : IInteger>(ISizedArray<T, D> indices)
    {
        int[D] idxVec;
        [ForceUnroll]
        for (int i = 0; i < D; ++i)
            idxVec[i] = indices[i].toInt();
        return idxVec;
    }
}

// #pragma warning(push)
// #pragma warning(disable: 30856)

public extension<T : IDifferentiable, let D : int, TensorType : ITensor<T, D>> TensorType
{
    __generic<each Is : IInteger>
    public __subscript(expand each Is indices)->T
    {
        [BackwardDifferentiable] get { return get(impl::makeIndex<D>(expand each indices)); }
    }
    __generic<I : IInteger>
    __subscript(ISizedArray<I, D> indices)->T
    {
        [BackwardDifferentiable] get { return get(impl::makeIndex<D, I>(indices)); }
    }

    [Differentiable]
    public void load(ContextND<D> ctx, out T value)
    {
        value = get(ctx.call_id);
    }

    public void load(Context0D context, out This value)
    {
        value = this;
    }

    [Differentiable]
    public void load<let N : int>(ContextND<D - 1> ctx, out T[N] value)
    {
        int idx[D];
        [ForceUnroll]
        for (int i = 0; i < D - 1; ++i)
            idx[i] = ctx.call_id[i];

        [ForceUnroll]
        for (int i = 0; i < N; i++) {
            idx[D - 1] = i;
            value[i] = get(idx);
        }
    }

    [Differentiable]
    public void load<let N : int>(ContextND<D - 1> ctx, out vector<T, N> value)
    {
        int idx[D];
        [ForceUnroll]
        for (int i = 0; i < D - 1; ++i)
            idx[i] = ctx.call_id[i];

        [ForceUnroll]
        for (int i = 0; i < N; i++) {
            idx[D - 1] = i;
            value[i] = get(idx);
        }
    }

    [Differentiable]
    public void load<let M : int, let N : int>(ContextND<D - 2> context, out T value[M][N])
    {
        int idx[D];
        [ForceUnroll]
        for (int i = 0; i < D - 2; ++i)
            idx[i] = context.call_id[i];
        [ForceUnroll]
        for (int i = 0; i < M; i++) {
            [ForceUnroll]
            for (int j = 0; j < N; j++) {
                idx[D - 2] = i;
                idx[D - 1] = j;

                value[i][j] = get(idx);
            }
        }
    }

    [Differentiable]
    public void load<let M : int, let N : int>(ContextND<D - 2> context, out matrix <T, M, N> value)
    {
        int idx[D];
        [ForceUnroll]
        for (int i = 0; i < D - 2; ++i)
            idx[i] = context.call_id[i];
        [ForceUnroll]
        for (int i = 0; i < M; i++) {
            [ForceUnroll]
            for (int j = 0; j < N; j++) {
                idx[D - 2] = i;
                idx[D - 1] = j;

                value[i][j] = get(idx);
            }
        }
    }
}

public extension<T : IDifferentiable, let D : int, TensorType : IRWTensor<T, D>> TensorType
{
    [Differentiable]
    public void store(ContextND<D> ctx, in T value)
    {
        set(ctx.call_id, value);
    }

    [Differentiable]
    public void store<let N : int>(ContextND<D - 1> context, in T[N] value)
    {
        int idx[D];
        [ForceUnroll]
        for (int i = 0; i < D - 1; ++i)
            idx[i] = context.call_id[i];
        [ForceUnroll]
        for (int i = 0; i < N; i++) {
            idx[D - 1] = i;
            set(idx, value[i]);
        }
    }

    [Differentiable]
    public void store<let N : int>(ContextND<D - 1> context, in vector<T, N> value)
    {
        int idx[D];
        [ForceUnroll]
        for (int i = 0; i < D - 1; ++i)
            idx[i] = context.call_id[i];
        [ForceUnroll]
        for (int i = 0; i < N; i++) {
            idx[D - 1] = i;
            set(idx, value[i]);
        }
    }

    [Differentiable]
    public void store<let M : int, let N : int>(ContextND<D - 2> context, in T value[M][N])
    {
        int idx[D];
        [ForceUnroll]
        for (int i = 0; i < D - 2; ++i)
            idx[i] = context.call_id[i];
        [ForceUnroll]
        for (int i = 0; i < M; i++) {
            [ForceUnroll]
            for (int j = 0; j < N; j++) {
                idx[D - 2] = i;
                idx[D - 1] = j;

                set(idx, value[i][j]);
            }
        }
    }

    [Differentiable]
    public void store<let M : int, let N : int>(ContextND<D - 2> context, in matrix <T, M, N> value)
    {
        int idx[D];
        [ForceUnroll]
        for (int i = 0; i < D - 2; ++i)
            idx[i] = context.call_id[i];
        [ForceUnroll]
        for (int i = 0; i < M; i++) {
            [ForceUnroll]
            for (int j = 0; j < N; j++) {
                idx[D - 2] = i;
                idx[D - 1] = j;

                set(idx, value[i][j]);
            }
        }
    }

    // Operator [] currently causes compiler crash and is disabled
#if 0
    __generic<each Is : IInteger>
    __subscript(expand each Is indices)->T
    {
        [BackwardDifferentiable] get { return get(impl::makeIndex<D>(expand each indices)); }
        [nonmutating, BackwardDifferentiable] set { return set(impl::makeIndex<D>(expand each indices), newValue); }
    }
    __generic<I : IInteger>
    __subscript(ISizedArray<I, D> indices)->T
    {
        [BackwardDifferentiable] get { return get(impl::makeIndex<D, I>(indices)); }
        [nonmutating, BackwardDifferentiable] set { return set(impl::makeIndex<D, I>(indices), newValue); }
    }
#endif
}

// #pragma warning(pop)

public struct Tensor<T : IDifferentiable, let D : int> : ITensor<T, D>
{
    typealias Storage = StorageTraits<T>.BufferType;

    public Storage buffer;
    public StridedLayout<D> layout;
    private uint _shape[D];

    public property shape : uint[D] { get { return _shape; } }

    [TreatAsDifferentiable]
    public T get(int[D] idx) { return buffer[layout.at(idx)]; }

    [TreatAsDifferentiable]
    public T getv(vector<int,D> idx) { return buffer[layout.at_vec(idx)]; }

    public void load<let SliceD : int>(ContextND<D - SliceD> ctx, out Tensor<T, SliceD> value)
    {
        value.buffer = buffer;
        value.layout = layout.slice<SliceD>(ctx.call_id);
        [ForceUnroll]
        for (int i = 0; i < SliceD; ++i)
            value._shape[i] = _shape[D - SliceD + i];
    }
}

public struct RWTensor<T : IDifferentiable, let D : int> : IRWTensor<T, D>
{
    typealias Storage = StorageTraits<T>.RWBufferType;

    public Storage buffer;
    public StridedLayout<D> layout;
    private uint _shape[D];

    public property shape : uint[D] { get { return _shape; } }

    [TreatAsDifferentiable]
    public T get(int[D] idx) { return buffer[layout.at(idx)]; }

    [TreatAsDifferentiable]
    public T getv(vector<int,D> idx) { return buffer[layout.at_vec(idx)]; }

    [TreatAsDifferentiable]
    public void set(int idx[D], T value) { buffer[layout.at(idx)] = value; }

    [TreatAsDifferentiable]
    public void setv(vector<int,D> idx, T value) { buffer[layout.at_vec(idx)] = value; }

    public void load<let SliceD : int>(ContextND<D - SliceD> ctx, out RWTensor<T, SliceD> value)
    {
        value.buffer = buffer;
        value.layout = layout.slice<SliceD>(ctx.call_id);
        [ForceUnroll]
        for (int i = 0; i < SliceD; ++i)
            value._shape[i] = _shape[D - SliceD + i];
    }
}

public struct AtomicTensor<T, let D : int> : IRWTensor<T, D> where T : IDifferentiable, IAtomicAddable
{
    typealias Storage = StorageTraits<T>.AtomicBufferType;

    public Storage buffer;
    public StridedLayout<D> layout;
    private uint _shape[D];

    public property shape : uint[D] { get { return _shape; } }

    [TreatAsDifferentiable]
    public T get(int[D] idx) { return StorageTraits<T>::atomicLoad(buffer, layout.at(idx)); }

    [TreatAsDifferentiable]
    public T getv(vector<int,D> idx) { return StorageTraits<T>::atomicLoad(buffer, layout.at_vec(idx)); }

    [TreatAsDifferentiable]
    public void set(int idx[D], T value) {
        StorageTraits<T>::atomicAdd(buffer, layout.at(idx), value);
    }

    [TreatAsDifferentiable]
    public void setv(vector<int,D> idx, T value) {
        StorageTraits<T>::atomicAdd(buffer, layout.at_vec(idx), value);
    }

    public void load<let SliceD : int>(ContextND<D - SliceD> ctx, out AtomicTensor<T, SliceD> value)
    {
        value.buffer = buffer;
        value.layout = layout.slice<SliceD>(ctx.call_id);
        [ForceUnroll]
        for (int i = 0; i < SliceD; ++i)
            value._shape[i] = _shape[D - SliceD + i];
    }
}

public struct GradOutTensor<T : IDifferentiable, let D : int> : ITensor<T, D> where T.Differential : IAtomicAddable
{
    public Tensor<T, D> primal;
    public AtomicTensor<T.Differential, D> d_out;

    public property shape : uint[D] { get { return primal.shape; } }

    [Differentiable]
    public T get(int[D] idx) { return primal.get(idx); }

    [Differentiable]
    public T getv(vector<int,D> idx) { return primal.getv(idx); }

    [BackwardDerivativeOf(get)]
    public void get_bwd(int[D] idx, T.Differential grad)
    {
        d_out.set(idx, grad);
    }

    [BackwardDerivativeOf(getv)]
    public void getv_bwd(vector<int,D> idx, T.Differential grad)
    {
        d_out.setv(idx, grad);
    }

    public void load<let SliceD : int>(ContextND<D - SliceD> ctx, out GradOutTensor<T, SliceD> value)
    {
        primal.load(ctx, value.primal);
        d_out.load(ctx, value.d_out);
    }
}

public struct GradInTensor<T : IDifferentiable, let D : int> : IRWTensor<T, D>
{
    public RWTensor<T, D> primal;
    public Tensor<T.Differential, D> d_in;

    public property shape : uint[D] { get { return primal.shape; } }

    [TreatAsDifferentiable]
    public T get(int[D] idx) { return primal.get(idx); }

    [TreatAsDifferentiable]
    public T getv(vector<int,D> idx) { return primal.getv(idx); }

    [Differentiable]
    public void set(int[D] idx, T value) { primal.set(idx, value); }

    [Differentiable]
    public void setv(vector<int,D> idx, T value) { primal.setv(idx, value); }

    [BackwardDerivativeOf(set)]
    public void set_bwd(int[D] idx, inout DifferentialPair<T> grad) { grad = diffPair(grad.p, d_in.get(idx)); }

    [BackwardDerivativeOf(setv)]
    public void setv_bwd(vector<int,D> idx, inout DifferentialPair<T> grad) { grad = diffPair(grad.p, d_in.getv(idx)); }

    public void load<let SliceD : int>(ContextND<D - SliceD> ctx, out GradInTensor<T, SliceD> value)
    {
        primal.load(ctx, value.primal);
        d_in.load(ctx, value.d_in);
    }
}

public struct GradInOutTensor<T : IDifferentiable, let D : int> : IRWTensor<T, D> where T.Differential : IAtomicAddable
{
    public RWTensor<T, D> primal;
    public AtomicTensor<T.Differential, D> d_out;
    public Tensor<T.Differential, D> d_in;

    public property shape : uint[D] { get { return primal.shape; } }

    [Differentiable]
    public T get(int[D] idx) { return primal.get(idx); }

    [Differentiable]
    public T getv(vector<int,D> idx) { return primal.getv(idx); }

    [BackwardDerivativeOf(get)]
    public void get_bwd(int[D] idx, T.Differential grad)
    {
        d_out.set(idx, grad);
    }

    [BackwardDerivativeOf(getv)]
    public void getv_bwd(vector<int,D> idx, T.Differential grad)
    {
        d_out.setv(idx, grad);
    }

    [Differentiable]
    public void set(int[D] idx, T value) { primal.set(idx, value); }

    [Differentiable]
    public void setv(vector<int,D> idx, T value) { primal.setv(idx, value); }

    [BackwardDerivativeOf(set)]
    public void set_bwd(int[D] idx, inout DifferentialPair<T> grad) { grad = diffPair(grad.p, d_in.get(idx)); }

    [BackwardDerivativeOf(setv)]
    public void setv_bwd(vector<int,D> idx, inout DifferentialPair<T> grad) { grad = diffPair(grad.p, d_in.getv(idx)); }

    public void load<let SliceD : int>(ContextND<D - SliceD> ctx, out GradInOutTensor<T, SliceD> value)
    {
        primal.load(ctx, value.primal);
        d_in.load(ctx, value.d_in);
        d_out.load(ctx, value.d_out);
    }
}

public extension<T: __BuiltinFloatingPointType> Tensor<T, 2>
{
    public T sample(float2 uv)
    {
        int2 shape = int2(shape[1], shape[0]);
        int2 maxidx = shape - 1;
        int2 texel = int2(uv * shape);
        float2 frac = frac(uv * shape);

        T c00 = getv(texel);
        T c10 = getv(min(texel + int2(1, 0),maxidx));
        T c01 = getv(min(texel + int2(0, 1),maxidx));
        T c11 = getv(min(texel + int2(1, 1),maxidx));

        return lerp(lerp(c00, c10, (T)frac.x), lerp(c01, c11, (T)frac.x), (T)frac.y);
    }
}

public extension<T: __BuiltinFloatingPointType, let N:int> Tensor<vector<T,N>, 2>
{
    public vector<T,N> sample(float2 uv)
    {
        int2 shape = int2(shape[1], shape[0]);
        int2 maxidx = shape - 1;
        int2 texel = int2(uv * shape);
        float2 frac = frac(uv * shape);

        vector<T,N> c00 = getv(texel);
        vector<T,N> c10 = getv(min(texel + int2(1, 0),maxidx));
        vector<T,N> c01 = getv(min(texel + int2(0, 1),maxidx));
        vector<T,N> c11 = getv(min(texel + int2(1, 1),maxidx));

        return lerp(lerp(c00, c10, (T)frac.x), lerp(c01, c11, (T)frac.x), (T)frac.y);
    }
}
