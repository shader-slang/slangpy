// SPDX-License-Identifier: Apache-2.0
implementing slangpy;

public interface ITensor<T : IDifferentiable, let D : int>
{
    [Differentiable] public T get(int idx[D]);

    __generic<each Is : IInteger>
    __subscript(expand each Is indices) -> T
    {
        [BackwardDifferentiable] get;
    }
}
public interface IRWTensor<T : IDifferentiable, let D : int> : ITensor<T, D>
{
    [Differentiable] public void set(int idx[D], T value);
}

public struct StridedLayout<let D : int>
{
    public int offset;
    public int[D] strides;

    public int at(int[D] idx)
    {
        int result = offset;
        for (int i = 0; i < D; ++i)
            result += strides[i] * idx[i];
        return result;
    }

    public __generic<let SliceD : int>
    StridedLayout<SliceD> slice(int[D - SliceD] idx)
    {
        StridedLayout<SliceD> result;

        for (int i = 0; i < SliceD; ++i)
            result.strides[i] = strides[D - SliceD + i];

        result.offset = offset;
        for (int i = 0; i < D - SliceD; ++i)
            result.offset += strides[i] * idx[i];

        return result;
    }
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
        for (int i = 0; i < D; ++i)
            idxVec[i] = indices[i].toInt();
        return idxVec;
    }
}

// Workaround for slang bug #5900. These should move into extensions once they are safe
#define ITensorSubscript                                         \
    public __generic<each Is : IInteger>                         \
    __subscript(expand each Is indices)->T                       \
    {                                                            \
        [BackwardDifferentiable] get { return get(impl::makeIndex<D>(expand each indices)); }                                                        \
    }                                                            \
    public __generic<I : IInteger> \
    __subscript(ISizedArray<I, D> indices) -> T \
    { \
        [BackwardDifferentiable] get { return get(impl::makeIndex<D, I>(indices)); } \
    }

#define IRWTensorSubscript                                                 \
    public __generic<each Is : IInteger>                                   \
    __subscript(expand each Is indices)->T                                 \
    {                                                                      \
        [BackwardDifferentiable] get { return get(impl::makeIndex<D>(expand each indices)); }                                                                  \
        [nonmutating, BackwardDifferentiable] set { return set(impl::makeIndex<D>(expand each indices), newValue); }                                                                  \
    }                                                                      \
    public __generic<I : IInteger> \
    __subscript(ISizedArray<I, D> indices) -> T \
    { \
        [BackwardDifferentiable] get { return get(impl::makeIndex<D, I>(indices)); } \
        [nonmutating, BackwardDifferentiable] set { return set(impl::makeIndex<D, I>(indices), newValue); } \
    }

#define ITensorLoads                                                                        \
    public __generic<each Ts : IInteger>                                                    \
    __subscript(expand each Ts indices)->T                                                  \
    {                                                                                       \
        [BackwardDifferentiable] get { return get(impl::makeIndex<D>(expand each indices)); }                                                                                   \
    }                                                                                       \
    [Differentiable]                                                                        \
    public void load(ContextND<D> ctx, out T value)                                         \
    {                                                                                       \
        value = get(ctx.call_id);                                                           \
    }                                                                                       \
    public void load(Context0D context, out This value)                                     \
    {                                                                                       \
        value = this;                                                                       \
    }                                                                                       \
    [Differentiable]                                                                        \
    public void load<let N : int>(ContextND<D - 1> ctx, out T[N] value)                     \
    {                                                                                       \
        int idx[D];                                                                         \
        /*[ForceUnroll]*/                                                                   \
        [MaxIters(8)] /* Workaround for slang bug #5917 */                                  \
        for (int i = 0; i < D - 1; ++i)                                                     \
            idx[i] = ctx.call_id[i];                                                        \
                                                                                            \
        [ForceUnroll]                                                                       \
        for (int i = 0; i < N; i++) {                                                       \
            idx[D - 1] = i;                                                                 \
            value[i] = get(idx);                                                            \
        }                                                                                   \
    }                                                                                       \
    [Differentiable]                                                                        \
    public void load<let N : int>(ContextND<D - 1> ctx, out vector<T,N> value)              \
    {                                                                                       \
        int idx[D];                                                                         \
        /*[ForceUnroll]*/                                                                   \
        [MaxIters(8)] /* Workaround for slang bug #5917 */                                  \
        for (int i = 0; i < D - 1; ++i)                                                     \
            idx[i] = ctx.call_id[i];                                                        \
                                                                                            \
        [ForceUnroll]                                                                       \
        for (int i = 0; i < N; i++) {                                                       \
            idx[D - 1] = i;                                                                 \
            value[i] = get(idx);                                                            \
        }                                                                                   \
    }                                                                                       \
                                                                                            \
    [Differentiable]                                                                        \
    public void load<let M : int, let N : int>(ContextND<D - 2> context, out T value[M][N]) \
    {                                                                                       \
        int idx[D];                                                                         \
        /*[ForceUnroll]*/                                                                   \
        [MaxIters(8)] /* Workaround for slang bug #5917 */                                  \
        for (int i = 0; i < D - 2; ++i)                                                     \
            idx[i] = context.call_id[i];                                                    \
        [ForceUnroll]                                                                       \
        for (int i = 0; i < M; i++) {                                                       \
            [ForceUnroll]                                                                   \
            for (int j = 0; j < N; j++) {                                                   \
                idx[D - 2] = i;                                                             \
                idx[D - 1] = j;                                                             \
                                                                                            \
                value[i][j] = get(idx);                                                     \
            }                                                                               \
        }                                                                                   \
    }


#define IRWTensorStores                                                                     \
    [Differentiable]                                                                        \
    public void store(ContextND<D> ctx, in T value)                                         \
    {                                                                                       \
        set(ctx.call_id, value);                                                            \
    }                                                                                       \
                                                                                            \
    [Differentiable]                                                                        \
    public void store<let N : int>(ContextND<D - 1> context, in T[N] value)                 \
    {                                                                                       \
        int idx[D];                                                                         \
        /*[ForceUnroll]*/                                                                   \
        [MaxIters(8)] /* Workaround for slang bug #5917 */                                  \
        for (int i = 0; i < D - 1; ++i)                                                     \
            idx[i] = context.call_id[i];                                                    \
        [ForceUnroll]                                                                       \
        for (int i = 0; i < N; i++) {                                                       \
            idx[D - 1] = i;                                                                 \
            set(idx, value[i]);                                                             \
        }                                                                                   \
    }                                                                                       \
                                                                                            \
    [Differentiable]                                                                        \
    public void store<let N : int>(ContextND<D - 1> context, in vector<T,N> value)          \
    {                                                                                       \
        int idx[D];                                                                         \
        /*[ForceUnroll]*/                                                                   \
        [MaxIters(8)] /* Workaround for slang bug #5917 */                                  \
        for (int i = 0; i < D - 1; ++i)                                                     \
            idx[i] = context.call_id[i];                                                    \
        [ForceUnroll]                                                                       \
        for (int i = 0; i < N; i++) {                                                       \
            idx[D - 1] = i;                                                                 \
            set(idx, value[i]);                                                             \
        }                                                                                   \
    }                                                                                       \
                                                                                            \
    [Differentiable]                                                                        \
    public void store<let M : int, let N : int>(ContextND<D - 2> context, in T value[M][N]) \
    {                                                                                       \
        int idx[D];                                                                         \
        /*[ForceUnroll]*/                                                                   \
        [MaxIters(8)] /* Workaround for slang bug #5917 */                                  \
        for (int i = 0; i < D - 2; ++i)                                                     \
            idx[i] = context.call_id[i];                                                    \
        [ForceUnroll]                                                                       \
        for (int i = 0; i < M; i++) {                                                       \
            [ForceUnroll]                                                                   \
            for (int j = 0; j < N; j++) {                                                   \
                idx[D - 2] = i;                                                             \
                idx[D - 1] = j;                                                             \
                                                                                            \
                set(idx, value[i][j]);                                                      \
            }                                                                               \
        }                                                                                   \
    }

public struct Tensor<T : IDifferentiable, let D : int> : ITensor<T, D>
{
    public StructuredBuffer<T> buffer;
    public StridedLayout<D> layout;
    public int[D] shape;

    [TreatAsDifferentiable]
    public T get(int[D] idx) { return buffer[layout.at(idx)]; }

    public void load<let SliceD : int>(ContextND<D - SliceD> ctx, out Tensor<T, SliceD> value)
    {
        value.buffer = buffer;
        value.layout = layout.slice<SliceD>(ctx.call_id);
    }

    ITensorSubscript
    ITensorLoads
}

public struct RWTensor<T : IDifferentiable, let D : int> : IRWTensor<T, D>
{
    public RWStructuredBuffer<T> buffer;
    public StridedLayout<D> layout;
    public int[D] shape;

    [TreatAsDifferentiable]
    public T get(int[D] idx) { return buffer[layout.at(idx)]; }

    [TreatAsDifferentiable]
    public void set(int idx[D], T value) { buffer[layout.at(idx)] = value; }

    public void load<let SliceD : int>(ContextND<D - SliceD> ctx, out RWTensor<T, SliceD> value)
    {
        value.buffer = buffer;
        value.layout = layout.slice<SliceD>(ctx.call_id);
    }

    IRWTensorSubscript
    ITensorLoads
    IRWTensorStores
}

public struct AtomicTensor<T, let D : int> : IRWTensor<T, D> where T : IDifferentiable, IAtomicAddable
{
    public RWByteAddressBuffer buffer;
    public StridedLayout<D> layout;
    public int[D] shape;

    [TreatAsDifferentiable]
    public T get(int[D] idx) { return buffer.Load<T>(layout.at(idx) * sizeof(T), sizeof(T)); }

    [TreatAsDifferentiable]
    public void set(int idx[D], T value) {
        T::atomicAdd(buffer, layout.at(idx) * sizeof(T), value);
        // TODO: Atomic adds are conservative and not always needed. If we inspect the layout
        // during load() and determine each call id sees a unique slice, we could do the Store below instead:
        //buffer.Store<T>(layout.at(idx) * sizeof(T), value, sizeof(T));
    }

    public void load<let SliceD : int>(ContextND<D - SliceD> ctx, out AtomicTensor<T, SliceD> value)
    {
        value.buffer = buffer;
        value.layout = layout.slice<SliceD>(ctx.call_id);
    }

    IRWTensorSubscript
    ITensorLoads
    IRWTensorStores
}

public struct GradOutTensor<T : IDifferentiable, let D : int> : ITensor<T, D> where T.Differential : IAtomicAddable
{
    public Tensor<T, D> primal;
    public AtomicTensor<T.Differential, D> d_out;

    [Differentiable]
    public T get(int[D] idx) { return primal.get(idx); }

    [BackwardDerivativeOf(get)]
    public void get_bwd(int[D] idx, T.Differential grad)
    {
        d_out.set(idx, grad);
    }

    public void load<let SliceD : int>(ContextND<D - SliceD> ctx, out GradOutTensor<T, SliceD> value)
    {
        primal.load(ctx, value.primal);
        d_out.load(ctx, value.d_out);
    }

    ITensorSubscript
    ITensorLoads
}

public struct GradInTensor<T : IDifferentiable, let D : int> : IRWTensor<T, D>
{
    public RWTensor<T, D> primal;
    public Tensor<T.Differential, D> d_in;

    [TreatAsDifferentiable]
    public T get(int[D] idx) { return primal.get(idx); }

    [Differentiable]
    public void set(int[D] idx, T value) { primal.set(idx, value); }

    [BackwardDerivativeOf(set)]
    public void set_bwd(int[D] idx, inout DifferentialPair<T> grad) { grad = diffPair(grad.p, d_in.get(idx)); }

    public void load<let SliceD : int>(ContextND<D - SliceD> ctx, out GradInTensor<T, SliceD> value)
    {
        primal.load(ctx, value.primal);
        d_in.load(ctx, value.d_in);
    }

    IRWTensorSubscript
    ITensorLoads
    IRWTensorStores
}

public struct GradInOutTensor<T : IDifferentiable, let D : int> : IRWTensor<T, D> where T.Differential : IAtomicAddable
{
    public RWTensor<T, D> primal;
    public AtomicTensor<T.Differential, D> d_out;
    public Tensor<T.Differential, D> d_in;

    [Differentiable]
    public T get(int[D] idx) { return primal.get(idx); }

    [BackwardDerivativeOf(get)]
    public void get_bwd(int[D] idx, T.Differential grad)
    {
        d_out.set(idx, grad);
    }

    [Differentiable]
    public void set(int[D] idx, T value) { primal.set(idx, value); }

    [BackwardDerivativeOf(set)]
    public void set_bwd(int[D] idx, inout DifferentialPair<T> grad) { grad = diffPair(grad.p, d_in.get(idx)); }

    public void load<let SliceD : int>(ContextND<D - SliceD> ctx, out GradInOutTensor<T, SliceD> value)
    {
        primal.load(ctx, value.primal);
        d_in.load(ctx, value.d_in);
        d_out.load(ctx, value.d_out);
    }

    IRWTensorSubscript
    ITensorLoads
    IRWTensorStores
}
