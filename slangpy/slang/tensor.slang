// SPDX-License-Identifier: Apache-2.0
__exported import staticarray;
__exported import atomics;
import "slangpy";

interface ITensor<T : IDifferentiable, let D : int>
{
    [Differentiable] T get(int idx[D]);

    __generic<each Is : IInteger>
    __subscript(expand each Is indices) -> T
    {
        [BackwardDifferentiable] get;
    }
}
interface IRWTensor<T : IDifferentiable, let D : int> : ITensor<T, D>
{
    [Differentiable] void set(int idx[D], T value);
}

struct StridedLayout<let D : int>
{
    int offset;
    int[D] strides;

    int at(int[D] idx)
    {
        int result = offset;
        for (int i = 0; i < D; ++i)
            result += strides[i] * idx[i];
        return result;
    }

    __generic<let SliceD : int>
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
#define ITensorSubscript \
    __generic<each Is : IInteger> \
    __subscript(expand each Is indices) -> T \
    { \
        [BackwardDifferentiable] get { return get(impl::makeIndex<D>(expand each indices)); } \
    } \
    __generic<I : IInteger> \
    __subscript(ISizedArray<I, D> indices) -> T \
    { \
        [BackwardDifferentiable] get { return get(impl::makeIndex<D, I>(indices)); } \
    }

#define IRWTensorSubscript \
    __generic<each Is : IInteger> \
    __subscript(expand each Is indices) -> T \
    { \
        [BackwardDifferentiable] get { return get(impl::makeIndex<D>(expand each indices)); } \
        [nonmutating, BackwardDifferentiable] set { return set(impl::makeIndex<D>(expand each indices), newValue); } \
    } \
    __generic<I : IInteger> \
    __subscript(ISizedArray<I, D> indices) -> T \
    { \
        [BackwardDifferentiable] get { return get(impl::makeIndex<D, I>(indices)); } \
        [nonmutating, BackwardDifferentiable] set { return set(impl::makeIndex<D, I>(indices), newValue); } \
    }

#define ITensorLoads \
    __generic<each Ts : IInteger> \
    __subscript(expand each Ts indices) -> T \
    { \
        [BackwardDifferentiable] get { return get(impl::makeIndex<D>(expand each indices)); } \
    } \
    [Differentiable] \
    void load(ContextND<D> ctx, out T value) \
    { \
        value = get(ctx.call_id); \
    } \
    void load(Context0D context, out This value) \
    { \
        value = this; \
    } \
    [Differentiable] \
    void load<let N : int>(ContextND<D - 1> ctx, out T[N] value) \
    { \
        int idx[D]; \
        /*[ForceUnroll]*/ \
        [MaxIters(8)] /* Workaround for slang bug #5917 */ \
        for (int i = 0; i < D - 1; ++i) \
            idx[i] = ctx.call_id[i]; \
    \
        [ForceUnroll] \
        for (int i = 0; i < N; i++) { \
            idx[D - 1] = i; \
            value[i] = get(idx); \
        } \
    } \
    \
    [Differentiable] \
    void load<let M : int, let N : int>(ContextND<D - 2> context, out T value[M][N]) \
    { \
        int idx[D]; \
        /*[ForceUnroll]*/ \
        [MaxIters(8)] /* Workaround for slang bug #5917 */ \
        for (int i = 0; i < D - 2; ++i) \
            idx[i] = context.call_id[i]; \
        [ForceUnroll] \
        for (int i = 0; i < M; i++) { \
            [ForceUnroll] \
            for (int j = 0; j < N; j++) { \
                idx[D - 2] = i; \
                idx[D - 1] = j; \
    \
                value[i][j] = get(idx); \
            } \
        } \
    }

#define IRWTensorStores \
    [Differentiable] \
    void store(ContextND<D> ctx, in T value) \
    { \
        set(ctx.call_id, value); \
    } \
    \
    [Differentiable] \
    void store<let N : int>(ContextND<D - 1> context, in T[N] value) \
    { \
        int idx[D]; \
        /*[ForceUnroll]*/ \
        [MaxIters(8)] /* Workaround for slang bug #5917 */ \
        for (int i = 0; i < D - 1; ++i) \
            idx[i] = context.call_id[i]; \
        [ForceUnroll] \
        for (int i = 0; i < N; i++) { \
            idx[D - 1] = i; \
            set(idx, value[i]); \
        } \
    } \
    \
    [Differentiable] \
    void store<let M : int, let N : int>(ContextND<D - 2> context, in T value[M][N]) \
    { \
        int idx[D]; \
        /*[ForceUnroll]*/ \
        [MaxIters(8)] /* Workaround for slang bug #5917 */ \
        for (int i = 0; i < D - 2; ++i) \
            idx[i] = context.call_id[i]; \
        [ForceUnroll] \
        for (int i = 0; i < M; i++) { \
            [ForceUnroll] \
            for (int j = 0; j < N; j++) { \
                idx[D - 2] = i; \
                idx[D - 1] = j; \
                \
                set(idx, value[i][j]); \
            } \
        } \
    }

struct Tensor<T : IDifferentiable, let D : int> : ITensor<T, D>
{
    StructuredBuffer<T> buffer;
    StridedLayout<D> layout;
    int[D] shape;

    [TreatAsDifferentiable]
    T get(int[D] idx) { return buffer[layout.at(idx)]; }

    void load<let SliceD : int>(ContextND<D - SliceD> ctx, out Tensor<T, SliceD> value)
    {
        value.buffer = buffer;
        value.layout = layout.slice<SliceD>(ctx.call_id);
    }

    ITensorSubscript
    ITensorLoads
}

struct RWTensor<T : IDifferentiable, let D : int> : IRWTensor<T, D>
{
    RWStructuredBuffer<T> buffer;
    StridedLayout<D> layout;
    int[D] shape;

    [TreatAsDifferentiable]
    T get(int[D] idx) { return buffer[layout.at(idx)]; }

    [TreatAsDifferentiable]
    void set(int idx[D], T value) { buffer[layout.at(idx)] = value; }

    void load<let SliceD : int>(ContextND<D - SliceD> ctx, out RWTensor<T, SliceD> value)
    {
        value.buffer = buffer;
        value.layout = layout.slice<SliceD>(ctx.call_id);
    }

    IRWTensorSubscript
    ITensorLoads
    IRWTensorStores
}

struct AtomicTensor<T, let D : int> : IRWTensor<T, D> where T : IDifferentiable, IAtomicAddable
{
    RWByteAddressBuffer buffer;
    StridedLayout<D> layout;
    int[D] shape;

    [TreatAsDifferentiable]
    T get(int[D] idx) { return buffer.Load<T>(layout.at(idx) * sizeof(T), sizeof(T)); }

    [TreatAsDifferentiable]
    void set(int idx[D], T value) {
        T::atomicAdd(buffer, layout.at(idx) * sizeof(T), value);
        // TODO: Atomic adds are conservative and not always needed. If we inspect the layout
        // during load() and determine each call id sees a unique slice, we could do the Store below instead:
        //buffer.Store<T>(layout.at(idx) * sizeof(T), value, sizeof(T));
    }

    void load<let SliceD : int>(ContextND<D - SliceD> ctx, out AtomicTensor<T, SliceD> value)
    {
        value.buffer = buffer;
        value.layout = layout.slice<SliceD>(ctx.call_id);
    }

    IRWTensorSubscript
    ITensorLoads
    IRWTensorStores
}

struct GradOutTensor<T : IDifferentiable, let D : int> : ITensor<T, D> where T.Differential : IAtomicAddable
{
    Tensor<T, D> primal;
    AtomicTensor<T.Differential, D> d_out;
    
    [Differentiable]
    T get(int[D] idx) { return primal.get(idx); }

    [BackwardDerivativeOf(get)]
    void get_bwd(int[D] idx, T.Differential grad)
    {
        d_out.set(idx, grad);
    }

    void load<let SliceD : int>(ContextND<D - SliceD> ctx, out GradOutTensor<T, SliceD> value)
    {
        primal.load(ctx, value.primal);
        d_out.load(ctx, value.d_out);
    }

    ITensorSubscript
    ITensorLoads
}

struct GradInTensor<T : IDifferentiable, let D : int> : IRWTensor<T, D>
{
    RWTensor<T, D> primal;
    Tensor<T.Differential, D> d_in;

    [TreatAsDifferentiable]
    T get(int[D] idx) { return primal.get(idx); }

    [Differentiable]
    void set(int[D] idx, T value) { primal.set(idx, value); }

    [BackwardDerivativeOf(set)]
    void set_bwd(int[D] idx, inout DifferentialPair<T> grad) { grad = diffPair(grad.p, d_in.get(idx)); }

    void load<let SliceD : int>(ContextND<D - SliceD> ctx, out GradInTensor<T, SliceD> value)
    {
        primal.load(ctx, value.primal);
        d_in.load(ctx, value.d_in);
    }

    IRWTensorSubscript
    ITensorLoads
    IRWTensorStores
}

struct GradInOutTensor<T : IDifferentiable, let D : int> : IRWTensor<T, D> where T.Differential : IAtomicAddable
{
    RWTensor<T, D> primal;
    AtomicTensor<T.Differential, D> d_out;
    Tensor<T.Differential, D> d_in;

    [Differentiable]
    T get(int[D] idx) { return primal.get(idx); }

    [BackwardDerivativeOf(get)]
    void get_bwd(int[D] idx, T.Differential grad)
    {
        d_out.set(idx, grad);
    }

    [Differentiable]
    void set(int[D] idx, T value) { primal.set(idx, value); }

    [BackwardDerivativeOf(set)]
    void set_bwd(int[D] idx, inout DifferentialPair<T> grad) { grad = diffPair(grad.p, d_in.get(idx)); }

    void load<let SliceD : int>(ContextND<D - SliceD> ctx, out GradInOutTensor<T, SliceD> value)
    {
        primal.load(ctx, value.primal);
        d_in.load(ctx, value.d_in);
        d_out.load(ctx, value.d_out);
    }

    IRWTensorSubscript
    ITensorLoads
    IRWTensorStores
}
