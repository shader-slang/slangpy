// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

implementing slangpy;

// Public interface for a read only tensor. This can be used as the argument
// to a function (or a generic constraint) to indicate it should take any
// type of read-only tensor.
public interface IDiffTensor<T : IDifferentiable, let D : int>
{
    T _read_primal_each<each I : __BuiltinIntegerType>(expand each I idx);
    void _write_grad_each<each I : __BuiltinIntegerType>(T.Differential grad, expand each I idx);
    void _accumulate_grad_each<each I : __BuiltinIntegerType>(T.Differential grad, expand each I idx);

    [Differentiable]
    public T load<I : __BuiltinIntegerType>(I idx[D]);
    [Differentiable]
    public T load<I : __BuiltinIntegerType>(vector<I, D> idx);

    public __subscript<I : __BuiltinIntegerType>(I indices[D])->T
    {
        [Differentiable] get;
    }
    public __subscript<I : __BuiltinIntegerType>(vector<I, D> indices)->T
    {
        [Differentiable] get;
    }

    public property shape : uint[D];
}

// Public interface for a write only differentiable tensor. This can be used as the argument
// to a function (or a generic constraint) to indicate it should take any
// type of write-only tensor.
public interface IWDiffTensor<T : IDifferentiable, let D : int>
{
    void _write_primal_each<each I : __BuiltinIntegerType>(T value, expand each I idx);
    T.Differential _read_grad_each<each I : __BuiltinIntegerType>(expand each I idx);

    [Differentiable]
    public void store<I : __BuiltinIntegerType>(I idx[D], T value);
    [Differentiable]
    public void store<I : __BuiltinIntegerType>(vector<I, D> idx, T value);

    public __subscript<I : __BuiltinIntegerType>(I indices[D])->T
    {
        [Differentiable] set;
    }
    public __subscript<I : __BuiltinIntegerType>(vector<I, D> indices)->T
    {
        [Differentiable] set;
    }

    public property shape : uint[D];
}

// Public interface for a read-write differentiable tensor. This can be used as the argument
// to a function (or a generic constraint) to indicate it should take any
// type of read-write tensor.
public interface IRWDiffTensor<T : IDifferentiable, let D : int>
{
    T _read_primal_each<each I : __BuiltinIntegerType>(expand each I idx);
    void _write_grad_each<each I : __BuiltinIntegerType>(T.Differential grad, expand each I idx);
    public void _accumulate_grad_each<each I : __BuiltinIntegerType>(T.Differential grad, expand each I idx);

    [Differentiable]
    public T load<I : __BuiltinIntegerType>(I idx[D]);
    [Differentiable]
    public T load<I : __BuiltinIntegerType>(vector<I, D> idx);

    void _write_primal_each<each I : __BuiltinIntegerType>(T value, expand each I idx);
    T.Differential _read_grad_each<each I : __BuiltinIntegerType>(expand each I idx);

    [Differentiable]
    public void store<I : __BuiltinIntegerType>(I idx[D], T value);
    [Differentiable]
    public void store<I : __BuiltinIntegerType>(vector<I, D> idx, T value);

    public __subscript<I : __BuiltinIntegerType>(I indices[D])->T
    {
        [Differentiable] get;
        [Differentiable] set;
    }
    public __subscript<I : __BuiltinIntegerType>(vector<I, D> indices)->T
    {
        [Differentiable] get;
        [Differentiable] set;
    }

    public property shape : uint[D];
}

// Extend IDiffTensor types to add their element type
public extension<T : IDifferentiable, let D : int, TensorType : IDiffTensor<T, D>> TensorType
    where T.Differential : IAtomicAddable
{
    typedef T ElementType;
}
public extension<T : IDifferentiable, let D : int, TensorType : IWDiffTensor<T, D>> TensorType
    where T.Differential : IAtomicAddable
{
    typedef T ElementType;
}
public extension<T : IDifferentiable, let D : int, TensorType : IRWDiffTensor<T, D>> TensorType
    where T.Differential : IAtomicAddable
{
    typedef T ElementType;
}


#define _SLANGPY_DIFFTENSOR_COMMON \
    public property shape : uint[D] { get { return _primal.shape; } }

#define _SLANGPY_DIFFTENSOR_LOAD                                                                                       \
    [ForceInline] [Differentiable]                                                                                     \
    [BackwardDerivative(_load_bwd_array)]                                                                              \
    public T load<I : __BuiltinIntegerType>(I idx[D])                                                                  \
    {                                                                                                                  \
        return _primal.load(idx);                                                                                      \
    }                                                                                                                  \
    [ForceInline] [Differentiable]                                                                                     \
    [BackwardDerivative(_load_bwd_vector)]                                                                             \
    public T load<I : __BuiltinIntegerType>(vector<I, D> idx)                                                          \
    {                                                                                                                  \
        return _primal.load(idx);                                                                                      \
    }                                                                                                                  \
    [ForceInline] T _read_primal_each<each I : __BuiltinIntegerType>(expand each I idx)                                \
    {                                                                                                                  \
        return _primal._read_each(expand each idx);                                                                    \
    }                                                                                                                  \
    [ForceInline] void _write_grad_each<each I : __BuiltinIntegerType>(T.Differential grad, expand each I idx)         \
    {                                                                                                                  \
        _grad_out._write_each(grad, expand each idx);                                                                  \
    }                                                                                                                  \
    [ForceInline] public void _accumulate_grad_each<each I : __BuiltinIntegerType>(T.Differential grad, expand each I idx) \
    {                                                                                                                  \
        _grad_out._accumulate_each(grad, expand each idx);                                                             \
    }

#define _SLANGPY_DIFFTENSOR_LOAD_BWD \
    [ForceInline] void _load_bwd_array<I : __BuiltinIntegerType>(I idx[D], T.Differential grad)\
    {\
        _grad_out.add(idx, grad);\
    }\
    [ForceInline] void _load_bwd_vector<I : __BuiltinIntegerType>(vector<I, D> idx, T.Differential grad)\
    {\
        _grad_out.add(idx, grad);\
    }

#define _SLANGPY_DIFFTENSOR_STORE_BWD \
    [ForceInline] void _store_bwd_array<I : __BuiltinIntegerType>(I idx[D], inout DifferentialPair<T> grad)\
    {\
        grad = diffPair(grad.p, _grad_in.load(idx));\
    }\
    [ForceInline] void _store_bwd_vector<I : __BuiltinIntegerType>(vector<I, D> idx, inout DifferentialPair<T> grad)\
    {\
        grad = diffPair(grad.p, _grad_in.load(idx));\
    }

#define _SLANGPY_DIFFTENSOR_STORE                                                                                      \
    [ForceInline] [Differentiable]                                                                                     \
    [BackwardDerivative(_store_bwd_array)]                                                                             \
    public void store<I : __BuiltinIntegerType>(I idx[D], T value)                                                     \
    {                                                                                                                  \
        _primal.store(idx, value);                                                                                     \
    }                                                                                                                  \
    [ForceInline] [Differentiable]                                                                                     \
    [BackwardDerivative(_store_bwd_vector)]                                                                            \
    public void store<I : __BuiltinIntegerType>(vector<I, D> idx, T value)                                             \
    {                                                                                                                  \
        _primal.store(idx, value);                                                                                     \
    }                                                                                                                  \
    [ForceInline] void _write_primal_each<each I : __BuiltinIntegerType>(T value, expand each I idx)                   \
    {                                                                                                                  \
        _primal._write_each(value, expand each idx);                                                                   \
    }                                                                                                                  \
    [ForceInline] T.Differential _read_grad_each<each I : __BuiltinIntegerType>(expand each I idx)                     \
    {                                                                                                                  \
        return _grad_in._read_each(expand each idx);                                                                   \
    }

#define _SLANGPY_DIFFTENSOR_SLANGPYSTORE                                                                               \
    [Differentiable]                                                                                                   \
    public void __slangpy_store(ContextND<D> context, in T value)                                                      \
    {                                                                                                                  \
        this.store(context.call_id, value);                                                                            \
    }                                                                                                                  \
    [Differentiable]                                                                                                   \
    public void __slangpy_store<let N : int>(ContextND<D - 1> context, in T[N] value)                                  \
    {                                                                                                                  \
        int idx[D];                                                                                                    \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < D - 1; ++i)                                                                                \
            idx[i] = context.call_id[i];                                                                               \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < N; i++) {                                                                                  \
            idx[D - 1] = i;                                                                                            \
            this.store(idx, value[i]);                                                                                 \
        }                                                                                                              \
    }                                                                                                                  \
    [Differentiable]                                                                                                   \
    public void __slangpy_store<let N : int>(ContextND<D - 1> context, in vector<T, N> value)                          \
    {                                                                                                                  \
        int idx[D];                                                                                                    \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < D - 1; ++i)                                                                                \
            idx[i] = context.call_id[i];                                                                               \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < N; i++) {                                                                                  \
            idx[D - 1] = i;                                                                                            \
            this.store(idx, value[i]);                                                                                 \
        }                                                                                                              \
    }                                                                                                                  \
    [Differentiable]                                                                                                   \
    public void __slangpy_store<let M : int, let N : int>(ContextND<D - 2> context, in T value[M][N])                  \
    {                                                                                                                  \
        int idx[D];                                                                                                    \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < D - 2; ++i)                                                                                \
            idx[i] = context.call_id[i];                                                                               \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < M; i++) {                                                                                  \
            [ForceUnroll]                                                                                              \
            for (int j = 0; j < N; j++) {                                                                              \
                idx[D - 2] = i;                                                                                        \
                idx[D - 1] = j;                                                                                        \
                this.store(idx, value[i][j]);                                                                          \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    [Differentiable]                                                                                                   \
    public void __slangpy_store<let M : int, let N : int>(ContextND<D - 2> context, in matrix<T, M, N> value)          \
    {                                                                                                                  \
        int idx[D];                                                                                                    \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < D - 2; ++i)                                                                                \
            idx[i] = context.call_id[i];                                                                               \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < M; i++) {                                                                                  \
            [ForceUnroll]                                                                                              \
            for (int j = 0; j < N; j++) {                                                                              \
                idx[D - 2] = i;                                                                                        \
                idx[D - 1] = j;                                                                                        \
                this.store(idx, value[i][j]);                                                                          \
            }                                                                                                          \
        }                                                                                                              \
    }

#define _SLANGPY_DIFFTENSOR_SLANGPYLOAD                                                                                \
    [Differentiable]                                                                                                   \
    public void __slangpy_load(ContextND<D> context, out T value)                                                      \
    {                                                                                                                  \
        value = this.load(context.call_id);                                                                            \
    }                                                                                                                  \
    [Differentiable]                                                                                                   \
    public void __slangpy_load<let N : int>(ContextND<D - 1> ctx, out T[N] value)                                      \
    {                                                                                                                  \
        int idx[D];                                                                                                    \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < D - 1; ++i)                                                                                \
            idx[i] = ctx.call_id[i];                                                                                   \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < N; i++) {                                                                                  \
            idx[D - 1] = i;                                                                                            \
            value[i] = this.load(idx);                                                                                 \
        }                                                                                                              \
    }                                                                                                                  \
    [Differentiable]                                                                                                   \
    public void __slangpy_load<let N : int>(ContextND<D - 1> ctx, out vector<T, N> value)                              \
    {                                                                                                                  \
        int idx[D];                                                                                                    \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < D - 1; ++i)                                                                                \
            idx[i] = ctx.call_id[i];                                                                                   \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < N; i++) {                                                                                  \
            idx[D - 1] = i;                                                                                            \
            value[i] = this.load(idx);                                                                                 \
        }                                                                                                              \
    }                                                                                                                  \
    [Differentiable]                                                                                                   \
    public void __slangpy_load<let M : int, let N : int>(ContextND<D - 2> context, out T value[M][N])                  \
    {                                                                                                                  \
        int idx[D];                                                                                                    \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < D - 2; ++i)                                                                                \
            idx[i] = context.call_id[i];                                                                               \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < M; i++) {                                                                                  \
            [ForceUnroll]                                                                                              \
            for (int j = 0; j < N; j++) {                                                                              \
                idx[D - 2] = i;                                                                                        \
                idx[D - 1] = j;                                                                                        \
                value[i][j] = this.load(idx);                                                                          \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    [Differentiable]                                                                                                   \
    public void __slangpy_load<let M : int, let N : int>(ContextND<D - 2> context, out matrix<T, M, N> value)          \
    {                                                                                                                  \
        int idx[D];                                                                                                    \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < D - 2; ++i)                                                                                \
            idx[i] = context.call_id[i];                                                                               \
        [ForceUnroll]                                                                                                  \
        for (int i = 0; i < M; i++) {                                                                                  \
            [ForceUnroll]                                                                                              \
            for (int j = 0; j < N; j++) {                                                                              \
                idx[D - 2] = i;                                                                                        \
                idx[D - 1] = j;                                                                                        \
                value[i][j] = this.load(idx);                                                                          \
            }                                                                                                          \
        }                                                                                                              \
    }

// A read only differentiable tensor, that internally contains a primal tensor, and a gradient output tensor.
public struct DiffTensor<T : IDifferentiable, let D : int> : IDiffTensor<T, D> where T.Differential : IAtomicAddable
{
    // Underlying data storage
    public Tensor<T, D> _primal;
    public AtomicTensor<T.Differential, D> _grad_out;

    _SLANGPY_DIFFTENSOR_COMMON
    _SLANGPY_DIFFTENSOR_LOAD
    _SLANGPY_DIFFTENSOR_LOAD_BWD
    _SLANGPY_DIFFTENSOR_SLANGPYLOAD

    public __subscript<I : __BuiltinIntegerType>(I indices[D])->T
    {
        [Differentiable] get { return load(indices); }
    }
     public __subscript<I : __BuiltinIntegerType>(vector<I, D> indices)->T
     {
         [Differentiable] get { return load(indices); }
    }

    public void __slangpy_load(Context0D context, out This value)
    {
        value = this;
    }
    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out DiffTensor<T, SliceD> value)
    {
        _primal.__slangpy_load<SliceD>(ctx, value._primal);
        _grad_out.__slangpy_load<SliceD>(ctx, value._grad_out);
    }

}

// A write only differentiable tensor, that internally contains a primal tensor, and a gradient input tensor.
public struct WDiffTensor<T : IDifferentiable, let D : int> : IWDiffTensor<T, D>
{
    // Underlying data storage
    public RWTensor<T, D> _primal;
    public Tensor<T.Differential, D> _grad_in;

    _SLANGPY_DIFFTENSOR_COMMON
    _SLANGPY_DIFFTENSOR_STORE
    _SLANGPY_DIFFTENSOR_STORE_BWD
    _SLANGPY_DIFFTENSOR_SLANGPYSTORE

    public __subscript<I : __BuiltinIntegerType>(I indices[D])->T
    {
        [Differentiable] set { return store(indices, newValue); }
    }
    public __subscript<I : __BuiltinIntegerType>(vector<I, D> indices)->T
    {
        [Differentiable] set { return store(indices, newValue); }
    }

    public void __slangpy_load(Context0D context, out This value)
    {
        value = this;
    }
    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out WDiffTensor<T, SliceD> value)
    {
        _primal.__slangpy_load<SliceD>(ctx, value._primal);
        _grad_in.__slangpy_load<SliceD>(ctx, value._grad_in);
    }
}

// A read-write differentiable tensor, that internally contains a primal tensor, and both gradient tensors.
public struct RWDiffTensor<T : IDifferentiable, let D : int> : IRWDiffTensor<T, D> where T.Differential : IAtomicAddable
{
    public RWTensor<T, D> _primal;
    public Tensor<T.Differential, D> _grad_in;
    public AtomicTensor<T.Differential, D> _grad_out;

    _SLANGPY_DIFFTENSOR_COMMON
    _SLANGPY_DIFFTENSOR_LOAD
    _SLANGPY_DIFFTENSOR_LOAD_BWD
    _SLANGPY_DIFFTENSOR_STORE
    _SLANGPY_DIFFTENSOR_STORE_BWD
    _SLANGPY_DIFFTENSOR_SLANGPYLOAD
    _SLANGPY_DIFFTENSOR_SLANGPYSTORE

    public __subscript<I : __BuiltinIntegerType>(I indices[D])->T
    {
        [Differentiable] get { return load(indices); }
        [Differentiable] set { return store(indices, newValue); }
    }
    public __subscript<I : __BuiltinIntegerType>(vector<I, D> indices)->T
    {
        [Differentiable] get { return load(indices); }
        [Differentiable] set { return store(indices, newValue); }
    }

    public void __slangpy_load(Context0D context, out This value)
    {
        value = this;
    }
    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out RWDiffTensor<T, SliceD> value)
    {
        _primal.__slangpy_load<SliceD>(ctx, value._primal);
        _grad_in.__slangpy_load<SliceD>(ctx, value._grad_in);
        _grad_out.__slangpy_load<SliceD>(ctx, value._grad_out);
    }
}

// Bespoke loads for directly reading underlying structured buffers from tensor
#ifndef __TARGET_CUDA__
public extension<T: IDifferentiable, let D : int> DiffTensor<T, D> where T.Differential : IAtomicAddable
{
    public void __slangpy_load(Context0D context, out StructuredBuffer<T> value)
    {
        this._primal.__slangpy_load(context, value);
    }
}
public extension<T: IDifferentiable, let D : int> WDiffTensor<T, D> where T.Differential : IAtomicAddable
{
    public void __slangpy_load(Context0D context, out RWStructuredBuffer<T> value)
    {
        this._primal.__slangpy_load(context, value);
    }
}
public extension<T: IDifferentiable, let D : int> RWDiffTensor<T, D> where T.Differential : IAtomicAddable
{
    public void __slangpy_load(Context0D context, out RWStructuredBuffer<T> value)
    {
        this._primal.__slangpy_load(context, value);
    }
}
#endif

/*
//TODO: DiffTensor sample

Having real trouble getting these to work, regardless of whether using interface extensions as below, or directly extending
the concrete DiffTensor/RWDiffTensor types. 2 problems:
- The ElementType (added in extension above) does not exist, where the same technique worked for Tensor
- Even if I switch to concrete types and just explicitly add the ElementType to DiffTensor/RWDiffTensor,
  the vector version doesn't work because it says vector<T,N> is not atomic addable. However it is made atomic
  addable by extending the ISizedArray extension of IAtomicAddable in atomics.slang.

#define _SLANGPY_DIFFTENSOR_SAMPLE                                                                                     \
    public ElementType sample(float2 uv)                                                                               \
    {                                                                                                                  \
        int2 _shape = int2(this.shape[1], this.shape[0]);                                                              \
        int2 idx0, idx1;                                                                                               \
        float2 frac;                                                                                                   \
        _get_sample_indices(uv, _shape, idx0, idx1, frac);                                                             \
        ElementType c00 = this.load(idx0);                                                                             \
        ElementType c10 = this.load(int2(idx1.x, idx0.y));                                                             \
        ElementType c01 = this.load(int2(idx0.x, idx1.y));                                                             \
        ElementType c11 = this.load(idx1);                                                                             \
        return _bilerp(c00, c10, c01, c11, frac);                                                                      \
    }

public extension<T : __BuiltinFloatingPointType, TensorType : IDiffTensor<T, 2>> TensorType
{
    _SLANGPY_DIFFTENSOR_SAMPLE
}

public extension<T : __BuiltinFloatingPointType, TensorType : IRWDiffTensor<T, 2>> TensorType
{
    _SLANGPY_DIFFTENSOR_SAMPLE
}

public extension<T : __BuiltinFloatingPointType, let N : int, TensorType : IDiffTensor<vector<T, N>, 2>> TensorType
{
    _SLANGPY_DIFFTENSOR_SAMPLE
}

public extension<T : __BuiltinFloatingPointType, let N : int, TensorType : IRWDiffTensor<vector<T, N>, 2>> TensorType
{
    _SLANGPY_DIFFTENSOR_SAMPLE
}
*/
