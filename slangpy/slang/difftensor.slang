// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

implementing slangpy;


#if 1

int _idx<I: IInteger, let N : int>(I[N] indices, uint stride[N], uint offset)
{
    int idx = 0;
    [ForceUnroll]
    for (int i = 0; i < N; i++) { idx += indices[i].toUInt() * stride[i]; }
    return idx + offset;
}
int _idx<I : IInteger, let N : int>(vector<I, N> indices, uint stride[N], uint offset)
{
    int idx = 0;
    int end = N - 1;
    [ForceUnroll]
    for (int i = 0; i < N; i++) { idx += indices[end - i].toUInt() * stride[i]; }
    return idx + offset;
}
int _idx<each I : IInteger, let N : int>(expand each I indices, uint stride[N], uint offset)
{
    // NOTE: Can probably optimize this to avoid the intermediate array allocation
    uint[N] idxVec;
    int i = 0;
    expand idxVec[i++] = (each indices).toUInt();
    return _idx(idxVec, stride, offset);
}

void _slice<let D : int, let SliceD : int>(int[D - SliceD] idx, uint stride[D], uint offset, out uint new_stride[SliceD], out uint new_offset)
{
    [ForceUnroll]
    for (int i = 0; i < SliceD; ++i)
        new_stride[i] = stride[D - SliceD + i];

    new_offset = offset;
    [ForceUnroll]
    for (int i = 0; i < D - SliceD; ++i)
        new_offset += stride[i] * idx[i];
}

// Public interface for a read only tensor. This can be used as the argument
// to a function (or a generic constraint) to indicate it should take any
// type of read-only tensor.
public interface IRODiffTensor<T : IDifferentiable, let D : int>
{
    [Differentiable]
    public T load<I : IInteger>(I idx[D]);
    [Differentiable]
    public T load<I : IInteger>(vector<I, D> idx);

    //NOTE: These prevent things compiling - can't resolve them for some reason
    //[Differentiable]
    //public T load<each I : IInteger>(expand each I indices);

    public property shape : uint[D];
}

// Type alias for a read only tensor interface This can be used as the argument
// to a function (or a generic constraint) to indicate it should take any
// type of read-only tensor.
public typealias IDiffTensor<T : IDifferentiable, let D : int> = IRODiffTensor<T, D>;

// Public interface for a write only tensor. This can be used as the argument
// to a function (or a generic constraint) to indicate it should take any
// type of write-only tensor.
public interface IWODiffTensor<T : IDifferentiable, let D : int>
{
    [Differentiable]
    public void store<I : IInteger>(I idx[D], T value);
    [Differentiable]
    public void store<I : IInteger>(vector<I, D> idx, T value);

    // NOTE: These prevent things compiling - can't resolve them for some reason
    //[Differentiable]
    //public void store<each I : IInteger>(expand each I indices, T value);

    public property shape : uint[D];
}

// Public interface for a read-write tensor. This can be used as the argument
// to a function (or a generic constraint) to indicate it should take any
// type of read-write tensor.
public interface IRWDiffTensor<T : IDifferentiable, let D : int> : IRODiffTensor<T, D>, IWODiffTensor<T, D>
{
}

#define _SLANGPY_COMMON_DIFFTENSOR_IMPL \
    public property shape : uint[D] { get { return _primal.shape; } }

    /*
#define _SLANGPY_DIFFDATA_STORE_IMPL                                                                                   \
    [Differentiable]                                                                                                   \
    public void store<I : IInteger>(I idx[D], T value)                                                                 \
    {                                                                                                                  \
        write_buffer(_idx(idx, _strides, _offset), value);                                                             \
    }                                                                                                                  \
    [Differentiable]                                                                                                   \
    public void store<I : IInteger>(vector<I, D> idx, T value)                                                         \
    {                                                                                                                  \
        write_buffer(_idx(idx, _strides, _offset), value);                                                             \
    }                                                                                                                  \
    [Differentiable]                                                                                                   \
    public void store<each I : IInteger>(expand each I indices, T value)                                               \
    {                                                                                                                  \
        write_buffer(_idx(expand each indices, _strides, _offset), value);                                             \
    }

#define _SLANGPY_DIFFDATA_LOAD_IMPL                                                                                    \
    [Differentiable]                                                                                                   \
    public T load<I : IInteger>(I idx[D])                                                                              \
    {                                                                                                                  \
        return read_buffer(_idx(idx, _strides, _offset));                                                              \
    }                                                                                                                  \
    [Differentiable]                                                                                                   \
    public T load<I : IInteger>(vector<I, D> idx)                                                                      \
    {                                                                                                                  \
        return read_buffer(_idx(idx, _strides, _offset));                                                              \
    }                                                                                                                  \
    [Differentiable]                                                                                                   \
    public T load<each I : IInteger>(expand each I indices)                                                            \
    {                                                                                                                  \
        return read_buffer(_idx(expand each indices, _strides, _offset));                                              \
    }*/

#define _SLANGPY_DIFFTENSOR_STORE_IMPL                                                                                 \
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

#define _SLANGPY_DIFFTENSOR_LOAD_IMPL                                                                                  \
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

// A read only tensor
public struct RODiffTensor<T : IDifferentiable, let D : int> : IRODiffTensor<T, D> where T.Differential : IAtomicAddable
{
    // Underlying data storage
    private ROTensor<T, D> _primal;
    private AtomicTensor<T.Differential, D> _grad_out;

    _SLANGPY_COMMON_DIFFTENSOR_IMPL
    _SLANGPY_DIFFTENSOR_LOAD_IMPL

    [Differentiable]
    [BackwardDerivative(_load_bwd_array)]
    public T load<I : IInteger>(I idx[D])
    {
        return _primal.load(idx);
    }
    [Differentiable]
    [BackwardDerivative(_load_bwd_vector)]
    public T load<I : IInteger>(vector<I, D> idx)
    {
        return _primal.load(idx);
    }

    //NOTE: Can't get bwds pattern to match - I != I or something!
    /*
    [Differentiable]
    [BackwardDerivative(_load_bwd_expand)]
    public T load<each I : IInteger>(expand each I indices)
    {
        return _primal.load(expand each indices);
    }
    */

    void _load_bwd_array<I : IInteger>(I idx[D], T.Differential grad)
    {
        _grad_out.add(idx, grad);
    }
    void _load_bwd_vector<I : IInteger>(vector<I, D> idx, T.Differential grad)
    {
        _grad_out.add(idx, grad);
    }

    /*
    void _load_bwd_expand<each I : IInteger>(expand each I indices, T.Differential grad)
    {
        _grad_out.add(expand each indices, grad);
    }
    */


    public __subscript<I : IInteger>(I indices[D])->T
    {
        [BackwardDifferentiable] get { return load(indices); }
    }
     public __subscript<I : IInteger>(vector<I, D> indices)->T
    {
        [BackwardDifferentiable] get { return load(indices); }
    }

    /*
    public __subscript<each I : IInteger>(expand each I indices)->T
    {
        get { return load(expand each indices); }
    }
    */

    public void __slangpy_load(Context0D context, out This value)
    {
        value = this;
    }
    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out RODiffTensor<T, SliceD> value)
    {
        _primal.__slangpy_load<SliceD>(ctx, value._primal);
        _grad_out.__slangpy_load<SliceD>(ctx, value._grad_out);
    }

}

// Type alias for a read only tensor.
public typealias DiffTensor<T : IDifferentiable, let D : int> where T.Differential : IAtomicAddable = RODiffTensor<T, D> ;

public struct WODiffTensor<T : IDifferentiable, let D : int> : IWODiffTensor<T, D>
{
    // Underlying data storage
    private RWTensor<T, D> _primal;
    private ROTensor<T.Differential, D> _grad_in;

    _SLANGPY_COMMON_DIFFTENSOR_IMPL
    _SLANGPY_DIFFTENSOR_STORE_IMPL

    [Differentiable]
    [BackwardDerivative(_store_bwd_array)]
    public void store<I : IInteger>(I idx[D], T value)
    {
        _primal.store(idx, value);
    }
    [Differentiable]
    [BackwardDerivative(_store_bwd_vector)]
    public void store<I : IInteger>(vector<I, D> idx, T value)
    {
        _primal.store(idx, value);
    }

    // NOTE: Can't get bwds pattern to match - I != I or something!
    /*
    [Differentiable]
    public void store<each I : IInteger>(expand each I indices, T value)
    {
        _primal.store(expand each indices, value);
    }
    */

    public void _store_bwd_array<I : IInteger>(I idx[D], inout DifferentialPair<T> grad)
    {
        grad = diffPair(grad.p, _grad_in.load(idx));
    }
    public void _store_bwd_vector<I : IInteger>(vector<I, D> idx, inout DifferentialPair<T> grad)
    {
        grad = diffPair(grad.p, _grad_in.load(idx));
    }

    public __subscript<I : IInteger>(I indices[D])->T
    {
        [BackwardDifferentiable] set { return store(indices, newValue); }
    }
     public __subscript<I : IInteger>(vector<I, D> indices)->T
    {
        [BackwardDifferentiable] set { return store(indices, newValue); }
    }

    /*
    public __subscript<each I : IInteger>(expand each I indices)->T
    {
        set { return store(expand each indices, newValue); }
    }
    */

    public void __slangpy_load(Context0D context, out This value)
    {
        value = this;
    }
    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out WODiffTensor<T, SliceD> value)
    {
        _primal.__slangpy_load<SliceD>(ctx, value._primal);
        _grad_in.__slangpy_load<SliceD>(ctx, value._grad_in);
    }
}

public struct RWDiffTensor<T : IDifferentiable, let D : int> : IRWDiffTensor<T, D> where T.Differential : IAtomicAddable
{
    private RWTensor<T, D> _primal;
    private ROTensor<T.Differential, D> _grad_in;
    private AtomicTensor<T.Differential, D> _grad_out;

    _SLANGPY_COMMON_DIFFTENSOR_IMPL
    _SLANGPY_DIFFTENSOR_LOAD_IMPL
    _SLANGPY_DIFFTENSOR_STORE_IMPL

    [Differentiable]
    public T load<I : IInteger>(I idx[D])
    {
        return _primal.load(idx);
    }
    [Differentiable]
    public T load<I : IInteger>(vector<I, D> idx)
    {
        return _primal.load(idx);
    }
    /*
    [Differentiable]
    public T load<each I : IInteger>(expand each I indices)
    {
        return _primal.load(expand each indices);
    }
    */
    [Differentiable]
    public void store<I : IInteger>(I idx[D], T value)
    {
        _primal.store(idx, value);
    }
    [Differentiable]
    public void store<I : IInteger>(vector<I, D> idx, T value)
    {
        _primal.store(idx, value);
    }
    /*
    [Differentiable]
    public void store<each I : IInteger>(expand each I indices, T value)
    {
        _primal.store(expand each indices, value);
    }
    */

    public __subscript<I : IInteger>(I indices[D])->T
    {
        get { return load(indices); }
        set { return store(indices, newValue); }
    }
     public __subscript<I : IInteger>(vector<I, D> indices)->T
    {
        get { return load(indices); }
        set { return store(indices, newValue); }
    }

    /*
    public __subscript<each I : IInteger>(expand each I indices)->T
    {
        get { return load(expand each indices); }
        set { return store(expand each indices, newValue); }
    }
    */

    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out RWDiffTensor<T, SliceD> value)
    {
        _primal.__slangpy_load<SliceD>(ctx, value._primal);
        _grad_in.__slangpy_load<SliceD>(ctx, value._grad_in);
        _grad_out.__slangpy_load<SliceD>(ctx, value._grad_out);
    }
}

#endif
