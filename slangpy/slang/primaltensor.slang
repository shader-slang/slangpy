// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

implementing slangpy;

#define _SLANGPY_PRIMALTENSOR_COMMON_FIELDS \
    public uint[D] _shape;\
    public uint[D] _strides;\
    public uint _offset;\
    public property shape : uint[D] { get { return _shape; } }


#define _SLANGPY_PRIMALTENSOR_STORE                                                                                    \
    [TreatAsDifferentiable]                                                                                            \
    public void store<I : __BuiltinIntegerType>(I idx[D], T value)                                                     \
    {                                                                                                                  \
        write_buffer(_idx(idx, _strides, _offset), value);                                                             \
    }                                                                                                                  \
    [TreatAsDifferentiable]                                                                                            \
    public void store<I : __BuiltinIntegerType>(vector<I, D> idx, T value)                                             \
    {                                                                                                                  \
        write_buffer(_idx(idx, _strides, _offset), value);                                                             \
    }                                                                                                                  \
    void _write_primal_each<each I : __BuiltinIntegerType>(T value, expand each I idx)                          \
    {                                                                                                                  \
        int lin_idx = _idx_each(expand each idx, _strides, _offset);                                                   \
        write_buffer(lin_idx, value);                                                                                  \
    }                                                                                                                  \
    T.Differential _read_grad_each<each I : __BuiltinIntegerType>(expand each I idx)                            \
    {                                                                                                                  \
    }


#define _SLANGPY_PRIMALTENSOR_LOAD                                                                                     \
    [TreatAsDifferentiable]                                                                                            \
    public T load<I : __BuiltinIntegerType>(I idx[D])                                                                  \
    {                                                                                                                  \
        return read_buffer(_idx(idx, _strides, _offset));                                                              \
    }                                                                                                                  \
    [TreatAsDifferentiable]                                                                                            \
    public T load<I : __BuiltinIntegerType>(vector<I, D> idx)                                                          \
    {                                                                                                                  \
        return read_buffer(_idx(idx, _strides, _offset));                                                              \
    }                                                                                                                  \
    T _read_primal_each<each I : __BuiltinIntegerType>(expand each I idx)                                       \
    {                                                                                                                  \
        int lin_idx = _idx_each(expand each idx, _strides, _offset);                                                   \
        return read_buffer(lin_idx);                                                                                   \
    }                                                                                                                  \
    void _write_grad_each<each I : __BuiltinIntegerType>(T.Differential grad, expand each I idx)                \
    {                                                                                                                  \
    }                                                                                                                  \
    void _accumulate_grad_each<each I : __BuiltinIntegerType>(T.Differential grad, expand each I idx)           \
    {                                                                                                                  \
    }

#define _SLANGPY_PRIMALTENSOR_SLANGPYSTORE                                                                             \
    public void __slangpy_store(ContextND<D> context, in T value)                                                      \
    {                                                                                                                  \
        this.store(context.call_id, value);                                                                            \
    }                                                                                                                  \
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

#define _SLANGPY_PRIMALTENSOR_SLANGPYLOAD                                                                              \
    public void __slangpy_load(ContextND<D> context, out T value)                                                      \
    {                                                                                                                  \
        value = this.load(context.call_id);                                                                            \
    }                                                                                                                  \
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



// A read only tensor that is compatible with the IDiffTensor interface but doesn't store
// any gradients. This is typically used in generated slangpy code to pass tensors to functions
// that expect an IDiffTensor type during the forward pass, when gradients are not needed.
public struct PrimalTensor<T: IDifferentiable, let D : int> : IDiffTensor<T, D> where T.Differential : IAtomicAddable
{
    // Underlying data storage
    public StorageTraits<T>::BufferType _data;

    public T read_buffer(int idx)
    {
        return _data[idx];
    }

    _SLANGPY_PRIMALTENSOR_COMMON_FIELDS
    _SLANGPY_PRIMALTENSOR_LOAD
    _SLANGPY_PRIMALTENSOR_SLANGPYLOAD

    public __subscript<I : __BuiltinIntegerType>(I indices[D])->T
    {
        get { return load(indices); }
    }
    public __subscript<I : __BuiltinIntegerType>(vector<I, D> indices)->T
    {
        get { return load(indices); }
    }

    public void __slangpy_load(Context0D context, out This value)
    {
        value = this;
    }

    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out PrimalTensor<T, SliceD> value)
    {
        value._data = _data;
        _slice(ctx.call_id, _strides, _offset, value._strides, value._offset);
        [ForceUnroll]
        for (int i = 0; i < SliceD; ++i)
            value._shape[i] = _shape[D - SliceD + i];
    }
}

// Write only tensor that is compatible with the IWDiffTensor interface but doesn't store
// any gradients. This is typically used in generated slangpy code to pass tensors to functions
// that expect an IWDiffTensor type during the forward pass, when gradients are not needed.
public struct WPrimalTensor<T : IDifferentiable, let D : int> : IWDiffTensor<T, D> where T.Differential : IAtomicAddable
{
    // Underlying data storage
    public StorageTraits<T>::RWBufferType _data;

    public void write_buffer(int idx, T value)
    {
        _data[idx] = value;
    }

    _SLANGPY_PRIMALTENSOR_COMMON_FIELDS
    _SLANGPY_PRIMALTENSOR_STORE
    _SLANGPY_PRIMALTENSOR_SLANGPYSTORE

    public __subscript<I : __BuiltinIntegerType>(I indices[D])->T
    {
        set { return store(indices, newValue); }
    }
    public __subscript<I : __BuiltinIntegerType>(vector<I, D> indices)->T
    {
        set { return store(indices, newValue); }
    }

    public void __slangpy_load(Context0D context, out This value)
    {
        value = this;
    }
    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out WPrimalTensor<T, SliceD> value)
    {
        value._data = _data;
        _slice(ctx.call_id, _strides, _offset, value._strides, value._offset);
        [ForceUnroll]
        for (int i = 0; i < SliceD; ++i)
            value._shape[i] = _shape[D - SliceD + i];
    }
}

// Read-write tensor that is compatible with the IRWDiffTensor interface but doesn't store
// any gradients. This is typically used in generated slangpy code to pass tensors to functions
// that expect an IRWDiffTensor type during the forward pass, when gradients are not needed.
public struct RWPrimalTensor<T : IDifferentiable, let D : int> : IRWDiffTensor<T, D> where T.Differential : IAtomicAddable
{
    // Underlying data storage
    public StorageTraits<T>::RWBufferType _data;

    public T read_buffer(int idx)
    {
        return _data[idx];
    }
    public void write_buffer(int idx, T value)
    {
        _data[idx] = value;
    }

    _SLANGPY_PRIMALTENSOR_COMMON_FIELDS
    _SLANGPY_PRIMALTENSOR_LOAD
    _SLANGPY_PRIMALTENSOR_SLANGPYLOAD
    _SLANGPY_PRIMALTENSOR_STORE
    _SLANGPY_PRIMALTENSOR_SLANGPYSTORE

    public __subscript<I : __BuiltinIntegerType>(I indices[D])->T
    {
        get { return load(indices); }
        set { return store(indices, newValue); }
    }
    public __subscript<I : __BuiltinIntegerType>(vector<I, D> indices)->T
    {
        get { return load(indices); }
        set { return store(indices, newValue); }
    }

    public void __slangpy_load(Context0D context, out This value)
    {
        value = this;
    }
    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out RWPrimalTensor<T, SliceD> value)
    {
        value._data = _data;
        _slice(ctx.call_id, _strides, _offset, value._strides, value._offset);
        [ForceUnroll]
        for (int i = 0; i < SliceD; ++i)
            value._shape[i] = _shape[D - SliceD + i];
    }
}

// Bespoke loads for directly reading underlying structured buffers from tensor
#ifndef __TARGET_CUDA__
public extension<T : IDifferentiable, let D : int> PrimalTensor<T, D> where T.Differential : IAtomicAddable
{
    public void __slangpy_load(Context0D context, out StructuredBuffer<T> value)
    {
        value = this._data;
    }
}
public extension<T : IDifferentiable, let D : int> WPrimalTensor<T, D> where T.Differential : IAtomicAddable
{
    public void __slangpy_load(Context0D context, out RWStructuredBuffer<T> value)
    {
        value = this._data;
    }
}
public extension<T : IDifferentiable, let D : int> RWPrimalTensor<T, D> where T.Differential : IAtomicAddable
{
    public void __slangpy_load(Context0D context, out RWStructuredBuffer<T> value)
    {
        value = this._data;
    }
}
#endif
