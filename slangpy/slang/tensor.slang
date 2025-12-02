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

// Internal base interface for tensor
public interface IBaseTensor<T, let D : int>
{
    public property shape : uint[D];
}

// Internal base interface for tensor
public interface IBaseTensor<T, let D : int>
{
    public property shape : uint[D];
}

// Interface for a readable tensor.
// This can be used as an argument to a function or as a constraint to
// indicate any type of read-only tensor.
public interface IROTensor<T, let D : int> : IBaseTensor<T, D>
{
    public T read_buffer(int idx);
    public T load<I : IInteger>(I idx[D]);
    public T load<I : IInteger>(vector<I, D> idx);
    public T load<each I : IInteger>(expand each I indices);
}

// Type alias for a read only tensor.
public typealias ITensor<T, let D : int> = IROTensor<T, D>;

// Interface for a writable tensor.
// This can be used as an argument to a function or as a constraint to
// indicate any type of write-only tensor.
public interface IWOTensor<T, let D : int> : IBaseTensor<T, D>
{
    public void write_buffer(int idx, T value);
    public void store<I : IInteger>(I idx[D], T value);
    public void store<I : IInteger>(vector<I, D> idx, T value);
    public void store<each I : IInteger>(expand each I indices, T value);
}

// Interface for a read-write tensor.
// This can be used as an argument to a function or as a constraint to
// indicate any type of read-write tensor.
public interface IRWTensor<T, let D : int> : IROTensor<T, D>, IWOTensor<T, D>
{
}

#define _SLANGPY_TENSOR_COMMON_FIELDS \
    private uint[D] _shape;\
    private uint[D] _strides;\
    private uint _offset;\
    public property shape : uint[D] { get { return _shape; } }


#define _SLANGPY_TENSOR_STORE \
    public void store<I : IInteger>(I idx[D], T value)\
    {\
        write_buffer(_idx(idx, _strides, _offset), value);\
    }\
    public void store<I : IInteger>(vector<I, D> idx, T value)\
    {\
        write_buffer(_idx(idx, _strides, _offset), value);\
    }\
    public void store<each I : IInteger>(expand each I indices, T value)\
    {\
        write_buffer(_idx(expand each indices, _strides, _offset), value);\
    }
    __generic<I : IInteger>
    __subscript(ISizedArray<I, D> indices)->T
    {
        [BackwardDifferentiable] get { return get(impl::makeIndex<D, I>(indices)); }
    }

#define _SLANGPY_TENSOR_LOAD \
    public T load<I : IInteger>(I idx[D])\
    {\
        return read_buffer(_idx(idx, _strides, _offset));\
    }\
    public T load<I : IInteger>(vector<I, D> idx)\
    {\
        return read_buffer(_idx(idx, _strides, _offset));\
    }\
    public T load<each I : IInteger>(expand each I indices)\
    {\
        return read_buffer(_idx(expand each indices, _strides, _offset));\
    }

#define _SLANGPY_TENSOR_SLANGPYSTORE \
    public void __slangpy_store(ContextND<D> context, in T value)\
    {\
        this.store(context.call_id, value);\
    }\
    public void __slangpy_store<let N : int>(ContextND<D - 1> context, in T[N] value)\
    {\
        int idx[D];\
        [ForceUnroll]\
        for (int i = 0; i < D - 1; ++i)\
            idx[i] = context.call_id[i];\
        [ForceUnroll]\
        for (int i = 0; i < N; i++) {\
            idx[D - 1] = i;\
            this.store(idx, value[i]);\
        }\
    }\
    public void __slangpy_store<let N : int>(ContextND<D - 1> context, in vector<T, N> value)\
    {\
        int idx[D];\
        [ForceUnroll]\
        for (int i = 0; i < D - 1; ++i)\
            idx[i] = context.call_id[i];\
        [ForceUnroll]\
        for (int i = 0; i < N; i++) {\
            idx[D - 1] = i;\
            this.store(idx, value[i]);\
        }\
    }\
    public void __slangpy_store<let M : int, let N : int>(ContextND<D - 2> context, in T value[M][N])\
    {\
        int idx[D];\
        [ForceUnroll]\
        for (int i = 0; i < D - 2; ++i)\
            idx[i] = context.call_id[i];\
        [ForceUnroll]\
        for (int i = 0; i < M; i++) {\
            [ForceUnroll]\
            for (int j = 0; j < N; j++) {\
                idx[D - 2] = i;\
                idx[D - 1] = j;\
                this.store(idx, value[i][j]);\
            }\
        }\
    }\
    public void __slangpy_store<let M : int, let N : int>(ContextND<D - 2> context, in matrix<T, M, N> value)\
    {\
        int idx[D];\
        [ForceUnroll]\
        for (int i = 0; i < D - 2; ++i)\
            idx[i] = context.call_id[i];\
        [ForceUnroll]\
        for (int i = 0; i < M; i++) {\
            [ForceUnroll]\
            for (int j = 0; j < N; j++) {\
                idx[D - 2] = i;\
                idx[D - 1] = j;\
                this.store(idx, value[i][j]);\
            }\
        }\
    }

#define _SLANGPY_TENSOR_SLANGPYLOAD \
        public void __slangpy_load(ContextND<D> context, out T value)\
        {\
            value = this.load(context.call_id);\
        }\
        public void __slangpy_load<let N : int>(ContextND<D - 1> ctx, out T[N] value)\
        {\
            int idx[D];\
            [ForceUnroll]\
            for (int i = 0; i < D - 1; ++i)\
                idx[i] = ctx.call_id[i];\
            [ForceUnroll]\
            for (int i = 0; i < N; i++) {\
                idx[D - 1] = i;\
                value[i] = this.load(idx);\
            }\
        }\
        public void __slangpy_load<let N : int>(ContextND<D - 1> ctx, out vector<T, N> value)\
        {\
            int idx[D];\
            [ForceUnroll]\
            for (int i = 0; i < D - 1; ++i)\
                idx[i] = ctx.call_id[i];\
            [ForceUnroll]\
            for (int i = 0; i < N; i++) {\
                idx[D - 1] = i;\
                value[i] = this.load(idx);\
            }\
        }\
        public void __slangpy_load<let M : int, let N : int>(ContextND<D - 2> context, out T value[M][N])\
        {\
            int idx[D];\
            [ForceUnroll]\
            for (int i = 0; i < D - 2; ++i)\
                idx[i] = context.call_id[i];\
            [ForceUnroll]\
            for (int i = 0; i < M; i++) {\
                [ForceUnroll]\
                for (int j = 0; j < N; j++) {\
                    idx[D - 2] = i;\
                    idx[D - 1] = j;\
                    value[i][j] = this.load(idx);\
                }\
            }\
        }\
        public void __slangpy_load<let M : int, let N : int>(ContextND<D - 2> context, out matrix<T, M, N> value)\
        {\
            int idx[D];\
            [ForceUnroll]\
            for (int i = 0; i < D - 2; ++i)\
                idx[i] = context.call_id[i];\
            [ForceUnroll]\
            for (int i = 0; i < M; i++) {\
                [ForceUnroll]\
                for (int j = 0; j < N; j++) {\
                    idx[D - 2] = i;\
                    idx[D - 1] = j;\
                    value[i][j] = this.load(idx);\
                }\
            }\
        }
    }

// Read only tensor
public struct ROTensor<T, let D : int> : IROTensor<T, D>
{
    // Underlying data storage
    private StorageTraits<T>::BufferType _data;

    public T read_buffer(int idx)
    {
        return _data[idx];
    }

    _SLANGPY_TENSOR_COMMON_FIELDS
    _SLANGPY_TENSOR_LOAD
    _SLANGPY_TENSOR_SLANGPYLOAD

    public __subscript<I : IInteger>(I indices[D])->T
    {
        get { return load(indices); }
    }
     public __subscript<I : IInteger>(vector<I, D> indices)->T
    {
        get { return load(indices); }
    }
     public __subscript<each I : IInteger>(expand each I indices)->T
    {
        get { return load(expand each indices); }
    }

    public void __slangpy_load(Context0D context, out This value)
    {
        value = this;
    }
    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out ROTensor<T, SliceD> value)
    {
        value._data = _data;
        _slice(ctx.call_id, _strides, _offset, value._strides, value._offset);
        [ForceUnroll]
        for (int i = 0; i < SliceD; ++i)
            value._shape[i] = _shape[D - SliceD + i];
    }

}

// Type alias for a read only tensor.
public typealias Tensor<T, let D : int> = ROTensor<T, D>;

// Write only tensor
public struct WOTensor<T, let D : int> : IWOTensor<T, D>
{
    // Underlying data storage
    private StorageTraits<T>::RWBufferType _data;

    public void write_buffer(int idx, T value)
    {
        _data[idx] = value;
    }

    _SLANGPY_TENSOR_COMMON_FIELDS
    _SLANGPY_TENSOR_STORE
    _SLANGPY_TENSOR_SLANGPYSTORE

    public __subscript<I : IInteger>(I indices[D])->T
    {
        set { return store(indices, newValue); }
    }
     public __subscript<I : IInteger>(vector<I, D> indices)->T
    {
        set { return store(indices, newValue); }
    }
     public __subscript<each I : IInteger>(expand each I indices)->T
    {
        set { return store(expand each indices, newValue); }
    }

    public void __slangpy_load(Context0D context, out This value)
    {
        value = this;
    }
    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out WOTensor<T, SliceD> value)
    {
        value._data = _data;
        _slice(ctx.call_id, _strides, _offset, value._strides, value._offset);
        [ForceUnroll]
        for (int i = 0; i < SliceD; ++i)
            value._shape[i] = _shape[D - SliceD + i];
    }
}

// Read-write tensor
public struct RWTensor<T, let D : int> : IRWTensor<T, D>
{
    // Underlying data storage
    private StorageTraits<T>::RWBufferType _data;

    public T read_buffer(int idx)
    {
        return _data[idx];
    }
    public void write_buffer(int idx, T value)
    {
        _data[idx] = value;
    }

    _SLANGPY_TENSOR_COMMON_FIELDS
    _SLANGPY_TENSOR_LOAD
    _SLANGPY_TENSOR_SLANGPYLOAD
    _SLANGPY_TENSOR_STORE
    _SLANGPY_TENSOR_SLANGPYSTORE

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
     public __subscript<each I : IInteger>(expand each I indices)->T
    {
        get { return load(expand each indices); }
        set { return store(expand each indices, newValue); }
    }

    public void __slangpy_load(Context0D context, out This value)
    {
        value = this;
    }
    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out RWTensor<T, SliceD> value)
    {
        value._data = _data;
        _slice(ctx.call_id, _strides, _offset, value._strides, value._offset);
        [ForceUnroll]
        for (int i = 0; i < SliceD; ++i)
            value._shape[i] = _shape[D - SliceD + i];
    }
}

// Atomic read-write tensor, that also supports atomic add operations
// that are typically used for accumulating gradients.
public struct AtomicTensor<T: IAtomicAddable, let D : int> : IRWTensor<T, D>
{
    // Underlying data storage
    private StorageTraits<T>::AtomicBufferType _data;

    public T read_buffer(int idx)
    {
        return StorageTraits<T>::atomicLoad(_data, idx);
    }
    public void write_buffer(int idx, T value)
    {
        StorageTraits<T>::atomicStore(_data, idx, value);
    }

    _SLANGPY_TENSOR_COMMON_FIELDS
    _SLANGPY_TENSOR_LOAD
    _SLANGPY_TENSOR_SLANGPYLOAD
    _SLANGPY_TENSOR_STORE
    _SLANGPY_TENSOR_SLANGPYSTORE


    public void add<I : IInteger>(I idx[D], T value)
    {
        StorageTraits<T>::atomicAdd(_data, _idx(idx, _strides, _offset), value);
    }
    public void add<I : IInteger>(vector<I, D> idx, T value)
    {
        StorageTraits<T>::atomicAdd(_data, _idx(idx, _strides, _offset), value);
    }
    public void add<each I : IInteger>(expand each I indices, T value)
    {
        StorageTraits<T>::atomicAdd(_data, _idx(expand each indices, _strides, _offset), value);
    }

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
    public __subscript<each I : IInteger>(expand each I indices)->T
    {
        get { return load(expand each indices); }
        set { return store(expand each indices, newValue); }
    }

    public void __slangpy_load(Context0D context, out This value)
    {
        value = this;
    }
    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out AtomicTensor<T, SliceD> value)
    {
        value._data = _data;
        _slice(ctx.call_id, _strides, _offset, value._strides, value._offset);
        [ForceUnroll]
        for (int i = 0; i < SliceD; ++i)
            value._shape[i] = _shape[D - SliceD + i];
    }
}


#endif
