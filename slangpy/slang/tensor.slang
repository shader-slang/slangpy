// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

implementing slangpy;

// Internal base interface for tensor
public interface IBaseTensor<T, let D : int>
{
    public property shape : uint[D];
}

// Interface for a readable tensor.
// This can be used as an argument to a function or as a constraint to
// indicate any type of read-only tensor.
public interface ITensor<T, let D : int> : IBaseTensor<T, D>
{
    T _read_each<each I : __BuiltinIntegerType>(expand each I idx);

    public T read_buffer(int idx);
    public T load<I : __BuiltinIntegerType>(I idx[D]);
    public T load<I : __BuiltinIntegerType>(vector<I, D> idx);
    public __subscript<I : __BuiltinIntegerType>(I indices[D])->T
    {
        get;
    }
    public __subscript<I : __BuiltinIntegerType>(vector<I, D> indices)->T
    {
        get;
    }
}

// Interface for a writable tensor.
// This can be used as an argument to a function or as a constraint to
// indicate any type of write-only tensor.
public interface IWTensor<T, let D : int> : IBaseTensor<T, D>
{
    void _write_each<each I : __BuiltinIntegerType>(T value, expand each I idx);

    public void write_buffer(int idx, T value);
    public void store<I : __BuiltinIntegerType>(I idx[D], T value);
    public void store<I : __BuiltinIntegerType>(vector<I, D> idx, T value);
    public __subscript<I : __BuiltinIntegerType>(I indices[D])->T
    {
        set;
    }
    public __subscript<I : __BuiltinIntegerType>(vector<I, D> indices)->T
    {
        set;
    }
}

// Interface for a read-write tensor.
// This can be used as an argument to a function or as a constraint to
// indicate any type of read-write tensor.
public interface IRWTensor<T, let D : int> : IBaseTensor<T, D>
{
    T _read_each<each I : __BuiltinIntegerType>(expand each I idx);
    void _write_each<each I : __BuiltinIntegerType>(T value, expand each I idx);

    public T read_buffer(int idx);
    public T load<I : __BuiltinIntegerType>(I idx[D]);
    public T load<I : __BuiltinIntegerType>(vector<I, D> idx);
    public void write_buffer(int idx, T value);
    public void store<I : __BuiltinIntegerType>(I idx[D], T value);
    public void store<I : __BuiltinIntegerType>(vector<I, D> idx, T value);

    public __subscript<I : __BuiltinIntegerType>(I indices[D])->T
    {
        get;
        set;
    }
    public __subscript<I : __BuiltinIntegerType>(vector<I, D> indices)->T
    {
        get;
        set;
    }
}

// Extend ITensor types to add their element type
public extension <T, let D: int, TensorType: ITensor<T, D>> TensorType
{
    typedef T ElementType;
}
public extension<T, let D : int, TensorType : IWTensor<T, D>> TensorType
{
    typedef T ElementType;
}
public extension<T, let D : int, TensorType : IRWTensor<T, D>> TensorType
{
    typedef T ElementType;
}

#define _SLANGPY_TENSOR_COMMON_FIELDS \
    public uint[D] _shape;\
    public uint[D] _strides;\
    public uint _offset;\
    public property shape : uint[D] { get { return _shape; } }


#define _SLANGPY_TENSOR_STORE                                                                                          \
    [ForceInline] public void store<I : __BuiltinIntegerType>(I idx[D], T value)                                       \
    {                                                                                                                  \
        write_buffer(_idx(idx, _strides, _offset), value);                                                             \
    }                                                                                                                  \
    [ForceInline] public void store<I : __BuiltinIntegerType>(vector<I, D> idx, T value)                               \
    {                                                                                                                  \
        write_buffer(_idx(idx, _strides, _offset), value);                                                             \
    }                                                                                                                  \
    [ForceInline] void _write_each<each I : __BuiltinIntegerType>(T value, expand each I idx)                          \
    {                                                                                                                  \
        int lin_idx = _idx_each(expand each idx, _strides, _offset);                                                   \
        write_buffer(lin_idx, value);                                                                                  \
    }

#define _SLANGPY_TENSOR_LOAD                                                                                           \
    [ForceInline] public T load<I : __BuiltinIntegerType>(I idx[D])                                                    \
    {                                                                                                                  \
        return read_buffer(_idx(idx, _strides, _offset));                                                              \
    }                                                                                                                  \
    [ForceInline] public T load<I : __BuiltinIntegerType>(vector<I, D> idx)                                            \
    {                                                                                                                  \
        return read_buffer(_idx(idx, _strides, _offset));                                                              \
    }                                                                                                                  \
    [ForceInline] T _read_each<each I : __BuiltinIntegerType>(expand each I idx)                                       \
    {                                                                                                                  \
        int lin_idx = _idx_each(expand each idx, _strides, _offset);                                                   \
        return read_buffer(lin_idx);                                                                                   \
    }

#define _SLANGPY_TENSOR_SLANGPYSTORE                                                                                   \
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

#define _SLANGPY_TENSOR_SLANGPYLOAD                                                                                    \
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

// Read only tensor
public struct Tensor<T, let D : int> : ITensor<T, D>
{
    // Underlying data storage
    public StorageTraits<T>::BufferType _data;

    public T read_buffer(int idx)
    {
        return _data[idx];
    }

    _SLANGPY_TENSOR_COMMON_FIELDS
    _SLANGPY_TENSOR_LOAD
    _SLANGPY_TENSOR_SLANGPYLOAD

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
    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out Tensor<T, SliceD> value)
    {
        value._data = _data;
        _slice(ctx.call_id, _strides, _offset, value._strides, value._offset);
        [ForceUnroll]
        for (int i = 0; i < SliceD; ++i)
            value._shape[i] = _shape[D - SliceD + i];
    }
}

// Write only tensor
public struct WTensor<T, let D : int> : IWTensor<T, D>
{
    // Underlying data storage
    public StorageTraits<T>::RWBufferType _data;

    public void write_buffer(int idx, T value)
    {
        _data[idx] = value;
    }

    _SLANGPY_TENSOR_COMMON_FIELDS
    _SLANGPY_TENSOR_STORE
    _SLANGPY_TENSOR_SLANGPYSTORE

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
    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out WTensor<T, SliceD> value)
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
    public StorageTraits<T>::RWBufferType _data;

    public T read_buffer(int idx)
    {
        return _data[idx];
    }
    public void write_buffer(int idx, T value)
    {
        _data[idx] = value;
    }

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

    _SLANGPY_TENSOR_COMMON_FIELDS
    _SLANGPY_TENSOR_LOAD
    _SLANGPY_TENSOR_SLANGPYLOAD
    _SLANGPY_TENSOR_STORE
    _SLANGPY_TENSOR_SLANGPYSTORE

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
    public StorageTraits<T>::AtomicBufferType _data;

#ifdef __TARGET_METAL__
    // Element byte stride - passed from C++ on Metal where sizeof(T) differs from buffer stride.
    // For example, sizeof(float3)=12 but Metal buffer stride=16.
    public uint _element_byte_stride;
#else
    // On non-Metal platforms, sizeof(T) matches buffer stride, so use static const.
    static const uint _element_byte_stride = sizeof(T);
#endif

    public T read_buffer(int idx)
    {
        return StorageTraits<T>::atomicLoadWithStride(_data, idx, _element_byte_stride);
    }
    public void write_buffer(int idx, T value)
    {
        StorageTraits<T>::atomicStoreWithStride(_data, idx, _element_byte_stride, value);
    }
    public void accumulate_buffer(int idx, T value)
    {
        StorageTraits<T>::atomicAddWithStride(_data, idx, _element_byte_stride, value);
    }
    [ForceInline] void _accumulate_each<each I : __BuiltinIntegerType>(T value, expand each I idx)
    {
        int lin_idx = _idx_each(expand each idx, _strides, _offset);
        accumulate_buffer(lin_idx, value);
    }

    _SLANGPY_TENSOR_COMMON_FIELDS
    _SLANGPY_TENSOR_LOAD
    _SLANGPY_TENSOR_SLANGPYLOAD
    _SLANGPY_TENSOR_STORE
    _SLANGPY_TENSOR_SLANGPYSTORE


    [ForceInline] public void add<I : __BuiltinIntegerType>(I idx[D], T value)
    {
        StorageTraits<T>::atomicAddWithStride(_data, _idx(idx, _strides, _offset), _element_byte_stride, value);
    }
    [ForceInline] public void add<I : __BuiltinIntegerType>(vector<I, D> idx, T value)
    {
        StorageTraits<T>::atomicAddWithStride(_data, _idx(idx, _strides, _offset), _element_byte_stride, value);
    }

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
    public void __slangpy_load<let SliceD : int>(ContextND<D - SliceD> ctx, out AtomicTensor<T, SliceD> value)
    {
        value._data = _data;
#ifdef __TARGET_METAL__
        value._element_byte_stride = _element_byte_stride;
#endif
        _slice(ctx.call_id, _strides, _offset, value._strides, value._offset);
        [ForceUnroll]
        for (int i = 0; i < SliceD; ++i)
            value._shape[i] = _shape[D - SliceD + i];
    }
}


// Bespoke load/store for accessing pointers from a uint64 tensor
public extension<let D : int, TensorType : ITensor<uint64_t, D>> TensorType
{
    public void __slangpy_load<PtrT>(ContextND<D> context, out Ptr<PtrT> value)
    {
        uint64_t rawPtr = this.load(context.call_id);
        value = (Ptr<PtrT>)rawPtr;
    }
    public void __slangpy_load<PtrT>(ContextND<D> context, out ImmutablePtr<PtrT> value)
    {
        uint64_t rawPtr = this.load(context.call_id);
        value = (ImmutablePtr<PtrT>)rawPtr;
    }
}
public extension<let D : int, TensorType : IWTensor<uint64_t, D>> TensorType
{
    public void __slangpy_store<PtrT>(ContextND<D> context, Ptr<PtrT> value)
    {
        uint64_t rawPtr = (uint64_t)value;
        this.store(context.call_id, rawPtr);
    }
    public void __slangpy_store<PtrT>(ContextND<D> context, ImmutablePtr<PtrT> value)
    {
        uint64_t rawPtr = (uint64_t)value;
        this.store(context.call_id, rawPtr);
    }
}

// Bespoke loads for directly reading underlying structured buffers from tensor
#ifndef __TARGET_CUDA__
public extension<T, let D: int> Tensor<T, D>
{
    public void __slangpy_load(Context0D context, out StructuredBuffer<T> value)
    {
        value = this._data;
    }
}
public extension<T, let D : int> WTensor<T, D>
{
    public void __slangpy_load(Context0D context, out RWStructuredBuffer<T> value)
    {
        value = this._data;
    }
}
public extension<T, let D : int> RWTensor<T, D>
{
    public void __slangpy_load(Context0D context, out RWStructuredBuffer<T> value)
    {
        value = this._data;
    }
}
#endif

void _get_sample_indices(float2 uv, int2 shape, out int2 idx0, out int2 idx1, out float2 interp)
{
    int2 maxidx = shape - 1;
    int2 texel = int2(uv * shape);
    interp = frac(uv * shape);

    idx0 = texel;
    idx1 = min(texel + int2(1, 1), maxidx);
}

T _bilerp<T: __BuiltinFloatingPointType>(T c00, T c10, T c01, T c11, float2 interp)
{
    return lerp(lerp(c00, c10, (T)interp.x), lerp(c01, c11, (T)interp.x), (T)interp.y);
}
vector<T,N> _bilerp<T: __BuiltinFloatingPointType, let N : int>(vector<T, N> c00, vector<T, N> c10, vector<T, N> c01, vector<T, N> c11, float2 interp)
{
    return lerp(lerp(c00, c10, (T)interp.x), lerp(c01, c11, (T)interp.x), (T)interp.y);
}

#define _SLANGPY_TENSOR_SAMPLE                                                                                         \
    public ElementType sample(float2 uv)                                                                               \
    {                                                                                                                  \
        int2 shape = int2(shape[1], shape[0]);                                                                         \
        int2 idx0, idx1;                                                                                               \
        float2 frac;                                                                                                   \
        _get_sample_indices(uv, shape, idx0, idx1, frac);                                                              \
        ElementType c00 = load(idx0);                                                                                  \
        ElementType c10 = load(int2(idx1.x, idx0.y));                                                                  \
        ElementType c01 = load(int2(idx0.x, idx1.y));                                                                  \
        ElementType c11 = load(idx1);                                                                                  \
        return _bilerp(c00, c10, c01, c11, frac);                                                                      \
    }

public extension<T : __BuiltinFloatingPointType, TensorType: ITensor<T,2>> TensorType
{
    _SLANGPY_TENSOR_SAMPLE
}

public extension<T : __BuiltinFloatingPointType, let N : int, TensorType: ITensor<vector<T,N>, 2>> TensorType
{
    _SLANGPY_TENSOR_SAMPLE
}

public extension<T : __BuiltinFloatingPointType, TensorType : IRWTensor<T, 2>> TensorType
{
    _SLANGPY_TENSOR_SAMPLE
}

public extension<T : __BuiltinFloatingPointType, let N : int, TensorType : IRWTensor<vector<T, N>, 2>> TensorType
{
    _SLANGPY_TENSOR_SAMPLE
}
