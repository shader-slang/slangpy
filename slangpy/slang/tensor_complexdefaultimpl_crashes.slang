// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

implementing slangpy;

/*
This adjusts increases dependency on default implemenations and properties to
move common code into the interfaces rather than duplicating it in each struct.
However it raises the question of how well slang deals with returning values
from properties, as well as using the more complex default implementations.

NOTE: This version crashes on complication.
*/

#if 0

int _idx<I: IInteger, let N : int>(I[N] indices, uint stride[N], uint offset)
{
    int idx = 0;
    [ForceUnroll]
    for (int i = 0; i < N; i++) { idx += indices[i].toUInt() * stride[i]; }
    return idx;
}
int _idx<I : IInteger, let N : int>(vector<I, N> indices, uint stride[N], uint offset)
{
    int idx = 0;
    int end = N - 1;
    [ForceUnroll]
    for (int i = 0; i < N; i++) { idx += indices[end - i].toUInt() * stride[i]; }
    return idx;
}
int _idx<each I : IInteger, let N : int>(expand each I indices, uint stride[N], uint offset)
{
    // NOTE: Can probably optimize this to avoid the intermediate array allocation
    uint[N] idxVec;
    int i = 0;
    expand idxVec[i++] = (each indices).toUInt();
    return _idx(idxVec, stride, offset);
}

// Now have a base tensor required for properties common to everything
public interface IBaseTensor<T, let D : int>
{
    public property shape : uint[D];
    public property strides : uint[D];
    public property offset : uint;

    /*__slangpy_loaders*/
    public void __slangpy_load(Context0D context, out This value)
    {
        value = this;
    }
}

// Public interface for a read only tensor. This can be used as the argument
// to a function (or a generic constraint) to indicate it should take any
// type of read-only tensor.
public interface IROTensor<T, let D : int> : IBaseTensor<T, D>
{
    public T read_buffer(int idx);

    public T load<I : IInteger>(I idx[D])
    {
        return this.read_buffer(_idx(idx, strides, offset));
    }
    public T load<I : IInteger>(vector<I, D> idx)
    {
        return this.read_buffer(_idx(idx, strides, offset));
    }
    public T load<each I : IInteger>(expand each I indices)
    {
        return this.read_buffer(_idx(expand each indices, strides, offset));
    }

    // NOTE: Can't do this IF I want IRWTensor to inherit from IROTensor and IWOTensor
    // public __subscript<I : IInteger>(I indices[D])->T
    //{
    //    get;
    //}
    //public __subscript<I : IInteger>(vector<I, D> indices)->T
    //{
    //    get;
    //}
    //public __subscript<each I : IInteger>(expand each I indices)->T
    //{
    //    get;
    //}

    /*__slangpy_loaders*/
    public void __slangpy_load(ContextND<D> context, out T value)
    {
        value = this.load(context.call_id);
    }
    //public void __slangpy_load<let N : int>(ContextND<D - 1> ctx, out T[N] value);
    //public void __slangpy_load<let N : int>(ContextND<D - 1> ctx, out vector<T, N> value);
    //public void __slangpy_load<let M : int, let N : int>(ContextND<D - 2> context, out T value[M][N]);
    //public void __slangpy_load<let M : int, let N : int>(ContextND<D - 2> context, out matrix<T, M, N> value);
}

// Type alias for a read only tensor interface This can be used as the argument
// to a function (or a generic constraint) to indicate it should take any
// type of read-only tensor.
public typealias ITensor<T, let D : int> = IROTensor<T, D>;

// Public interface for a write only tensor. This can be used as the argument
// to a function (or a generic constraint) to indicate it should take any
// type of write-only tensor.
public interface IWOTensor<T, let D : int> : IBaseTensor<T, D>
{
    public void write_buffer(int idx, T value);

    public void store<I : IInteger>(I idx[D], T value)
    {
        this.write_buffer(_idx(idx, strides, offset), value);
    }
    public void store<I : IInteger>(vector<I, D> idx, T value)
    {
        this.write_buffer(_idx(idx, strides, offset), value);
    }
    public void store<each I : IInteger>(expand each I indices, T value)
    {
        this.write_buffer(_idx(expand each indices, strides, offset), value);
    }

    // NOTE: Can't do this IF I want IRWTensor to inherit from IROTensor and IWOTensor
    // public __subscript<I : IInteger>(I indices[D])->T
    //{
    //    set;
    //}
    // public __subscript<I : IInteger>(vector<I, D> indices)->T
    //{
    //    set;
    //}
    // public __subscript<each I : IInteger>(expand each I indices)->T
    //{
    //    set;
    //}

    /*__slangpy_storers*/
    public void __slangpy_store(ContextND<D> context, in T value)
    {
        this.store(context.call_id, value);
    }
    //public void __slangpy_store<let N : int>(ContextND<D - 1> context, in T[N] value);
    //public void __slangpy_store<let N : int>(ContextND<D - 1> context, in vector<T, N> value);
    //public void __slangpy_store<let M : int, let N : int>(ContextND<D - 2> context, in T value[M][N]);
    //public void __slangpy_store<let M : int, let N : int>(ContextND<D - 2> context, in matrix<T, M, N> value);
}

// Public interface for a read-write tensor. This can be used as the argument
// to a function (or a generic constraint) to indicate it should take any
// type of read-write tensor.
public interface IRWTensor<T, let D : int> : IROTensor<T, D>, IWOTensor<T, D>
{
    // NOTE: I could add this to the RW tensor because it doesn't conflict with the RO or WO versions.
    // However that's pretty inconsistent with the other interfaces not having it.
    //public __subscript<I : IInteger>(I indices[D])->T
    //{
    //    get;
    //    set;
    //}
    // public __subscript<I : IInteger>(vector<I, D> indices)->T
    //{
    //    get;
    //    set;
    //}
    // public __subscript<each I : IInteger>(expand each I indices)->T
    //{
    //    get;
    //    set;
    //}
}



// A read only tensor
public struct ROTensor<T, let D : int> : IROTensor<T, D>
{
    // Underlying data storage
    private StructuredBuffer<T> _data;
    public uint[D] _shape;
    public uint[D] _strides;
    public uint _offset;

    public property shape : uint[D] { get { return _shape; } }
    public property strides : uint[D] { get { return _strides; } }
    public property offset : uint { get { return _offset; } }

    public T read_buffer(int idx)
    {
        return _data[idx];
    }

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
}

// Type alias for a read only tensor.
public typealias Tensor<T, let D : int> = ROTensor<T, D>;

public struct WOTensor<T, let D : int> : IWOTensor<T, D>
{
    // Underlying data storage
    private RWStructuredBuffer<T> _data;
    public uint[D] _shape;
    public uint[D] _strides;
    public uint _offset;

    public property shape : uint[D] { get { return _shape; } }
    public property strides : uint[D] { get { return _strides; } }
    public property offset : uint { get { return _offset; } }

    public void write_buffer(int idx, T value)
    {
        _data[idx] = value;
    }

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
}

// NOTE: By not using default implementations or fancy extensions, RWTensor
// duplicates all the code for loading / storing.
public struct RWTensor<T, let D : int> : IRWTensor<T, D>
{
    // Underlying data storage
    private RWStructuredBuffer<T> _data;
    public uint[D] _shape;
    public uint[D] _strides;
    public uint _offset;

    public property shape : uint[D] { get { return _shape; } }
    public property strides : uint[D] { get { return _strides; } }
    public property offset : uint { get { return _offset; } }

    public T read_buffer(int idx)
    {
        return _data[idx];
    }
    public void write_buffer(int idx, T value)
    {
        _data[idx] = value;
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
}

#endif
