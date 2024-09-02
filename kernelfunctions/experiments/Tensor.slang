import Helpers;
import Atomics;

// ********************************** Interfaces (only these are exposed to the user) ******************

// Abstract interface of a D-dimensional tensor with elements T.
// Read-only tensors only expose their shape and reading elements
// at some D-dimensional index.
interface ITensor<T, let D : int>
{
    property uint[D] shape { get; }

    [BackwardDifferentiable] T get(uint idx[D]);
}
// Read-write tensors additionally allow writing elements at a D-dimensional index
interface IRWTensor<T, let D : int> : ITensor<T, D>
{
    [BackwardDifferentiable] void set(uint idx[D], T value);
}


// Helper interfaces for more concrete implementations of tensors.
// Unclear if these should be exposed to the user

// Allows safely accumulating values from multiple threads (for accumulating gradients)
interface IGradTensor<T : IAtomicAddable, let D : int> : IRWTensor<T, D>
{
    void accumulate(uint idx[D], T value);
}

// The physical layout of a tensor, i.e. the mapping from a D-dimensional
// index to a flat index in the underlying storage.
// Most commonly just a strided view with strides+offset specified
// at run-time, but could also be e.g. layout with statically known strides,
// a sparse layout, a hash grid mapping, etc.
interface ITensorLayout<let D : int>
{
    // Map D-dimensional index to flat index
    uint at(uint idx[D]);
}

// Abstract interfaces for a typed storage that can read, write, or
// accumulate (i.e. atomic adding from multiple threads).
// Indices are in whole units of T (not bytes).
interface ITensorStorage<T>
{
    T get(uint idx);
}
interface IRWTensorStorage<T> : ITensorStorage<T>
{
    void set(uint idx, T value);
}
interface IGradTensorStorage<T : IAtomicAddable> : IRWTensorStorage<T>
{
    void accumulate(uint idx, T value);
}

// Syntactic sugar: Allow accessing a tensor with [] syntax. Support array, vector and bare indices
// I.e. these are all equivalent:
//
//  ITensor<float, 2> t;
//
//  t[2, 5] = 1.0f;
//
//  t[uint2(2, 5)] = 1.0f;
//
//  uint idx[2] = {2, 5};
//  t[idx] = 1.0f;
__generic<TensorType : ITensor<ElementType, D>, ElementType, let D : int>
extension TensorType
{
    __generic<T : IInteger>
    __subscript(ISizedArray<T, D> indices)->ElementType
    {
        [BackwardDifferentiable] get { return get(impl::makeIndex<D, T>(indices)); }
    }
    __generic<each Ts : IInteger>
    __subscript(expand each Ts indices)->ElementType
    {
        [BackwardDifferentiable] get { return get(impl::makeIndex<D>(expand each indices)); }
    }
}
__generic<TensorType : IRWTensor<ElementType, D>, ElementType, let D : int>
extension TensorType
{
    __generic<T : IInteger>
    __subscript(ISizedArray<T, D> indices)->ElementType
    {
        [BackwardDifferentiable] set { set(impl::makeIndex<D, T>(indices), newValue); }
    }
    __generic<each Ts : IInteger>
    __subscript(expand each Ts indices)->ElementType
    {
        [BackwardDifferentiable] set { set(impl::makeIndex<D>(expand each indices), newValue); }
    }
}
namespace impl
{
    // Helper function to turn a variadic list into an array of statically known size
    // Slang seems to crash with variadic constructors, so this helper is needed for now
    uint[D] makeIndex<let D : int, each T : IInteger>(expand each T indices)
    {
        // We can't currently specify a type constraint that we expect exactly D Ts, so
        // static_assert is needed
        static_assert(countof(T) == D, "Number of indices does not match Tensor dimensionality");
        uint[D] idxVec;
        int i = 0;
        expand idxVec[i++] = (each indices).toUInt();
        return idxVec;
    }
    // Helper for turning integer array-likes into indices
    uint[D] makeIndex<let D : int, T : IInteger>(ISizedArray<T, D> indices)
    {
        uint[D] idxVec;
        for (int i = 0; i < D; ++i)
            idxVec[i] = indices[i].toUInt();
        return idxVec;
    }
}

// ********************************** Implementation (generally invisible to the user) ******************

// Implementations of the storage for common types (i.e. structured/byte address buffers)
// Can extend in the future for other types of storage (textures? shared memory? cuda ptrs?)
__generic<T>
extension StructuredBuffer<T> : ITensorStorage<T>
{
    T get(uint idx) { return this[idx]; }
}
__generic<T>
extension RWStructuredBuffer<T> : IRWTensorStorage<T>
{
    T get(uint idx) { return this[idx]; }
    void set(uint idx, T value) { this[idx] = value; }
}
struct TypedByteAddressBuffer<T> : ITensorStorage<T>
{
    ByteAddressBuffer buf;

    T get(uint idx)
    {
        return buf.Load<T>(idx * sizeof(T), sizeof(T));
    }
}
struct TypedRWByteAddressBuffer<T> : IRWTensorStorage<T>
{
    RWByteAddressBuffer buf;

    T get(uint idx)
    {
        return buf.Load<T>(idx * sizeof(T), sizeof(T));
    }
    void set(uint idx, T value)
    {
        buf.Store<T>(idx * sizeof(T), value, sizeof(T));
    }
}
__generic<T : IAtomicAddable>
extension TypedRWByteAddressBuffer<T> : IGradTensorStorage<T>
{
    void accumulate(uint idx, T value)
    {
        T::atomicAdd(buf, idx * sizeof(T), value);
    }
}

// Strided layout with strides+offset specified at runtime
struct StridedLayout<let D : int> : ITensorLayout<D>
{
    typealias SlicedLayout = StridedLayout<D - 1>;

    uint strides[D];
    uint offset;

    // Map D-dimensional index to flat index
    uint at(uint idx[D])
    {
        uint result = offset;
        for (int i = 0; i < D; ++i)
        {
            result += strides[i] * idx[i];
        }
        return result;
    }

    // Create a view of a slice of this tensor. For e.g.
    // a 5-D tensor (with indices (x, y, z, u, v)) and a 3-D index (x, y, z),
    // return a view of the resulting 2-D slice (with indices (u, v))
    __generic<let SliceD : int>
    StridedLayout<D - SliceD> slice(uint[SliceD] idx)
    {
        StridedLayout<D - SliceD> result;
        for (uint i = 0; i < D - SliceD; ++i)
            result.strides[i] = strides[SliceD + i];
        result.offset = offset;
        for (uint i = 0; i < SliceD; ++i)
            result.offset += strides[i] * idx[i];
        return result;
    }
}

// Straightforward implementation of a tensor: Just index the storage
// (of any storage type) with the given layout. This implementation does
// nothing during differentiation and can be used for e.g. primal functions
// where we know backwards differentiation doesn't happen
struct TensorWithStorage<T, let D : int, Layout : ITensorLayout<D>, Storage : ITensorStorage<T>> : ITensor<T, D>
{
    Layout layout;
    Storage data;
    uint[D] size;

    property uint[D] shape
    {
        get { return size; }
    }

    [TreatAsDifferentiable]
    T get(uint idx[D])
    {
        return data.get(layout.at(idx));
    }
}
// If the storage is writeable, the tensor is, too
__generic<T, let D : int, Layout : ITensorLayout<D>, Storage : IRWTensorStorage<T>>
extension TensorWithStorage<T, D, Layout, Storage> : IRWTensor<T, D>
{
    [TreatAsDifferentiable]
    void set(uint idx[D], T value)
    {
        data.set(layout.at(idx), value);
    }
}
// If the storage allows accumulation, so does the tensor
__generic<T : IAtomicAddable, let D : int, Layout : ITensorLayout<D>, Storage : IGradTensorStorage<T>>
    extension TensorWithStorage<T, D, Layout, Storage> : IGradTensor<T, D>
{
    void accumulate(uint idx[D], T value)
    {
        data.accumulate(layout.at(idx), value);
    }
}
// Tensors with sliceable layout are also sliceable
__generic<T, let D : int, Storage : ITensorStorage<T>>
extension TensorWithStorage<T, D, StridedLayout<D>, Storage>
{
    __generic<let SliceD : int>
    TensorWithStorage<T, D - SliceD, StridedLayout<D - SliceD>, Storage> slice(uint idx[SliceD])
    {
        TensorWithStorage<T, D - SliceD, StridedLayout<D - SliceD>, Storage> result;
        result.layout = layout.slice(idx);
        result.data = data;
        for (uint i = 0; i < D - SliceD; ++i)
            result.size[i] = size[SliceD + i];
        return result;
    }
}

// Implementation of a differentiable tensor.
// Stores a tensor of primal values, and a separate tensor that stores
// gradients. Reading this tensor reads the primal tensor; differentiating
// reads accumulates into the gradient tensor.
struct DiffTensor<T : IDiffAtomicAddable, let D : int, Primal : ITensor<T, D>, Diff : IGradTensor<T.Differential, D>> : ITensor<T, D>
{
    Primal primal;
    Diff grads;

    property uint[D] shape
    {
        get { return primal.shape; }
    }

    [BackwardDifferentiable]
    T get(uint idx[D])
    {
        return primal.get(idx);
    }
    [BackwardDerivativeOf(get)]
    void get_bwd(uint idx[D], T.Differential dOut)
    {
        grads.accumulate(idx, dOut);
    }
}
// WIP implementation of a differentiable RW tensor - some issues here that need thinking about, including a slang bug
__generic<T : IDiffAtomicAddable, let D : int, Primal : IRWTensor<T, D>, Diff : IGradTensor<T.Differential, D>>
extension DiffTensor<T, D, Primal, Diff> : IRWTensor<T, D>
{
    [BackwardDifferentiable]
    void set(uint idx[D], T value)
    {
        primal.set(idx, value);
    }
    [BackwardDerivativeOf(set)]
    void set(uint idx[D], inout DifferentialPair<T> dValue)
    {
        let grad = grads.get(idx);

        // Workaround for slang bug
        typealias dT = DifferentialPair<T>::DifferentialElementType;
        let gradCast = __slang_noop_cast<dT>(grad);

        dValue = diffPair(dValue.p, gradCast);
    }
}
