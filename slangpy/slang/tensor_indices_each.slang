// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

#if 0

void _idx_accumulate_helper<let D : int, I : __BuiltinIntegerType>(I value, inout int dest, inout int dim, uint stride[D])
{
    dest += value.toInt() * stride[dim++];
}
int _idx<let D : int, each I : __BuiltinIntegerType>(expand each I idx, uint stride[D], uint offset)
{
    static_assert(countof(I) == D, "Mismatch between index count and tensor dimension");
    int res = 0;
    int dim = 0;
    expand _idx_accumulate_helper( each idx, res, dim, stride );
    return res + offset;
}

public extension<T, let D : int, TensorType : IROTensor<T, D>> TensorType
{
    public T load<each I : __BuiltinIntegerType>(I i);
}

public extension<T, let D: int> ROTensor<T, D>
{
    public T load<each I : __BuiltinIntegerType>(I i)
    {
        static_assert(countof(I) == D, "Mismatch between load argument count and tensor dimension");
        int idx = _idx(expand each i, this._strides, this._offset);
        return this.read_buffer(idx);
    }
    public __subscript<each I : __BuiltinIntegerType>(I i)->T
    {
        get { return load(expand each i); }
    }
}

public extension<T, let D : int, TensorType : IWOTensor<T, D>> TensorType
{
    public void store<each I : __BuiltinIntegerType>(I i, T value);
}

public extension<T, let D : int> WOTensor<T, D>
{
    public void store<each I : __BuiltinIntegerType>(I i, T value)
    {
        static_assert(countof(I) == D, "Mismatch between store argument count and tensor dimension");
        int idx = _idx(expand each i, this._strides, this._offset);
        this.write_buffer(idx, value);
    }
    public __subscript<each I : __BuiltinIntegerType>(I i)->T
    {
        set { store(expand each i, newValue); }
    }
}


public extension<T, let D : int, TensorType : IRWTensor<T, D>> TensorType
{
    public T load<each I : __BuiltinIntegerType>(I i);
    public void store<each I : __BuiltinIntegerType>(I i, T value);
}

public extension<T, let D : int> RWTensor<T, D>
{
    public T load<each I : __BuiltinIntegerType>(I i)
    {
        static_assert(countof(I) == D, "Mismatch between load argument count and tensor dimension");
        int idx = _idx(expand each i, this._strides, this._offset);
        return this.read_buffer(idx);
    }
    public void store<each I : __BuiltinIntegerType>(I i, T value)
    {
        static_assert(countof(I) == D, "Mismatch between store argument count and tensor dimension");
        int idx = _idx(expand each i, this._strides, this._offset);
        this.write_buffer(idx, value);
    }
    public __subscript<each I : __BuiltinIntegerType>(I i)->T
    {
        get { return load(expand each i); }
        set { store(expand each i, newValue); }
    }
}

public extension<T: IDifferentiable, let D : int, TensorType : IRODiffTensor<T, D>> TensorType where T.Differential: IAtomicAddable
{
    [Differentiable]
    public T load<each I : __BuiltinIntegerType>(I i);
}

public extension<T : IDifferentiable, let D : int> RODiffTensor<T, D> where T.Differential : IAtomicAddable
{
    [Differentiable]
    [BackwardDerivative(_load_bwd_indices)]
    public T load<each I : __BuiltinIntegerType>(I i)
    {
        static_assert(countof(I) == D, "Mismatch between load argument count and tensor dimension");
        return this._primal.load(expand each i);
    }
    void _load_bwd_indices<each I : __BuiltinIntegerType>(I i, T.Differential grad)
    {
        _grad_out.add(expand each i, grad);
    }

    public __subscript<each I : __BuiltinIntegerType>(I i)->T
    {
        [Differentiable]
        get { return load(expand each i); }
    }
}

public extension<T : IDifferentiable, let D : int, TensorType : IWODiffTensor<T, D>> TensorType where T.Differential : IAtomicAddable
{
    public void store<each I : __BuiltinIntegerType>(I i, T value);
}

public extension<T : IDifferentiable, let D : int> WODiffTensor<T, D> where T.Differential : IAtomicAddable
{
    [Differentiable]
    [BackwardDerivative(_store_bwd_indices)]
    public void store<each I : __BuiltinIntegerType>(I i, T value)
    {
        static_assert(countof(I) == D, "Mismatch between store argument count and tensor dimension");
        this._primal.store(expand each i, value);
    }
    void _store_bwd_indices<each I : __BuiltinIntegerType>(I i, inout DifferentialPair<T> grad)
    {
        grad = diffPair(grad.p, _grad_in.load(expand each i));
    }

    public __subscript<each I : __BuiltinIntegerType>(I i)->T
    {
        [Differentiable]
        set { store(expand each i, newValue); }
    }
}

public extension<T : IDifferentiable, let D : int, TensorType : IRWDiffTensor<T, D>> TensorType where T.Differential : IAtomicAddable
{
    [Differentiable]
    public T load<each I : __BuiltinIntegerType>(I i);
}

public extension<T : IDifferentiable, let D : int> RWDiffTensor<T, D> where T.Differential : IAtomicAddable
{
    [Differentiable]
    [BackwardDerivative(_load_bwd_indices)]
    public T load<each I : __BuiltinIntegerType>(I i)
    {
        static_assert(countof(I) == D, "Mismatch between load argument count and tensor dimension");
        return this._primal.load(expand each i);
    }
    void _load_bwd_indices<each I : __BuiltinIntegerType>(I i, T.Differential grad)
    {
        _grad_out.add(expand each i, grad);
    }

    [Differentiable]
    [BackwardDerivative(_store_bwd_indices)]
    public void store<each I : __BuiltinIntegerType>(I i, T value)
    {
        static_assert(countof(I) == D, "Mismatch between store argument count and tensor dimension");
        this._primal.store(expand each i, value);
    }
    void _store_bwd_indices<each I : __BuiltinIntegerType>(I i, inout DifferentialPair<T> grad)
    {
        grad = diffPair(grad.p, _grad_in.load(expand each i));
    }

    public __subscript<each I : __BuiltinIntegerType>(I i)->T
    {
        [Differentiable]
        get { return load(expand each i); }
        [Differentiable]
        set { store(expand each i, newValue); }
    }
}

public extension<T, let D: int> AtomicTensor<T, D> where T : IAtomicAddable
{
    public T load<each I : __BuiltinIntegerType>(I i)
    {
        static_assert(countof(I) == D, "Mismatch between load argument count and tensor dimension");
        int idx = _idx(expand each i, this._strides, this._offset);
        return this.read_buffer(idx);
    }
    public void store<each I : __BuiltinIntegerType>(I i, T value)
    {
        static_assert(countof(I) == D, "Mismatch between store argument count and tensor dimension");
        int idx = _idx(expand each i, this._strides, this._offset);
        this.write_buffer(idx, value);
    }
    public void add<each I : __BuiltinIntegerType>(I i, T value)
    {
        int idx = _idx(expand each i, this._strides, this._offset);
        StorageTraits<T>::atomicAdd(_data, idx, value);
    }
    public __subscript<each I : __BuiltinIntegerType>(I i)->T
    {
        get { return load(expand each i); }
        set { store(expand each i, newValue); }
    }
}

#endif
