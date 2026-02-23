// DiffTensorView extensions for slangtorch.
// Provides loadUniform, loadVecUniform, loadVecOnce, storeVecOnce helpers
// with correct wave-level gradient reductions for uniform (broadcast) loads.
//
// Based on github.com/nv-tlabs/ppisp (Apache 2.0).

#pragma once

__generic<T : __BuiltinFloatingPointType>
extension TensorView<T> {
    __generic <let M : int, let R : int, let N : int>
    vector<T, M> loadVec(vector<uint, N> x) {
        vector<T, M> result;
        [ForceUnroll]
        for (int j = 0; j < M; j++) {
            result[j] = this.load(x);
            x[R] += 1;
        }
        return result;
    }

    __generic <let M : int, let N : int>
    vector<T, M> loadVec(vector<uint, N> x) {
        return this.loadVec<M, N - 1, N>(x);
    }
}

__generic<T : __BuiltinFloatingPointType, A : IDiffTensorWrapper>
extension DiffTensorView<T, A>
{
    [ForwardDerivative(__loadExUniform_forward)]
    [BackwardDerivative(__loadUniform_backward)]
    [NoDiffThis]
    __generic<let N : int>
    T loadUniform(vector<uint, N> x) { return primal.load(x); }

    __generic<let N : int>
    DifferentialPair<T> __loadExUniform_forward(vector<uint, N> x)
    {
        return diffPair(primal.load(x), reinterpret<T.Differential, T>(diff.load_forward<T, N>(x)));
    }

    __generic<let N : int>
    void __loadUniform_backward(vector<uint, N> x, T.Differential dOut)
    {
        T reducedGrad = WaveActiveSum(reinterpret<T, T.Differential>(dOut));
        if (WaveIsFirstLane())
        {
            diff.load_backward<T, N>(x, reducedGrad);
        }
    }

    [ForwardDerivative(__loadVecExUniform_forward)]
    [BackwardDerivative(__loadVecUniform_backward)]
    [NoDiffThis]
    __generic<let M : int, let R : int, let N : int>
    vector<T, M> loadVecUniform(vector<uint, N> x) {
        vector<T, M> result;
        [ForceUnroll]
        for (int j = 0; j < M; j++) {
            result[j] = primal.load(x);
            x[R] += 1;
        }
        return result;
    }

    [Differentiable]
    __generic<let M : int, let N : int>
    vector<T, M> loadVecUniform(vector<uint, N> x) {
        return this.loadVecUniform<M, N - 1, N>(x);
    }

    __generic<let M: int, let R : int, let N : int>
    DifferentialPair<vector<T, M>> __loadVecExUniform_forward(vector<uint, N> x)
    {
        vector<T, M> y;
        vector<T, M> dy;
        [ForceUnroll]
        for (int j = 0; j < M; j++) {
            y[j] = primal.load(x);
            dy[j] = diff.load_forward<T, N>(x);
            x[R] += 1;
        }
        return diffPair(y, reinterpret<vector<T, M>.Differential, vector<T, M>>(dy));
    }

    __generic<let M: int, let R : int, let N : int>
    void __loadVecUniform_backward(vector<uint, N> x, vector<T, M>.Differential dOut)
    {
        vector<T, M> reducedGrad = WaveActiveSum(dOut);
        if (WaveIsFirstLane())
        {
            [ForceUnroll]
            for (int j = 0; j < M; j++) {
                diff.load_backward<T, N>(x, reducedGrad[j]);
                x[R] += 1;
            }
        }
    }

    [Differentiable]
    __generic<let M: int, let R : int, let N : int>
    vector<T, M> loadVecOnce(vector<uint, N> x) {
        vector<T, M> result;
        [ForceUnroll]
        for (int j = 0; j < M; j++) {
            result[j] = this.loadOnce(x);
            x[R] += 1;
        }
        return result;
    }

    [Differentiable]
    __generic<let M: int, let N : int>
    vector<T, M> loadVecOnce(vector<uint, N> x) {
        return this.loadVecOnce<M, N - 1, N>(x);
    }

    [Differentiable]
    __generic<let M: int, let R : int, let N : int>
    void storeVecOnce(vector<uint, N> x, vector<T, M> y) {
        [ForceUnroll]
        for (int j = 0; j < M; j++) {
            this.storeOnce(x, y[j]);
            x[R] += 1;
        }
    }

    [Differentiable]
    __generic<let M: int, let N : int>
    void storeVecOnce(vector<uint, N> x, vector<T, M> y) {
        return this.storeVecOnce<M, N - 1, N>(x, y);
    }
};
