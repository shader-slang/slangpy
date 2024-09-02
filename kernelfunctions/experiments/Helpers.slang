// Shared implementation for anything that implements IArray and has a statically known size
interface ISizedArray<T, let D : int> : IArray<T> {}

// Array, vector and matrix conform to ISizedArray
__generic<T, let D : int>
extension Array<T, D> : ISizedArray<T, D> {}
__generic<T, let D : int>
extension vector<T, D> : ISizedArray<T, D> {}
__generic<T, let R : int, let C : int, let L : int>
extension matrix<T, R, C, L> : ISizedArray<vector<T, C>, R> {}
