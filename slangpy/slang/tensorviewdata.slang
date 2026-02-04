// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

implementing slangpy;

// TensorViewData - a struct with identical memory layout to CUDA's TensorView.
// Used as a wrapper in CallData because TensorView is a magic type with size=0
// in Slang's reflection, which breaks struct layout calculations.
public struct TensorViewData {
    uint64_t data;              // 8 bytes - GPU pointer
    uint32_t strides[5];        // 20 bytes - strides in BYTES
    uint32_t sizes[5];          // 20 bytes - shape
    uint32_t dimensionCount;    // 4 bytes
    // Total: 52 bytes + 4 padding = 56 bytes

    [require(cuda)]
    public TensorView<T> toTensorView<T>() {
        __intrinsic_asm "*reinterpret_cast<TensorView*>(&$0)", this;
    }

    [require(cuda)]
    public void __slangpy_load<T, let D : int>(ContextND<D> context, out TensorView<T> value) {
        value = this.toTensorView<T>();
    }

    // SlangPy store interface - no-op since TensorView writes directly to GPU memory
    [require(cuda)]
    public void __slangpy_store<T, let D : int>(ContextND<D> context, TensorView<T> value) {
    }
}
