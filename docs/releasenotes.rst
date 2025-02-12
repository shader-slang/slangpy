Release Notes
=============

**Version 0.20.0**

The main change in this release was to address the confusion over how a Slang vector type is interpreted when used to represent coordinates
in the call id grid, or as indices into a buffer. To map naturally to images and the Texture.Load function in Slang, the vector coordinates 
are treated as the **transpose** of their corresponding coordinates when specified as an array. This means that for a given call id [A,B,C],
a corresponding vector representation would be int3(C,B,A).  

Known Issues:

- PyTorch support is working but underperforms due to CUDA interop
- PyTorch currently makes buffers contigous during CUDA buffer copies
- MacOS is supported by SGL/SlangPy, but not well tested
- Passing int64s and doubles currently causes resolution to fail in certain situations
- Value refs still contain none-native dispatch code.

**Version 0.19.5**

Past releases since 0.17 have focussed on extensive bug fixes, documentation and optimizations. This release includes 
documentation for and examples of the generators system in SlangPy. 

**Version 0.17.0**

This release includes a major update to the `nv-sgl` library, which now includes CoopVec support, alongside more significant optimizations 
for buffers, textures and tensors. Planned imminent API changes to the 'from_numpy' functions on tensors/buffers.

**Version 0.15.0**

Recent changes have added extensive optimizations through native extension to the dispatch process, and fully functional PyTorch autograd support.

