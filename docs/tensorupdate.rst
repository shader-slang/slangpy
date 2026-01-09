.. _tensorupdate:

Migration guide for Tensor update
=================================

In SlangPy version X we undertook a major rewrite of the Tensor API to bring it inline with existing standards and simplify its use going forwards. The primary changes made are:

Python API
----------
- ``NDBuffer`` is fully deprecated
- ``Tensor`` is the sole ND container type, and supports both differentiable and non-differentiable data

Slang API
-----------
- ``NDBuffer`` is removed
- ``Tensor`` and ``RWTensor`` types now only store **non-differentiable** data
- ``WTensor`` introduced to store write-only non-differentiable data
- ``DiffTensor``, ``WDiffTensor`` and ``RWDiffTensor`` introduced to store differentiable data
- ``get`` and ``set`` methods replaced with ``load`` and ``store`` methods respectively
- ``getv`` and ``setv`` methods replaced with ``load`` and ``store`` methods respectively
- subscript operators correctly implemented for all Tensors

Migrating
---------

Most migration steps can be automated using search-and-replace, however doing them in the correct order is important to avoid naming issues.

1. Remove use of ``NDBuffer`` in Python code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Replace all instances of ``NDBuffer`` with ``Tensor``
- Fix any use of the ``NDBuffer`` constructor to use ``Tensor.empty``

Note: Use of ``NDBuffer`` constructor with positional arguments may not be easily fixable with search-and-replace. It is probably worth reviewing these instances manually.

2. Replace differentiable Tensors parameters in Slang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have exclusively used the Tensor type as function parameters, simply search and replacing as below should be sufficient.

- Replace all instances of ``Tensor`` with ``IDiffTensor``
- Replace all instances of ``RWTensor`` with ``IRWDiffTensor``
- Replace all instances of ``GradInTensor`` with ``IWDiffTensor``
- Replace all instances of ``GradOutTensor`` with ``IDiffTensor``
- Replace all instances of ``GradInOutTensor`` with ``IRWDiffTensor``

Where tensors have been used as variables, it can depend on the use case. Typically:

- Tensor/RWTensor will typically be either:
    - Keep the same if using purely as ND storage
    - Replace with PrimalTensor/ RWPrimalTensor if you need it to be compatible with IDiffTensor interfaces
- GradInTensor should be changed to WDiffTensor
- GradOutTensor should be changed to DiffTensor
- GradInOutTensor should be changed to RWDiffTensor

3. Replace NDBuffers with non-differentiable Tensors in Slang code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For function arguments:

- Replace all instances of ``NDBuffer`` with ``ITensor``
- Replace all instances of ``RWWBuffer`` with ``IRWTensor``

Where NDBuffers have been used as variables:

- Replace all instances of ``NDBuffer`` with ``Tensor``
- Replace all instances of ``RWBuffer`` with ``RWTensor``

4. Fix ``get``/``set`` and ``getv``/``setv`` method calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Replace all instances of ``.get(`` with ``.load(``
- Replace all instances of ``.set(`` with ``.store(``
- Replace all instances of ``.getv(`` with ``.load(``
- Replace all instances of ``.setv(`` with ``.store(``

5. Fix any errors attempting to access gradient buffers in Slang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If Slang has compile errors attempting to access the 'grad' properties of tensors,
switch them from using `IDiffTensor` to the concrete `DiffTensor` (or corresponding) types.
