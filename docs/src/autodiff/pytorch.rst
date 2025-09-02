PyTorch
=======

Building on the previous `auto-diff <autodiff.html>`_ example, the switch to PyTorch and its auto-grad capabilities is straightforward.

Initialization
--------------

To use SlangPy with PyTorch, you first need to create a device configured for PyTorch integration:

.. code-block:: python

    import slangpy as spy
    import torch

    # Create a device configured for PyTorch integration
    # CUDA backend is recommended for best performance
    device = spy.create_torch_device(type=spy.DeviceType.cuda)

    # Load module using the standard Module type
    module = spy.Module.load_from_file(device, "example.slang")

SlangPy automatically detects when PyTorch tensors are used and integrates them into PyTorch's auto-grad graph. No special module types are needed - you can use the standard ``spy.Module`` type as documented in `First Functions <../basics/firstfunctions.html>`_.

Creating a tensor
-----------------

Now, rather than use a SlangPy ``Tensor``, we create a ``torch.Tensor`` to store the inputs:

.. code-block:: python

    # Create a tensor
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device='cuda', requires_grad=True)

Note:

- We set ``requires_grad=True`` to tell PyTorch to track the gradients of this tensor.
- We set ``device='cuda'`` to ensure the tensor is on the GPU and matches our device configuration.

Running the kernel
------------------

Calling the function is unchanged from the standard SlangPy API, but calculation of gradients is now done via PyTorch:

.. code-block:: python

    # Evaluate the polynomial. Result will automatically be a torch tensor.
    # Expecting result = 2x^2 + 8x - 1
    result = module.polynomial(a=2, b=8, c=-1, x=x)
    print(result)

    # Run backward pass on result, using result grad == 1
    # to get the gradient with respect to x
    result.backward(torch.ones_like(result))
    print(x.grad)

This works because SlangPy automatically detects PyTorch tensors and wraps the call to `polynomial` in a custom autograd function. As a result, the call to `result.backward` automatically invokes `module.polynomial.bwds` to compute gradients.

Device Backend Selection
------------------------

SlangPy supports multiple backend types for PyTorch integration:

**CUDA Backend (Recommended)**

The CUDA backend provides the best performance by directly sharing the CUDA context with PyTorch:

.. code-block:: python

    device = spy.create_torch_device(type=spy.DeviceType.cuda)

This approach avoids expensive context switching and memory copies, making it ideal for performance-critical applications.

**Graphics Backends (D3D12, Vulkan)**

For applications that need access to graphics features (such as rasterization), you can use D3D12 or Vulkan backends:

.. code-block:: python

    # D3D12 backend (Windows only)
    device = spy.create_torch_device(type=spy.DeviceType.d3d12)

    # Vulkan backend (Cross-platform)
    device = spy.create_torch_device(type=spy.DeviceType.vulkan)

These backends use CUDA interop with shared memory and semaphores to synchronize between SlangPy and PyTorch. While functional, this approach has higher overhead due to hardware context switching and memory copies.

A word on performance
---------------------

The choice of backend significantly impacts performance:

- **CUDA Backend**: Provides the best performance for compute-focused workloads. Very simple operations may still be faster in pure PyTorch, but as functions become more complex, the benefits of SlangPy's vectorization and GPU optimization become apparent.

- **Graphics Backends (D3D12/Vulkan)**: Useful when graphics features are required, but expect substantially worse performance due to context switching overhead. Consider whether the graphics features are truly necessary for your use case.

Summary
-------

PyTorch integration with SlangPy is seamless and automatic. This example covered:

- Device creation using `create_torch_device` with support for CUDA, D3D12, and Vulkan backends
- Automatic detection of PyTorch tensors - no special module types required
- Use of PyTorch's `.backward()` process to track an auto-grad graph and backpropagate gradients
- Performance considerations when choosing between CUDA and graphics backends

The CUDA backend is recommended for best performance, while graphics backends provide access to additional GPU features at the cost of some performance overhead.
