PyTorch
=======

Building on the previous `auto-diff <autodiff.rst>`_ example, the switch to PyTorch and its auto-grad capabilities is trivial.

Initialization
--------------

The critical line that changes is loading the module:

.. code-block:: python

    # Load torch wrapped module.
    module = spy.TorchModule.load_from_file(device, "example.slang")

Here, rather than simply write ``spy.Module.load_from_file``, we write ``spy.TorchModule.load_from_file``. From here, all structures or functions utilizing the module will support PyTorch tensors and be injected into PyTorch's auto-grad graph.

In future SlangPy versions we intend to remove the need for wrapping altogether, instead auto-detecting the need for auto-grad support at the point of call.

Creating a tensor
-----------------

Now, rather than use a SlangPy ``Tensor``, we create a ``torch.Tensor`` tensor to store the inputs:

.. code-block:: python

    # Create a tensor
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device='cuda', requires_grad=True)

Note:

- We set ``requires_grad=True`` to tell PyTorch to track the gradients of this tensor.
- We set ``device='cuda'`` to ensure the tensor is on the GPU.

Running the kernel
------------------

Calling the function is pretty much unchanged, however calculation of gradients is now done via PyTorch:

.. code-block:: python

    # Evaluate the polynomial. Result will now default to a torch tensor.
    # Expecting result = 2x^2 + 8x - 1
    result = module.polynomial(a=2, b=8, c=-1, x=x)
    print(result)

    # Run backward pass on result, using result grad == 1
    # to get the gradient with respect to x
    result.backward(torch.ones_like(result))
    print(x.grad)

This works because the wrapped PyTorch module automatically wrapped the call to `polynomial` in a custom autograd function. As a result, the call to `result.backwards` automatically called `module.polynomial.bwds`.

A word on performance
---------------------

This example showed a very basic use of PyTorch's auto-grad capabilities. However in practice, the switch from a CUDA PyTorch context to a D3D or Vulkan context has an overhead. Typically, very simple logic will be faster in PyTorch. However as functions become more complex, writing them as simple scalar processes that are vectorised by SlangPy and wrapped in PyTorch quickly becomes apparent.

Additionally, we intend to add a pure CUDA backend to SlangPy in the future, which will allow for seamless switching between PyTorch and SlangPy contexts.

Summary
-------

That's it! You can now use PyTorch tensors with SlangPy, and take advantage of PyTorch's auto-grad capabilities. This example covered:

- Initialization with a `TorchModule` to enable PyTorch support 
- Use of PyTorch's `.backward` process to track an auto-grad graph and back propagate gradients.
- Performance considerations when wrapping Slang code with PyTorch.


