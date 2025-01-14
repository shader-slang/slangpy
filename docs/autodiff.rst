Auto-diff
=========

One Slang's most powerful features is it's auto-diff capabilities, documented in detail `here <https://shader-slang.com/slang/user-guide/autodiff.html>`_. SlangPy carries this feature over to Python, allowing you to easily the backwards derivative of a function.

Let's start with a simple polynomial function:

.. code-block:: 

    [Differentiable]
    float polynomial(float a, float b, float c, float x) {
        return a * x * x + b * x + c;
    }

Note that it has the ``[Differentiable]`` attribute, which tells Slang to generate the backward propagation function.

