CUDA Graphs
===========

A `CUDA graph <https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs>`_ records a sequence of GPU operations once, then replays the whole sequence with a single launch. For work that repeats the same small operations many times, such as a training step, this removes most of the per-call CPU overhead.

PyTorch records CUDA graphs with ``torch.cuda.graph``. SlangPy provides ``spy.capture_torch_graph``, which is used in its place and records SlangPy calls into the graph together with the PyTorch operations around them.

Requirements
------------

Graph capture needs a CUDA device that shares PyTorch's context, created as described in the `PyTorch <pytorch.html>`_ section:

.. code-block:: python

    import slangpy as spy
    import torch

    device = spy.create_torch_device(type=spy.DeviceType.cuda)
    module = spy.Module.load_from_file(device, "example.slang")

D3D12 and Vulkan devices do not run on a CUDA stream, so their calls cannot be recorded. ``capture_torch_graph`` raises a ``ValueError`` if given one.

Recording a graph
-----------------

Wrap the code to record in ``spy.capture_torch_graph``, then call ``replay()`` on the graph it returns:

.. code-block:: python

    x = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device="cuda")

    def step() -> torch.Tensor:
        # A SlangPy call followed by a PyTorch operation
        return module.polynomial(a=2, b=8, c=-1, x=x) + 1.0

    # Record. Nothing inside the block is executed.
    with spy.capture_torch_graph(device) as graph:
        result = step()

    # Run the recorded operations
    graph.replay()
    print(result)

Note:

- Nothing runs while recording. ``result`` holds valid data only after ``graph.replay()``.
- A call made for the first time inside the block has its kernel generated and compiled there. This only happens while recording, and adds nothing to the cost of a replay.
- Only SlangPy calls that take a ``torch.Tensor`` are recorded. Those run on PyTorch's current CUDA stream, which is the stream being captured.

Replaying with new inputs
-------------------------

The graph refers to the memory of the tensors used while recording. To run it on new data, write into those tensors in place and replay:

.. code-block:: python

    for scale in (1.0, 5.0):
        x.copy_(torch.tensor([1, 2, 3, 4], dtype=torch.float32, device="cuda") * scale)
        graph.replay()
        print(result)

Assigning a new tensor to ``x`` has no effect on the graph, as it still reads the original one. The same applies to outputs: each replay writes into the ``result`` tensor created during recording.

Options
-------

``capture_torch_graph`` accepts an existing ``torch.cuda.CUDAGraph`` as its second argument, and passes any further keyword arguments on to ``torch.cuda.graph``:

.. code-block:: python

    graph = torch.cuda.CUDAGraph()
    with spy.capture_torch_graph(device, graph, capture_error_mode="global"):
        result = step()

By default the graph is recorded with ``capture_error_mode="relaxed"``, which allows objects that own GPU memory (including earlier graphs) to be released inside the block.

Lifetime
--------

The command buffers SlangPy submits while recording are not run and released as usual. The graph points into their memory every time it is replayed, so they are attached to the graph and released together with it. No extra bookkeeping is needed: keep the graph alive for as long as it is replayed, and drop it when finished.

Summary
-------

- Use ``spy.capture_torch_graph(device)`` in place of ``torch.cuda.graph`` to record SlangPy calls and PyTorch operations into one CUDA graph.
- A CUDA device created with ``create_torch_device`` is required.
- Update inputs in place, and call ``graph.replay()`` to run.
