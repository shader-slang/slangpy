
# Background / thought process / brain dumps

<b>Note: Everything below is bits of the thought process / brain dumps that fed into the above design. They're kept for posterity / reference now, but not key to the API design.</b>

### Differentials

The interface to request differentials is tricky. I believe strongly that we need to give users the ability to request them without requiring PyTorch auto grad. The simplest starting point is to look at what slang actually generates in the case of forward and backward differentials:

```
xx.slang
//Function
[ForwardDifferentiable]
[Differentiable]
float myFunc(float a, float x)
{
    return a * x * x;
}

//Forward derivative
DifferentialPair<float> myFunc_fwd_derivative(
    DifferentialPair<float> a,
    DifferentialPair<float> x);

//Backward derivative
void myFunc_backProp(
    inout DifferentialPair<float> a,
    inout DifferentialPair<float> x,
    float dResult);
```

If we were to follow the same line of thinking thus far, the logical (though ugly) approach to making this accessible as a kernel function would be

```
yy.py
a = 3
x = 2
res = myFunc.call(a,x)
#res -> 12

#Use forward differentiation
dout_by_da = myFunc.forward.call(diffpair(a, 1), diffpair(x, 0))
dout_by_dx = myFunc.forward.call(diffpair(a, 0), diffpair(x, 1))
#dout_by_da.d -> 4
#dout_by_dx.d -> 12

#Use backwards differentiation
dout_by_da = diffpair(a)
dout_by_dx = diffpair(x)
myFunc.backward.call(dout_by_da, dout_by_dx, res)
```

This, however is not particularly user friendly, Python like, or remotely similar to how PyTorch operates!

To begin, I would propose we leave the forward option out of this convenience api (it can still be accessed via slang obviously), especially as it will cause extra confusion given in PyTorch world 'forward' is the 'feed forward' step.


In the above example, dealing with scalar, immutable values, there still isn't a lot of improvement could be done. However once dealing with lists of values, the PyTorch '.grad' model can be asily applied:

```
yy.py
a = [3]
x = [2]
res = myFunc.call(a,x)
#res -> [12]

#Use backwards differentiation
myFunc.backward.call(a, x, res)
#a.grad -> [4]
#x.grad -> [12]
```

If use of scalars was really desired, they could be wrapped in some minimal 'gradient supporting' type:

```
yy.py
a = DiffFloat(3)
x = DiffFloat(2)
res = myFunc.call(a,x)
#res -> 12

#Use backwards differentiation
myFunc.backward.call(a, x, res)
#a.grad -> 4
#x.grad -> 12
```

At that point, we could also introduce the 'needs grad' approach taken in PyTorch, either inferring it directly from the type provided, or checking for a 'needsgrad' property internally, and as in Copper only generate gradients for variables that're useful:

```
yy.py
a = DiffFloat(3)
x = 2
res = myFunc.call(a,x)
#res -> 12

#Use backwards differentiation
myFunc.backward.call(a, x, res)
#a.grad -> 4
```

This basic layout feels correct to me, and much more akin to how most Python libraries operate.

My knowledge of PyTorch at the moment isn't strong enough to know how we'd integrate that with PyTorch if-and-only-if we wanted to (i.e. PyTorch might not be there, or just because PyTorch is, doesn't mean calling backward means we want autograd doing stuff).

### Gradient accumulation

With the system above, even if we wanted to ignore gradient accumulation, we couldn't. Code such as the following would introduce an ambiguity:

```
xx.slang
[Differentiable]
float myFunc(float a, float x)
{
    return a * x * x;
}

yy.py
a = DiffFloat(3)
x = Tensor([1,2,3], needsgrad=True)
res = myFunc.call(a,x)
#res -> [3,12,27]

#Use backwards differentiation
myFunc.backward.call(a, x, res)
#a.grad -> ???
#x.grad -> [6,12,18]
```

Following the earlier batching rules we've assumed it's valid to pass a and x into backward to get resulting gradients. However whilst it was possible to batch differently during the call, there is now no clearly defined value for a.grad. You could argue:
- a.grad should be d/d_a for all values of res (i.e. a become list)
- a.grad should be an accumulation of d/d_a (i.e. remain a scalar)

A database 'group' query often hits a similar problem, and typically treats the first option as a special case of accumulation - 'push' accumulation, in which every result is added to list rather than accumulated through some mathematical function.

With that in mind, _if_ we wanted to the logical way to clear up the ambiguity would be some form of accumulator flag:

```
a.accumulator = 'sum'
myFunc.backward.call(a, x, res)
#a.grad -> sum of dres_by_da

a.accumulator = 'push'
myFunc.backward.call(a, x, res)
#a.grad -> list of dres_by_da

a.accumulator = 'first'
myFunc.backward.call(a, x, res)
#a.grad -> first dres_by_da
```

This has the neat benefit that it can be applied to both a AND x, with the batch dimensionality defined by res. For example, even if the input 'x' was a list, we could still ask gradients to be accumulated into a single scalar.

## Benedikit thoughts (18/07/24)

### Kernels/batching:
My gut feeling is that there should be a pretty explicit signal of user intent about what should be batched. In the Falcor copper version (Falcopper...?) this is basically decided by checking if an input is_tensor_like. This spec is a bit more liberal and allows e.g. lists to be batched as well (does that extend to all iterables?). I worry this might be a bit too accepting and would allow committing some weird bugs when code gets more complex. I think the batched list case is sufficiently rare---input data is either uniform or you have a lot of it and a list would be a poor choice then---that you can require things to be wrapped in a tensor-like object if they should be batched. Then you can reserve lists for describing slang types like arrays, vectors or matrices and throw more detailed errors when users make mistakes

In "More complex batching" you actually implemented something I was debating for a while, which is batching things with different dimensionality (i.e. shape [2, 3] and [3] batched together make an output shape [2, 3]). In the "array of numbers" world of python libraries, there are established rules for how this should work. Here's the torch page about broadcasting rules: https://pytorch.org/docs/stable/notes/broadcasting.html . This can be really useful, e.g. pass a tensor of shape [256, 1] and one of shape [1, 256], and the output is returned with shape [256, 256]. OTOH, it can lead to some amazing bugs when you accidentally swap some arguments, or they're just not the shape you thought they were, and the framework happily accepts it and mashes the shapes together into an enormous giga tensor of failure :slightly_smiling_face: For now I was erring on being more restrictive and requiring the batch sizes of everything to either be [] (i.e. broadcast) or to be exactly identical. I do admit I like the torch broadcasting rules---they're very convenient---but they also scare me a bit
The spec doesn't touch on inout / out parameters, are you planning to prohibit those in the initial version?

Writing the differentiation spec separate from torch.autograd is good. Whatever the spec looks like, we'll offer compatibility layers for various frameworks (like torch) and wrap it into the form their autograd expects. I only know the internals of torch, but I can double check the spec stays compatible with at least that.

### Some thoughts about differentiation:
The base spec only mentions marshalling of scalars/basic types, but differentiation already asks for some basic structure support. DifferentialPair is a struct with a .p and .d field, and those would hold different tensors (they come separately from the forward or the backward direction). When you go to higher derivatives, you even start nesting differential pairs, and .p and .d might not even nest to the same depth... My main motivation for dealing with struct marshalling early was that differentiation could be handled without needing special casing for DifferentialPairs (which would end up looking like struct marshalling anyway)

I understand why the diffpair API is required in slang, but from a user perspective it's quite painful, and every time I do non-trivial differentiation I have to go back and read the slang spec again to understand what it expects. I wonder if we can make it easier to use from the python side; I have some thoughts, but I'm also not sure this is needed, since most of the time it will be hidden behind some autograd framework anyway.

Outputs from kernel calls need to be tensor-like if you want them to be tracked by the autograd from other frameworks. Doing something like z = add(x,y) with everything a python float would break, invisibly---the user could go on to pass z to other things, and then invoke e.g. backward down the line expecting everything to propagate, but the graph is disconnected at z in a way that can't be detected. You can't even abstract it in a way where the base layer talks scalars and somewhere higher in the abstraction (e.g. the torch layer) wraps it into tensors; as soon as you unwrap it into a scalar / python type you lose tracking. For that reason I was forcing return values and out / inout parameters to always be tensors

### About accumulation:

Gradient accumulation warrants being extremely careful. The easiest thing for now is to just not do any of it: Accumulation is none of copper's business; if you want gradients of a parameter, it has to be batched with no broadcasting whatsoever, and you will get back a batch's worth of gradients. On the torch side, before invoking the kernel you would then e.g. call expand to turn your model parameters into a batch's worth of copies of the model parameters, and  torch.autograd will perform the sum for you when it backprops through the expand call. This will work for basic things, but when you have many parameters then this becomes impractical. Say you have a 4k texture and a batch size of 64k; may I offer you a giga tensor of 4096 * 4096 * 65536 gradients?

The next step is to support accumulation and use the same rules as autograd frameworks would. Mathematically, differentiating a broadcast always becomes a sum. We know when the user broadcasts an input, and so internally we know we have to do accumulation when you call backward. With some care this should be doable and intuitive. There might be cases when an input is broadcast only over some dimensions; e.g. in the "More complex batching" example, y=[1,2,3] is broadcast over dimension 0 and batched over dimension 1. Mathematically you would sum gradients over dimension 0 and get back a 1D tensor of gradients, and this is what happens in any current autodiff framework. We could do the same, but I'm wary of the bug attack surface---it's actually the main reason I decided against allowing more liberal broadcasting rules for now.
