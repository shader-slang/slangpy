# Trainable Neural Networks in Slangpy

This sample contains a reusable neural network library written for Slang and SlangPy, and an example using it to fit a network to a texture.

The focus of the library is on _neural shading_: Small, fused neural networks intended to run in real-time, such as inside a rendering pipeline. Model evaluation is done entirely in registers within a single fused kernel, and can take advantage of hardware acceleration via the CoopVec API. There is no intent to rival ML frameworks like pytorch geared towards models with billions of parameters, but the goal is to be just as easy to use.

## Running the Sample

There is sample code in `main.py` and `NeuralTexture.slang` showing how to use the library to fit a neural network to a texture. This is just intended as a demonstration, and is not state-of-the-art neural texture compression.

The sample may be run with
```
python main.py
```
and should converge to this output after a while:

![Picture of a cat represented by a neural network](./example-output.jpg)

No external libraries are needed, and it should run out of the box after installing SlangPy. See [the SlangPy docs](https://slangpy.shader-slang.org/en/latest/installation.html) for instructions for installing SlangPy.

## Library Documentation

The library consists of two components:
- A Slang library that can be imported via `import NeuralNetworks;` in slang, and runs the actual network code
- A Python module that can be imported via `import neuralnetworks` for setting up model architectures and training

### Table of Contents
1. [Motivation and Design Goals](#motivation-and-design-goals)
2. [Basic Usage](#basic-usage)
    1. [Evaluating a Model](#evaluating-a-model)
    2. [Creating the Model](#creating-the-model)
    3. [Training](#training)
    4. [Converting Inputs](#converting-inputs)
    5. [Automatic Parameters](#automatic-parameters)
3. [Adding New Components](#adding-new-components)
    1. [Implementing the Slang Component](#implementing-the-slang-component)
    2. [Exposing it in Python](#exposing-it-in-python)
    3. [Passing Parameters](#passing-parameters)
    4. [Making it Generic](#making-it-generic)
    5. [Supporting Auto Parameters](#supporting-auto-parameters)
    6. [Where to go from here](#where-to-go-from-here)

### Motivation and Design Goals

When designing and training a network for a new task, it's almost impossible to know ahead of time what architecture or training strategy will perform best. Developing a new neural model usually involves an exploration phase, where many different architectures and training setups are tried.

Frameworks like `pytorch` and `jax` are designed to make this easy: They provide a library of network components that adhere to a simple API, and make it easy to stick them together to build large networks.

The goal is to provide a similar experience in Slang and SlangPy, geared towards small networks. The main challenge is that our networks now run entirely in the strongly typed language of Slang instead of Python. This is very different from `pytorch`, where model execution is written in Python code that launches kernels to perform individual operations.

The need to build models dynamically in Python, but run them as fused Slang code is what motivated the current design of the library. The approach chosen here is to express trainable model components as Slang types deriving from a shared interface---`IModel`---with explicit inputs and outputs. This makes it possible to chain those components together to build up larger models using Slang's type system and generics: The entire model architecture essentially becomes one (very large) typedef. To make this easy to use, the Slang components are exposed to Python via simple classes that take care of building up the types under the hood and passing the data over to slang.

### Basic Usage

Let's look at how this is implemented, and how to setup and train/evaluate a basic neural network in SlangPy.

#### Evaluating a Model

All trainable models and model components derive from the same Slang interface:
```
interface IModel<InputT : IDifferentiable, OutputT : IDifferentiable>
{
    [BackwardDifferentiable]
    OutputT forward(InputT x);
}
```
Anything that implements `IModel` is guaranteed to provide a differentiable `forward` method that transforms an `InputT` into an `OutputT`. This lets you write generic code that works with any model regardless of its architecture, as long as it satisfies those constraints.

For example, in the neural texture sample, we use this function in `NeuralTexture.slang` to evaluate the network loss:
```
[BackwardDifferentiable]
float evalModelLoss<Model : IModel<float2, float3>>(Model model, no_diff float2 inputUV, no_diff float3 targetRGB)
{
    let lossFunc = Losses::L2();
    
    float3 prediction = model.forward(inputUV);

    return lossFunc.eval(prediction, targetRGB);
}
```
This function requests a model that satisfies `IModel<float2, float3>`, i.e. that takes a `float2` (UV) and returns a `float3` (RGB) in its `forward` method.

`forward` is marked differentiable, and we can back-propagate the loss gradient to the parameters of the model. For example, this is done by the `trainTexture` function in the same file:
```
void trainTexture<Model : IModel<float2, float3>>(Model model, inout RNG rng, Texture2D<float4> targetTex, SamplerState sampler, float lossScale)
{
    float2 inputUV = rng.next2D();

    float4 target = targetTex.SampleLevel(sampler, inputUV, 0.0f);

    bwd_diff(evalModelLoss)(model, inputUV, target.rgb, lossScale);
}
```
This evaluates the texture at a random location, samples the reference texture and backpropagates the loss gradient to the network parameters.

#### Creating the Model

From Python, we can set up the model architecture using classes from the `neuralnetworks` module. For example, `main.py` creates a neural texture model like this:
```
import neuralnetworks as nn

# Set up model architecture:
model = nn.ModelChain(
    # [ ... ]
    nn.FrequencyEncoding(5),
    # [ ... ]
    nn.LinearLayer(nn.Auto, 64),
    nn.LeakyReLU(),
    # [ ... ]
    nn.Exp(),
)
```
Here, `nn.LinearLayer`, `nn.LeakyReLU`, `nn.Exp` and `nn.FrequencyEncoding` represent simple network components in Slang that perform matrix multiplies, activations, etc. `nn.ModelChain` takes a list of components and composes them together into a larger network, where the output of the first component's `forward` is fed into the second, and so forth. If you're familiar with `pytorch`, this will look very similar to `torch.nn.Sequential`, `torch.nn.Linear`, etc.

The creation of the `model` is light-weight and doesn't do much work yet. To finalize it, we load the Slang module containing our training code and initialize the model:
```
# Load slang module containing our eval/training code and initialize the model
module = Module.load_from_file(device, "NeuralTexture.slang")
model.initialize(module, "float2")
```
The `initialize` call allocates the model parameters and does type checking to make sure the model is sound---for example, to check that the output of one component is actually compatible with the next. The second argument (`"float2"`) is the input type we intend to pass to the model in Slang.

Finally, with an initialized model in hand, we can start calling Slang functions that use it, e.g. to backprop the network gradients:
```
# [ ... ]
# Backpropagate network loss
module.trainTexture(model, rng, target_tex, sampler, loss_scale)
```
This calls the `void trainTexture<Model : IModel<float2, float3>>(Model model, [...])` function we saw earlier. In Python, the `model` object we constructed communicates to SlangPy the exact Slang type it maps to, and the generic `<Model : IModel....>` parameter is automatically inferred for us. The result is a fully fused kernel that executes the model, computes the loss, and backpropagates the gradients.

#### Training

Finally, we tie it all together by updating the model parameters with the gradients. This is done by creating an optimizer:
```
optim = nn.AdamOptimizer(learning_rate=0.0005)
optim.initialize(module, model.parameters())
```
Similar to the model, this starts with a light-weight creation step, followed by an `initialize` call that passes in a list of parameter Tensors we want to optimize.

The main training loop alternates computing the model gradients and a step of the optimizer:
```
for i in range(num_batches_per_epoch):
    # Backpropagate network loss
    module.trainTexture(model, rng, target_tex, sampler, loss_scale)
    # Update network parameters with gradients
    optim.step()
```

Creating the optimizer in two phases (constructor and `initialize`) allows us to do interesting things with optimizers.

For example, the neural texture sample evaluates the network at half precision if the hardware supports it. The `AdamOptimizer` does support running at half precision, but there's a potential concern about accumulating the rounding errors of 16bit floats over many training iterations. To solve this, we can wrap it in a `FullPrecisionOptimizer`:
```
optim = nn.AdamOptimizer(learning_rate=0.0005)
if mlp_precision == Real.half:
    optim = nn.FullPrecisionOptimizer(optim)
optim.initialize(module, model.parameters())
```
The `FullPrecisionOptimizer` keeps a running copy of the network parameters at full precision. During a `.step`, it runs the nested optimizer (here, Adam) on the full precision copy, and propagates the update to the network parameters. This way, model inference can keep the performance benefits of running in 16 bit, while optimization is done on the full weights and rounding errors are not accumulated.

#### Converting Inputs

The functions `trainTexture` and `evalModelLoss` expect a model that turns a `float2` into a `float3` (UV->RGB). However, what if we want to try running the model at a different precision (e.g. half)? It would be inconvenient to update all the places it is referenced in the code to use `half2` and `half3`.

To help with this, the `Convert` component can perform a number of different casts between input types. For example, we could set up our model like this:
```
model = nn.ModelChain(
    nn.Convert.to_half(),
    nn.FrequencyEncoding(5),
    # [ ... ]
    nn.Exp(),
    nn.Convert.to_float()
)
```
This converts the `float2` input to a `half2` and passes it on, which causes the rest of the network to be evaluated at half precision. Finally, we convert the `half3` output back to `float3` so that the model satisfies the `IModel<float2, float3>` constraint.

This can be very flexible, and easily allows for mixed precision. For example, you might experience precision issues evaluating `Exp` in 16 bit, and might choose to convert to float earlier:
```
model = nn.ModelChain(
    # [ ... ]
    nn.Convert.to_float(),
    nn.Exp(),
)
```
`nn.Convert` supports `.to_half()`, `.to_float()`, `.to_double()` and `.to_precision(dtype: Real)`, where the `nn.Real` enum can be used to specify data types programmatically.

`Convert` can also be used to cast between different array types. In Slang, we have a number of array-like types (i.e. types that can be indexed with `[]`) at our disposal: Plain arrays (e.g. `float[10]`), vectors (e.g. `float4`) or cooperative vectors (e.g. `CoopVec<float, 16>`). In `trainTexture`, we expect to pass in a vector and get out a vector for convenience. However, `FrequencyEncoding` is currently only implemented for plain arrays, and `model.initialize` would give us an error complaining that the types are not compatible.

We can convert between array types using `nn.Convert.to_array()`, `.to_vector()`, `.to_coopvec()` and `.to_array_kind(kind: ArrayKind)`. The full model initialization in `model.py` actually looks like this:
```
if "cooperative-vector" in device.features:
    print("Cooperative vector enabled!")
    mlp_input = nn.ArrayKind.coopvec
    mlp_precision = nn.Real.half
else:
    print("Device does not support cooperative vector. Sample will run, but it will be slow")
    mlp_input = nn.ArrayKind.array
    mlp_precision = nn.Real.float

# Set up model architecture:
model = nn.ModelChain(
    nn.Convert.to_array(),
    nn.FrequencyEncoding(5),
    nn.Convert.to_precision(mlp_precision),
    nn.Convert.to_array_kind(mlp_input),
    nn.LinearLayer(nn.Auto, 64),
    nn.LeakyReLU(),
    nn.LinearLayer(nn.Auto, 64),
    nn.LeakyReLU(),
    nn.LinearLayer(nn.Auto, 64),
    nn.LeakyReLU(),
    nn.LinearLayer(nn.Auto, 3),
    nn.Convert.to_vector(),
    nn.Convert.to_float(),
    nn.Exp(),
)
```
This first converts the `float2` input to an array compatible with `FrequencyEncoding`. If hardware support for cooperative vector is detected, the output of the encoding is then converted to a half precision cooperative vector for fast execution of the neural network in the middle. Finally, the output is converted back to a `float3`.

#### Automatic Parameters

Because Slang is strongly typed, each component of the model needs to specify its exact input and output type. For example, the Slang implementation of the `LeakyReLU` activation we used looks like this:
```
struct LeakyReLU<T : IReal, int Width>
```
The generic parameters `T` and `Width` specify the input precision and input width, and are required so that Slang can generate concrete code. They are also needed so `LeakyReLU` can communicate that it implements (among other things) `IModel<T[Width], T[Width]>`, i.e. takes an array and returns an array of equal width and element type.

This means that in Python, the call `nn.LeakyReLU()` is underspecified: There's no mention of `T` or `Width`! This is because they are automatically inferred for us. If we look at the `LeakyReLU` constructor, we see
```
def __init__([...], width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
```
Here, `width` and `dtype` are marked as `AutoSettable`. This means we can either provide a concrete value---like `int` for `width`---or pass the value `nn.Auto`, which means they will be automatically inferred when we call `model.initialize`. `AutoSettable` is just a Python type hint. It doesn't change any of the functionality, but helps communicate to the end user this parameter accepts `Auto`, and helps with IDE code analysis.

Generally, parameters are marked as `AutoSettable` when they can be uniquely inferred from the input type. For example, if the `forward` method of `nn.LinearLayer` returns a `float[64]`, there is only one allowable setting of the following `LeakyReLU` component that makes it compatible (i.e. `width=64` and `dtype=Real.float` ). In that case, they can be omitted or set to `Auto`.

This makes it much more convenient to set up model architectures. If we didn't have `Auto`, the full model definition would look like this:
```
model = nn.ModelChain(
    nn.ConvertArrayKind(width=2, dtype=Real.float, to_kind=ArrayKind.array),
    nn.FrequencyEncoding(input_width=2, dtype=Real.float, num_octaves=5),
    nn.ConvertArrayPrecision(from_dtype=Real.float, to_dtype=Real.half, width=20, kind=ArrayKind.array),
    nn.ConvertArrayKind(width=20, dtype=Real.half, to_kind=ArrayKind.coopvec),
    nn.LinearLayer(num_inputs=20, num_outputs=64, dtype=Real.half, use_coopvec=True),
    nn.LeakyReLU(width=64, dtype=Real.half),
    nn.LinearLayer(num_inputs=64, num_outputs=64, dtype=Real.half, use_coopvec=True),
    nn.LeakyReLU(width=64, dtype=Real.half),
    nn.LinearLayer(num_inputs=64, num_outputs=64, dtype=Real.half, use_coopvec=True),
    nn.LeakyReLU(width=64, dtype=Real.half),
    nn.LinearLayer(num_inputs=64, num_outputs=3, dtype=Real.half, use_coopvec=True),
    nn.ConvertArrayKind(width=3, dtype=Real.half, to_kind=ArrayKind.vector),
    nn.ConvertArrayPrecision(from_dtype=Real.half, to_dtype=Real.float, width=3, kind=ArrayKind.vector),
    nn.Exp(width=3, dtype=Real.half),
)
```
Needless to say, this is a lot more unwieldy. Beyond being harder to read, it also violates the principle of "don't repeat yourself": For example, the network precision or the width of the output is repeated in many different places, and if we wanted to change these values, we would need to modify lots of code at the same time. This runs a much higher risk of components getting out of sync and goes against our goal of allowing rapid experimentation with different architectures. This is especially true when parameters aren't immediately obvious: For example, the number of outputs from the `FrequencyEncoding` depends on the input width and number of octaves via `num_outputs = 2 * input_width * num_octaves`. If we had to manually specify the width of the layers that follow, we would have to compute this value ahead of time somehow, and the resulting code would become increasingly complex. It's much easier to have the library infer these values for us.

This explains why `model.initialize(module, "float2")` needed to know the input type (`"float2"`) to the model: This allows it to infer every `Auto` parameter we omitted.

For convenience, `initialize` allows a wide range of values for the `input_type` parameter:
- A `str` specifying a slang type name, e.g. `model.initialize(module, "float2")`
- A `RealArray` instance specifying an array type, e.g. `model.initialize(module, nn.RealArray(nn.ArrayKind.vector, nn.Real.float, 2))`
- A slang type reflection (`slangpy.SlangType` or `slangpy.Struct`), e.g. `model.initialize(module, module.float2)`

*Fully specified components:* In some cases, the `input_type` parameter is optional. Most components in the library do know their input type if they are fully specified. For example,
```
nn.LinearLayer(num_inputs=20, num_outputs=64, dtype=Real.float, use_coopvec=False)
```
knows it expects a `float[20]` as input, without additional help from `.initialize(module, "float[20]")`. If the first component of the model is fully specified, then `model.initialize(module)` is sufficient.

### Adding New Components

It is easy to extend the library with new components to add new functionality. New components that you add will be compatible with the existing ones, and e.g. can be passed to `ModelChain` and will work with the `Auto` mechanism.

As an example, let's walk through adding a basic `ToGrayscale` model to the neural texture that converts colors to grayscale.

#### Implementing the Slang Component

Let's start by writing the slang code to do the grayscale conversion:
```
struct ToGrayscale : IModel<float3, float3>
{
    // ...
}
```
We've added a new class `ToGrayscale`, which takes a `float3` (RGB) and returns a `float3`, with the components replaced by the grayscale of the input. This is specified by `: IModel<float3, float3>`, which tells Slang we intend to implement `IModel` with the given input- and output types.

Now we add an implementation of `forward` that does the actual grayscale computation:
```
    [BackwardDifferentiable]
    float3 forward(float3 x)
    {
        float gray = (x[0] + x[1] + x[2]) / 3.0f;
        return float3(gray);
    }
```
This method takes an RGB `float3`, averages its values, and returns a new `float3` set to the average.

#### Exposing it in Python

Next, we need to expose the Slang code to Python to integrate it with the `neuralnetworks` library. The minimum amount of code to do this is to derive from the Python class `IModel` and implement the `type_name` property:
```
class ToGrayscale(nn.IModel):
    @property
    def type_name(self) -> str:
        return "ToGrayscale"
```
The `type_name` property returns the name of the corresponding Slang type that implements the component, and has to derive from the `IModel` interface in Slang.

This is all that is needed to get a basic component up and running. When we use this with the `neuralnetworks` module, it will use reflection to look up the `struct ToGrayscale` type we defined in Slang, make sure it conforms to `IModel` and retrieves the input/output types of `forward` to do type checking and resolution of `Auto` values. We can now add it to our model and see what happens:
```
model = nn.ModelChain(
    # [ ... ]
    nn.Exp(),
    ToGrayscale(),
)
```
If everything is implemented correctly, we should now get this output when running `main.py`:

![Grayscale picture of a cat represented by a neural network](./example-output-grayscale.jpg)

#### Passing Parameters

We can easily pass parameters from Python to Slang components. As an example, we'll change `ToGrayscale` so that it supports weighting the color channels, e.g. to compute perceptual grayscale.

First, we'll add a `channelWeights` field to the Slang definition:
```
struct ToGrayscale : IModel<float3, float3>
{
    float3 channelWeights;

    [BackwardDifferentiable]
    float3 forward(float3 x)
    {
        float gray = dot(channelWeights, x);
        return float3(gray);
    }
}
```
Next, we'll add the parameter to the Python class:
```
class ToGrayscale(nn.IModel):
    def __init__(self, weights: list[float]):
        super().__init__()
        self.weights = weights

    #[ ... ]

# [ ... ]
model = nn.ModelChain(
    # [ ... ]
    nn.Exp(),
    ToGrayscale([0.2126, 0.7152, 0.0722]), # sRGB luminance
)
```

Finally, we'll need a way of passing the `weights` field from Python to Slang. We can do this by implementing `get_this()`:
```
class ToGrayscale(nn.IModel):
    # [ ... ]

    def get_this(self):
        return {
            "channelWeights": self.weights,
            "_type": self.type_name
        }
```
In SlangPy, `get_this` is a special method that is called by SlangPy when we try to pass a custom Python object to a Slang function. Because `class ToGrayscale` is a Python class we invented, SlangPy has no way of knowing what is supposed to happen when we pass it to a Slang function. In that case, it will check if there is a `get_this` method on the object that can turn it into an object SlangPy can understand.

In this case, we return a Python `dict`, which SlangPy is happy to pass to Slang. `"channelWeights"` matches up with the field of the same name of our Slang `struct`, and `"_type"` is a special member that tells SlangPy what specific Slang type it should attempt to map this `dict` to; that allows SlangPy to resolve e.g. generics or function overloads.

`nn.IModel` provides a default implementation of `get_this`, which simply announces the type it should map to:
```
class IModel:
    # [ ... ]
    def get_this(self) -> dict[str, Any]:
        return {'_type': self.type_name}
```
This is why `ToGrayscale` worked before, even though we didn't provide our own `get_this`.

#### Making it Generic

The `ToGrayscale` type we defined only works with `float3` so far. Let's now make it generic, so it supports any input precision or element count:

```
struct ToGrayscale<T : IReal, int N> : IModel<vector<T, N>, vector<T, N>>
{
    vector<T, N> channelWeights;

    [BackwardDifferentiable]
    OutputType forward(InputType x)
    {
        T gray = dot(channelWeights, x);
        return OutputType(gray);
    }
}
```
`ToGrayscale<T, N>` now accepts a vector with `N` elements, with element type `T`. With `ToGrayscale<float, 3>`, we get the same behavior as before. Here, we made use of the typedefs `InputType` and `OutputType`: These are defined by `IModel` for convenience, so we don't have to repeat `vector<T, N>` multiple times. `IReal` is an interface supplied by the `NeuralNetworks` module, and is used for requesting a scalar, differentiable type, currently implemented by `half`, `float` and `double`.

As a last step, we need to change our Python class to take care of the generic parameters. We can do this by adding new `dtype` and `width` parameters to the constructor:
```
class ToGrayscale(nn.IModel):
    def __init__(self, weights: list[float], dtype: nn.Real, width: int):
        super().__init__()
        self.weights = weights
        self.dtype = dtype
        self.width = width
```
...and change `type_name` to reference the appropriate generic specialization of the `struct ToGrayscale<T, N>`:
```
    @property
    def type_name(self) -> str:
        return f"ToGrayscale<{self.dtype}, {self.width}>"
```
`nn.Real` is an enum that maps to the types that implement `IReal` in slang. Its `__str__()` method returns the appropriate slang type, which is handy for formatting a Slang-compatible string in `type_name` like shown above.

Finally, we need to pass the appropriate arguments when setting up the model:
```
model = nn.ModelChain(
    # [ ... ]
    ToGrayscale([0.2126, 0.7152, 0.0722], nn.Real.float, 3), # sRGB luminance, accepts float3
)
```

#### Supporting Auto Parameters

After the previous step, `ToGrayscale` is generic, but it has unfortunately become very verbose. We can fix this by accepting `Auto` for the new `dtype` and `width` arguments, and resolving them automatically if they're not supplied by the user.

First, let's change the constructor of the Python type to accept `Auto`:
```
class ToGrayscale(nn.IModel):
    def __init__(self, weights: list[float], dtype: nn.AutoSettable[nn.Real] = nn.Auto, width: nn.AutoSettable[int] = nn.Auto):
        super().__init__()
        self.weights = weights
        self._dtype = dtype
        self._width = width
```
Here, `AutoSettable[T]` is a type hint for Python telling it that the parameter accepts either `T` or `Auto`. This doesn't affect any of the functionality, but is useful for end users to tell them a parameter can be set to `Auto`, and also adds useful info for Python IDEs that do code analysis (like VS Code).

Because `dtype` and `width` are not resolved yet and could contain `Auto` instead of a concrete value, a useful pattern is to assign them to an interim field `self._dtype` instead of `self.dtype` directly. We'll set the `self.dtype` field later when we resolve the types.

Finally, let's add a method to figure out concrete values for `dtype` and `width`. The `IModel` base class defines a `model_init` method, which is called once during `model.initialize` and supplies the type that will be passed to this model's `forward` method in Slang. This gives us enough information to resolve `Auto`s:
```
class ToGrayscale(nn.IModel):
    # [ ... ]
    def model_init(self, module: Module, input_type: SlangType):
        input_array = nn.RealArray.from_slangtype(input_type)
        self.width = nn.resolve_auto(self._width, input_array.length)
        self.dtype = nn.resolve_auto(self._dtype, input_array.dtype)
```
SlangPy supplies the `SlangType` class that represents a type in Slang retrieved from reflection. We could do all kinds of type inspection on it, but the most common use case is that our model works with array-like things and just wants to know those properties. To make that easy, the `neuralnetworks` library provides a `RealArray` type with `length: int`, `dtype: Real` and `kind: ArrayKind` fields for simple inspection. `RealArray.from_slangtype(input_type)` will try to parse the input type as a 1D array, and throws an error if the input type is not compatible.

With this, we can figure out concrete values for `self.width` and `self.dtype` using `nn.resolve_auto`. The helper `resolve_auto(a, b)` simply returns `a` if it is a concrete value, and returns `b` if `a` is set to `Auto`.

Now we have a generic `ToGrayscale`, but can still write this kind of code:
```
model = nn.ModelChain(
    # [ ... ]
    ToGrayscale([0.2126, 0.7152, 0.0722]), # sRGB luminance, accepts float3
)
```
which automatically resolves the generic parameters for us.

You're free to make any parameter `AutoSettable` if it makes sense in your application. For example, you could declare the weights parameter as `weights: nn.AutoSettable[list[float]] = nn.Auto`, and resolve it to a default value in `model_init`:
```
class ToGrayscale(nn.IModel):
    # [ ... ]
    def model_init(self, module: Module, input_type: SlangType):
        input_array = nn.RealArray.from_slangtype(input_type)
        self.width = nn.resolve_auto(self._width, input_array.length)
        self.dtype = nn.resolve_auto(self._dtype, input_array.dtype)

        if self._weights is nn.Auto:
            # No weights supplied? -> Generate weights for a simple average
            self.weights = []
            for i in range(self.width):
                self.weights.append(1.0 / self.width)
        else:
            # Weights supplied? Double check they agree with the resolved width
            if len(self._weights) != self.width:
                self.model_error(f"Expected {self.width} weights; received {len(self._weights)} instead")
            self.weights = self._weights
```
The `model_error` method is a useful method to throw an exception when the model is not consistent, and will supply extra information about the model and the failing component to help the user with debugging.

#### Where to go from here

If you're interested in implementing more complex components, I recommend reading the doc strings of the `IModel` base class for other useful methods. You might also find it useful to study types like `LinearLayer`, which allocate trainable parameters and select different Slang implementations based on the input type, or `ModelChain`, which creates and initializes nested components.
