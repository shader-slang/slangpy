
# Want-to-have Features for `kernelfunctions` (from a CUDA/Torch programmer's perspective)

##  `vmap` : Vectorize a scalar function and convert to a kernel

Here's a simple slang file with nice scalar functions with simple types
```csharp
// simple_two_funcs.slang
public float sqr(float x)
{
    return x * x;
}

public float sum(float x, float y)
{
    return x + y;
}
```


```python
# simple_two_funcs.py
import kernelfunctions as kf
dev = kf.device("cuda") // Good idea to put module loading behind a device object to take care of capability stuff later on.

m = dev.loadModule('simple_two_funcs.slang')
```

- Calling this method normally results in a scalar call:
	```python
	print(m.sqr(1)) # 1 (scalar call)
	```

- Vectorization happens using 'vector map' semantics inspired by JAX. It might be helpful to be explicit about what dimensions are added and how. See [jax.vmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) for more.
- Broadly speaking, `vmap` simultaneously adds 1 dimension to the inputs and the outputs:
	```python
  # m.sqr is a scalar method with a scalar output. Represented by "m.sqr := float -> float"
  # 
	# Vectorize with vmap: in_axes=(0,) means that the new dimension is added 
	# to the 'front' of the existing dimensions.
	# 
	# vsqr can be represented as "vsqr<N> := float[N] -> float[N]"
	vsqr = kf.vmap(m.sqr, in_axes=(0,), out_axes=0)
	print(vsqr([1, 2, 3])) # numpy.ndarray [1, 4, 9] (vector call)
	```
- Note that `vmap` allows direct inference of output size from input dimensions, since we have an explicit mapping between input and output dimensions.
- `vmap` can be repeatedly invoked to create kernels that work with multi-dimensional inputs and outputs.
- `vmap` can be used to broadcast one parameter while treating the other one as constant. For example, `m.sum` has multiple inputs:
	```python
	
	# m.sum := (float, float) -> float
	
	# Passing "None" will avoid vectorizing the corresponding argument
	# vsum<N> := (float[N], (float)) -> float[N]
	vsum = kf.vmap(m.sum, in_axes=(0, None), out_axes=0)

	# First argument is 'uniform'. Second argument is 'constant' (broadcasted). 
	print(vsum((1, 2, 3, 4), 10))

	# Can also map this further to create a method that is 2D in the first arg, and 1D in the second arg.
	# vvsum<M, N> := (float[M, N], float[M]) -> (float[M, N])
	vvsum = kf.vmap(vsum, in_axes=(0, 0), out_axes=0)

	# Alternatively, if you want the second axis of the first arg to be 'M', you can specify '1' instead of '0' which places the new dimension in position 1
	# vvsum_T<M, N> := (float [N, M], float[M]) -> (float[M, N])
	vvsum_T = kf.vmap(vsum, in_axes=(1, 0), out_axes=0)
	```

- `vmap` is intended to be a very light-weight wrapper that generates a kernel around a function & replaces scalar inputs with buffer types coded with the appropriate number of dimensions (as generic arguments)

- A slang kernel has not been 'compiled' yet upon using `vmap`. We just record the dimensions we need for the final kernel and compile the kernel upon being called.
- **Under-the-hood**: upon invocation, we create a new function that translates a method into a new method that operates on buffer objects with strongly-typed sizes. See next section (on `IBuffer`) for more.

## Universal buffer interface `IBuffer<T, M: Mode, let A : int, let B: int ...>`
- Slang has recently added *variadic generics* as well as *generically-defined interfaces*. This allows us to define a strongly-typed N-dimensional buffer interface.
- Kernels can be defined using strongly-typed buffers. Example:
	```csharp
	void outer_product_kernel<let N : int, let M: int>(
		IBuffer<float, In, N> vec_a,
		IBuffer<float, In, M> vec_b,
		IBuffer<float, Out, N, M> output)
	{
		 uint2 thread_id = get_thread_id();
		 output[thread_id] = vec_a[thread_id.x] * vec_b[thread_id.y];
	}
	```
- Before dispatching, generic parameters are unified and the function is transformed. `In` buffers become inputs, `Out` buffers have their types deduced (using Slang reflection) and turned into outputs from the python object. Example:
	```python
	m = dev.loadModule("outer_product.slang")

	# result is a python 'namedtuple' of slang objects along with their output parameter names.
	result = m.outer_product_kernel(vec_a = torch.rand((100,)), vec_b = torch.rand((50,)))
	
	# result has one object ('output'). If the method had more `Out` tensors, each one becomes a member of the namedtuple
	mat_ab = result.output
	print(mat_ab.shape) # shape deduced from inferred generic parameters: (100,50)
	```
-  `.of` can be used to explicitly specialize functions and methods before calling them, though this is not necesary. Slang's type system can deduce N and M for the entire function from the `IBuffer` specialization arguments. (See later section for more info about `.of()`)
	```python
	# Works. N and M are consistent.
	m.outer_product_kernel.of(N=100, M=50)(vec_a = torch.rand((100,)), vec_b = torch.rand((50,)))

	# Errors. N and M inconsistent.
	m.outer_product_kernel.of(N=500, M=500)(vec_a = torch.rand((100,)), vec_b = torch.rand((50,)))
	```
- `.vmap`'s implementation simply wraps argument types into buffers, and adds a dimension to args that are already buffers. Example: 
	```csharp
	float f(float x, float y) { 
		return x + y;
	}
	
	// mapped_f = vmap(f, in_axes=(0, 0), out_axes=0) 
  // mapped_f when called results in the following wrapper being synthesized on the fly.
	void synthesized_vmapped_f<let Dim1: int>(
		IBuffer<float, In, Dim1> x, 
		IBuffer<float, In, Dim1> y,
		IBuffer<float, Out, Dim1> output)
	{
		 uint thread_id = get_thread_id<1>();
		 output[thread_id] = f(x[thread_id], y[thread_id]);
	}
  
  // Call vmap again on the result simply adds another dimension. 
  // Let's say we map it again, but this time 
  // mapped_2d_f = vmap(mapped_f, in_axes=(0, None), out_axes=0)
  //
  // This is the synthesized wrapper signature:
  void synthesized_vmapped_f_2<let Dim1: int, let Dim2: int>(
	  IBuffer<float, In, Dim1, Dim2> x, 
	  IBuffer<float, In, Dim1> y,
	  IBuffer<float, Out, Dim1, Dim2> output)
	{
		 vector<uint, 2> thread_id = get_thread_id<2>();
		 output[thread_id] = f(x[thread_id], y[thread_id.x]);
	}
	```
- `IBuffer`'s are specialized with the concrete buffer types from the type wrappers. Since `kf.wrap()` is device specific, it can choose to create a `StructuredBuffer` or `TensorView` or a user provided buffer type. Example implementations:
	```csharp
	struct StructuredBuffer2DImpl<T, let N: int, let M: int> : IBuffer<T, In, N, M> { 
		StructuredBuffer2D<T> buffer;
	}

    struct RWStructuredBufferImpl<T, let N: int> : IBuffer<T, Out, N> { 
		RWStructuredBuffer<T> buffer;
	}
	```
	The wrapper type can be reflected to get the size and type, which provides all the information we need for the allocation.
- Important note: we want to rely on Slang's generics for checking the sizes and types for consistency (and producing any errors along the way), instead of trying to do it in our framework.

## `fuse`: Chain Slang methods

- `fuse` allows Slang functions to be chained together to create a single kernel that calls all of them together
	```python
	# Create a new function that calls sqr twice
	composed = lambda x : m.sqr(m.sqr(x))

	# Fuse into a single function
	composed = kf.fuse(composed)
	```
	Under the hood, this is qquivalent to internally emitting the set of functions as a single slang expression:
	```csharp
	// Synthesized:
	import simple_two_funcs;
	float anonymous_generated_func(float x)
	{
		return sqr(sqr(x));
	}
	```
	where the types are inferred from the function input types. 

- We disallow overloaded methods for simplicity. Otherwise, this creates a tricky problem where we have to resolve both the type and the overload simultaneously, and since the type could be a coerced version of a python type, this can make the overload resolution very complicated (and needs to include the python code in-the-loop).

- `fuse`d methods can be used with `vmap` to create a vectorized kernel out of the composed method.
	```python
	# Then create a vectorized kernel out of the composed function
	vcomposed = kf.vmap(composed, in_axes=(0,), out_axes=0)
	print(vcomposed([1, 2, 3])) # [4, 16, 36] (vector call)
	```
- `vmap` **cannot** be used inside a fuse call. `vmap` turns Slang functions into a kernel and `fuse` can only be used to chain together Slang functions.
- Since `fuse` and `vmap` are higher-order functions, they can be invoked as python decorators. This is the intended approach to use `vmap` and `fuse`
    ```python
    # fusion and mapping operators are executed in succession.
    @kf.vmap(in_axes=(0, 0), out_axes=0)
    @kf.vmap(in_axes=(0, 0), out_axes=0)
    @kf.fuse
    def hyp_sqr_2d(x, y):
        return m.sum(m.sqr(x), m.sqr(y))
    
    # hyp_sqr_2d<N, M> := (float[N, M], float[N, M]) -> float[N, M]
    ```

## Direct access to `public` structs and their methods
This is a feature already partially present in slangtorch (structs have their fields exposed, but no methods)

- Here's a more complex example of a Slang-based rasterizer that uses some Slang types and buffer types:
	```csharp
	public struct Triangle2D
	{
		float2 v0;
		float2 v1;
		float2 v2;
		float4 color;

		float2 get_barycentric(float2 pos) { /* ... */ }
		float4 get_color_at(float2 pos) { /* ... */ }
		bool bbox_test(float2 pos) { /* ... */ }
	};

	public struct Rasterizer2D
	{
		Triangle2D primitive;

		float4 rasterize_pixel(uint2 pixel_id)
		{
			float2 pos = float2(pixel_id.x, pixel_id.y);
			float4 color = float4(0, 0, 0, 0);
			return triangle.get_color_at(pixel_id);
		}
	}
	```

	Now we can create a method from python to call a member method
	```python
	import kernelfunctions as kf
	m = dev.loadModule("rasterizer2d.slang")

	def my_func(triangle : m.Triangle2D, pix_id : m.float2): # Note that type annotations are optional in python, but it helps to follow the code (and all these types are exposed publicly because they're used in the code)
		# Create a rasterizer object 
		rasterizer2D = m.Rasterizer2D(triangle=triangle)
		return rasterizer2D.rasterize_pixel(pix_id)

	```

	Then create a fused kernel
	```python
	callable_func = kf.fuse(my_func)
	```

	This will simply make an anonymous slang function of this form:
	```csharp
	float anonymous_func_1(Triangle2D inp_1, float2 inp_2){ Rasterizer2D(inp_1).rasterize_pixel(inp_2); }
	```
	Under-the-hood, it should be straightforward to implement, since all we are doing is tracing and building up a slang expression. 

	Then, we can `vmap` this twice to map it in 2D and then call it to create a kernel:
	```python
	# "None" says "do not add a dimension to this input", so the triangles buffer is not transformed further
	cf1d = kf.vmap(callable_func, in_axes=(None, 0), out_axes=0)
	cf2d = kf.vmap(cf1d, in_axes=(None, 0), out_axes=0)

	# Type constructors can be invoked as kernel functions (more on this in a bit)
	triangle = m.Triangle2D(v0=(0, 10), v1=(10, 0), v2=(10, 10), color=(1,1,0,1))

    # Create an MxNx2 grid of pixel indices
	xx = torch.linspace(0, 50, N=50)
	yy = torch.linspace(0, 50, N=50)
	x, y = torch.meshgrid(xx, yy)
	xy = torch.stack([x, y], axis=-1) 

	cf2d(triangle, xy) # Launch kernel.. the launch parameters can be deduced in a direct manner because each vmap adds a corresponding dimension to the input.

	```

	The generated kernel should look something like this:
	```csharp
	__generic<let Dim1: int, let Dim2: int>
	void anonymous_kernel_1(
        Triangle inp_1,
        IBuffer<int, In, Dim1, Dim2> buf_1,
        IBuffer<float2, Out, Dim1, Dim2> buf_out)
	{ 
		uint3 tid = get_platform_independent_global_thread_id();
        if (tid.x >= Dim1 || tid.y >= Dim2)
			buf_out[tid] = anonymous_func_1(inp_1, buf_1[tid]);
	}
	```

## `wrap`/`unwrap`: Transfer python type (and associated data) into a Slang type and vice-versa

- `wrap` creates an appropriate wrapper object out of the provided data and Slang type.
- The Slang type need not be fully specialized. wrapper object is responsible for inferring unknowns.
- Since `wrap` must be called on an active device object, we have device specific information if necessary.
- `wrap` is called automatically by the framework, when trying to pass a non-Slang type into a kernel. `.wrap` is invoked automatically with the argument type info reflected from the function's signature
- `wrap` simply calls a registered 'handler' for that particular Slang type. A user can register their own handler for their Slang type if necessary.
- `unwrap` works in the same way, but converts Slang type to a target python type (this could be a tuple or namedtuple or ndarray or something else if the user supplies their own unwrapper)

Example:
```csharp
float2 f(float2 xy)
{
	return xy.yx;
}
```

```python
import kernelfunctions as kf
dev = kf.device("cuda")
m = dev.loadModule("rasterizer2D")
	
# invec's type is a wrapper object with the given data, and cannot be used within python anymore.
invec = kf.wrap(m.float2, (5, 2))

# You can unwrap it back to a python type
outvec = kf.unwrap(tuple, invec)

# outvec is automatically 'unwrapped' from float2 back to a tuple (2, 5)
outvec = m.f(invec)

# vectorize the method 
# f : float2 -> float2
# mapped_f<N> : float2[N] -> float2[N]
#
mapped_f = kf.vmap(m.f, in_axes=(0,), out_axes=0)

# mapped_f's input is an `IBuffer<N, In, float2>` which is an abstract buffer that can be satisfied with a D3D `StructuredBuffer<float2>` or a CUDA
# `TensorView<float2>`. We can invoke the handler for `IBuffer` to let it decide based on the device type.
# This call is invoked _automatically_ under the hood (but the user can invoke it to be explicit if they wish to)
#
buffer = kf.wrap(m.IBuffer.of(float2), [(1, 2), (4, 6), (10, 9), (2, 6), (10, 10)])

# output has been unwrapped automatically into an ndarray for each python use. (location of the ndarray memory is based on the device object)
output = mapped_f(buffer)

# Sometimes unwrapping automatically is unnecessary (maybe you simply want to pass it to the next kernel without inspecting the contents), so you can pass `no_unwrap=True` to keep the original Slang type.
output_of_slang_buffer_type = mapped_f(buffer, no_unwrap=True)

```

### Registering a wrapper/unwrapper
- A wrapper class needs 4 methods: 
	-  constructor `__init__(device, slang_type, python_object)` where `device` is the current active device object and `slang_type` is the provided slang type. Can throw a `CastError` if the data is not in an acceptable format.
	-  destructor `__del__()` : release any resources.
	-  `slang_type() -> SlangType` : return the fully specialized slang type. The type originally provided to the constructor could have unspecialized arguments (for example `IBuffer<float2, In, M, N>`, where M and N are unknowns). Inspecting the python object can fill in the blanks.  If the full concrete type cannot be deduced, a `CastError` should be thrown.
	- `bind(device, slang_var)`: Given a variable reflection, perform any binding actions. The framework handles emitting variables & resources to the global scope, if necessary, before providing the variable reflection. 
-  Wrapper classes can be registered using something like `sc.register_wrapper(slang_type, wrapper_class)` and `sc.register_unwrapper`. Most standard types will have predefined wrapper classes, but the user can add a wrapper class to an existing type if they wish. We can run the wrapper classes in reverse order of registration until one succeeds, to enable user overrides to go first.

## `.of()/.arg()`: Setting & getting specialization arguments
Slang types can (and often are) generically defined. It is helpful to be able to form references to generic types with specialization arguments.
As an example `StructuredBuffer<T>` is a buffer that is parameterized on `T`
- All Slang types have an `.of()` and `.arg()` method that can be used to form specialized references to types and query specialization args from types. Example:
	```python
	m = loadModule('rasterizer2D.slang')
	sbufferFloat = m.StructuredBuffer.of(m.float)

	# Can use keyword args (for partial specialization)
	sbufferFloat = m.StructuredBuffer.of(T = m.float)

	# Retreive a specialization arg
	assert(sbufferFloat.arg("T") == m.float)
	```

- References to the same types with the same specialization args will be logically equal and can be directly compared. 
- `is_subtype(slang_super_type)` triggers Slang's sub-type check and can be used to verify conformance to interfaces. Example:
	```python
  # Note that IBuffer is not an actual interface currently (this is just for demonstration)
	m.StructuredBuffer.is_subtype(m.IBuffer)

	assert(m.float.is_subtype(m.IDifferentiable) == True)
	```

## Type constructors can be invoked as kernel functions
- Type constructors behave the same way as standard functions. For example, from the `Triangle2D` sample above:
	```python
	# Invokes the default constructor
	triangle = m.Triangle2D(v0=(0, 10), v1=(10, 0), v2=(10, 10), color=(1,1,0,1))
	```
	is equivalent to invoking a brace initialization constructor in Slang like this:
	```csharp
	Triangle2D anonymous_func(float2 v0, float2 v1, float2 v2, float4 color) { return Triangle2D{v0, v1, v2, color}; }
	```
	where the ordering is automatically determined by reflecting the field names of the type being constructed, and the python tuples are coerced into their slang-types through the `wrap` mechanism

- Custom constructor definitions `__init(...) { }` can also be invoked, but not via keyword arguments (very confusing to allow both keyword args and overloads at the same time). If keyword arguments are present, we assume it is brace-style initialization.

- Unlike regular function calls, `.unwrap` is **not** automatically called on the result of a constructor call. The result is still a slang type that can be passed as-is to another kernel call.

- Like regular functions & methods, constructors can be `vmap`'d. This is handy to create a buffer of objects from a set of tensors:
	```python
	# multi_triangle_constructor<N> := (float2[N], float2[N], float2[N], float4[N]) -> Triangle2D[N]
	multi_triangle_constructor = kf.vmap(m.Triangle2D, in_axes=(0, 0, 0, 0), out_axes=(0))

	N = 200
	triangles = multi_triangle_constructor(
		v0=torch.rand((N,2)),
		v1=torch.rand((N,2)),
		v2=torch.rand((N,2)),
		color=torch.rand((N,4)))

	# On D3D, triangles will be a `StructuredBuffer<Triangle>`
	# On CUDA, triangles will be a `TensorView<Triangle>`
	#
	# All buffer types are a subtype of IBuffer
	#
	assert(triangles.type().is_subtype(m.IBuffer) == True)
	```
 
## Fusion with generics & interfaces
- `fuse()`'s biggest benefit is its ability to work seamlessly with Slang's generics and interfaces. 
Here's the same example above, but with `Triangle2D` replaced with `IRasterPrimitive2D`
	```csharp
	// rasterizer2D.slang
	interface IRasterPrimitive2D
	{
		 float4 get_color_at(float2 pix);
	}

	public struct Rasterizer2D<T: IRasterPrimitive2D>
	{
		T primitive;

		float4 rasterize_pixel(uint2 pixel_id)
		{
			float2 pos = float2(pixel_id.x + .5f, pixel_id.y + .5f);
			float4 color = float4(0, 0, 0, 0);
			return primitive.get_color_at(pixel_id);
		}
	}
	```

	And then a bunch of different files with primitive definitions:

	```csharp
	// circle.slang
	import rasterizer2D;
	public struct Circle2D : IRasterPrimitive2D
	{
		float2 o;
		float r;
		float4 color;

		float4 get_color_at(float2 pix) { return (r > length(pix - o)) ? color : float4(0.f); }
	}
	```
	```csharp
	// triangle.slang
	import rasterizer2D;
	public struct Triangle2D : IRasterPrimitive2D
	{
		float2 v[3];
		float4 color;

		float4 get_color_at(float2 pix) { /* ... */ }
	}
	```

- In python, we can compose these together to make one specialized shader program
	```python
	import kernelfunctions as kf
	dev = kf.device("cuda")
	m_rasterizer = dev.loadModule("rasterizer2D.slang")
	m_circle = dev.loadModule("circle.slang")
	m_triangle = dev.loadModule("triangle.slang")

	# Quick note: as long as modules are loaded with the same device object, common types are de-duplicated.
	# This means m_circle.float2 == m_rasterizer.float2

    # generic_rasterizer_2d<T: IRasterPrimitive2D, M, N> := (T, (float2[M, N])) -> (float4[M, N])
    @kf.vmap(in_axes=(None, 0), out_axes=0)
    @kf.vmap(in_axes=(None, 0), out_axes=0)
    @kf.fuse
	def generic_rasterizer_2d(primitive, pix_id):
		 return m_rasterizer.Rasterizer2D(primitive).rasterize_pixel(pix_id)

	xx, yy = torch.meshgrid(
			 torch.linspace(-50, 50, N=50),
			 torch.linspace(-50, 50, N=50))

	xys = torch.stack(xx, yy, axis=-1)

	circle_obj = m_circle.Circle2D(o=(-10, 10), r=10, color=(1.0, 0.0, 1.0, 1.0))
	triangle_obj = m_triangle.Triangle2D(v=[(-10, 10), (10, -10), (-10, -10)], color=(0.0, 1.0, 0.0, 1.0))

	# Use rasterizer with circle. 
	# Under-the-hood, we simply create a kernel that calls the fused method with the concrete type and let Slang's type system
	# handle the specialization.
	#
	image = generic_rasterizer_2d(circle_obj, xys)

	# Use rasterizer with triangle objects.
	image = generic_rasterizer_2d(triangle_obj, xys)
	```

