import slangpy as spy
from slangpy.slangpy import Shape
import pathlib
import numpy as np
from slangpy.core.calldata import set_dump_generated_shaders, set_dump_slang_intermediates
import os
logger = spy.Logger.get()
logger.level = spy.LogLevel.debug

# might be able to get rid of this
# Set up a callback to capture the shader
shader_source = []
def log_callback(level, msg, frequency):
    if "shader" in msg.lower():
        shader_source.append(msg)

# Enable shader dumping
set_dump_generated_shaders(True)
set_dump_slang_intermediates(True)

# Create an SGL device with the local folder for slangpy includes
device = spy.create_device(include_paths=[
        pathlib.Path(__file__).parent.absolute(),
        spy.SHADER_PATH,  # Add the SlangPy shader path
])

# Load the module
module = spy.Module.load_from_file(device, "call-id-test.slang")

# Clear the kernel cache to force regeneration
module.kernel_cache.clear()

# Can already do an N-dimensional run, this works
#results = module.test4D(spy.grid((2,2,2,2)), _result='numpy')
#print(results)

### Debugging ###
#Get the call data without _result parameter
call_data = module.test2D.call_group_shape(Shape((4,8))).debug_build_call_data(spy.grid((32,64)))
# Print the generated code
# Seems to only work when the shader changes?
print(call_data.code)  # This should contain the generated shader code
### End Debugging ###

# Manually inspecting the results of the below at the moment.

# Doing things this way uses the default 32x1x1 call group size
# 32x64 => y,x
results_default = module.test2D_thread(spy.grid((32,64)), spy.thread_id(), _result='numpy')
print(results_default)

# Expect this to spit out something like
"""
[DEBUG]   Call type: call
[DEBUG]   Call shape: [32, 64]
[DEBUG]   Call mode: prim
[DEBUG]   Strides: [64, 1]
[DEBUG]   Threads: 2048
[DEBUG]   Call grid shape: [32, 2]
[DEBUG]   Call grid strides: [2, 1]
[DEBUG]   Call group shape: [1, 32]
[DEBUG]   Call group strides: [32, 1]
[[[   0    0    0]
  [   1    0    1]
  [   2    0    2]
  ...
  [  61    0   61]
  [  62    0   62]
  [  63    0   63]]

 [[   0    1   64]
  [   1    1   65]
  [   2    1   66]
  ...
  [  61    1  125]
  [  62    1  126]
  [  63    1  127]]

...

 [[   0   31 1984]
  [   1   31 1985]
  [   2   31 1986]
  ...
  [  61   31 2045]
  [  62   31 2046]
  [  63   31 2047]]]
"""

# This is interpreted as module.test2D.call_group_shape(Shape((y,x)))(spy.grid((y,x)), _result='numpy')
results_call_group_shape = module.test2D_thread.call_group_shape(Shape((4,8)))(spy.grid((32,64)), spy.thread_id(), _result='numpy')
print(results_call_group_shape)

# Expect this to spit out something like
"""
[DEBUG]   Call type: call
[DEBUG]   Call shape: [32, 64]
[DEBUG]   Call mode: prim
[DEBUG]   Strides: [64, 1]
[DEBUG]   Threads: 2048
[DEBUG]   Call grid shape: [8, 8]
[DEBUG]   Call grid strides: [8, 1]
[DEBUG]   Call group shape: [4, 8]
[DEBUG]   Call group strides: [8, 1]
[[[   0    0    0]
  [   1    0    1]
  [   2    0    2]
  ...
  [  61    0  229]
  [  62    0  230]
  [  63    0  231]]

 [[   0    1    8]
  [   1    1    9]
  [   2    1   10]
  ...
  [  61    1  237]
  [  62    1  238]
  [  63    1  239]]

...

 [[   0   31 1816]
  [   1   31 1817]
  [   2   31 1818]
  ...
  [  61   31 2045]
  [  62   31 2046]
  [  63   31 2047]]]
"""

# Do we expect the call shape to be perfectly divisable by the group shape?
results_call_group_shape = module.test2D_thread(spy.grid((3,3)), spy.thread_id(), _result='numpy')
print(results_call_group_shape)

#results_call_group_shape_thread_id_info = module.test2D_thread_id_info.call_group_shape(Shape((4,8)))(spy.grid((32,64)), _result='numpy')
#
# hmm, slangPy's NDBuffer currently doesn't seem to support any multi-component element types. Things like:
# -call_ids_buf = spy.NDBuffer(device, "int2", (32, 64))
# -call_ids_buf = spy.NDBuffer(device, "int[2]", (32, 64))
#
# Seem to give an error like SystemError: nanobind::detail::nb_func_error_except(): exception could not be translated!
# For now, we'll get creative and use something like call_ids_buf = spy.NDBuffer(device, "int", (32, 64, 2))
#   -This also doesn't seem to work
#
#call_ids_buf = spy.NDBuffer(device, "int", (32, 64, 2))
#call_group_ids_buf = spy.NDBuffer(device, "uint", (32, 64, 2))
#call_group_thread_ids_buf = spy.NDBuffer(device, "uint", (32, 64, 2))
#flat_call_ids_buf = spy.NDBuffer(device, "uint", (32, 64))
#flat_call_group_ids_buf = spy.NDBuffer(device, "uint", (32, 64))
#flat_call_group_thread_ids_buf = spy.NDBuffer(device, "uint", (32, 64))
#thread_id_infos_buf = spy.NDBuffer(device, dtype=module.thread_id_info_2d, shape=(32, 64))
#module.test2D_thread_id_info.call_group_shape(Shape((4,8)))(spy.grid((32,64)), call_ids_buf, call_group_ids_buf, call_group_thread_ids_buf, flat_call_ids_buf, flat_call_group_ids_buf, flat_call_group_thread_ids_buf)
#module.test2D_thread_id_info(spy.grid((32,64)), call_ids_buf, call_group_ids_buf, call_group_thread_ids_buf, flat_call_ids_buf, flat_call_group_ids_buf, flat_call_group_thread_ids_buf)
#module.test2D_thread_id_info_test_struct(spy.grid((32,64)), thread_id_infos_buf)
#module.test2D_thread_id_info_test(spy.grid((32,64)), flat_call_ids_buf, flat_call_group_ids_buf, flat_call_group_thread_ids_buf)
#print(results_call_group_shape_thread_id_info)

# Convert to numpy
#call_ids = call_ids_buf.to_numpy()
#call_group_ids = call_group_ids_buf.to_numpy()
#call_group_thread_ids = call_group_thread_ids_buf.to_numpy()
#flat_call_ids = flat_call_ids_buf.to_numpy()
#flat_call_group_ids = flat_call_group_ids_buf.to_numpy()
#flat_call_group_thread_ids = flat_call_group_thread_ids_buf.to_numpy()

#result_cursor = thread_id_infos_buf.cursor()
#for x in range(16):
#    for y in range(16):
#        thread_id_info = result_cursor[x+y*16].read()
#        print(f"thread_id_info ({x},{y}): {thread_id_info}")

# Print first few elements
#print("Grid contents (first 5x5):")
#for y in range(min(5, call_ids.shape[0])):
#    for x in range(min(5, call_ids.shape[1])):
#        #print(f"call_id={call_ids[y,x]}, call_group_id={call_group_ids[y,x]}, call_group_thread_id={call_group_thread_ids[y,x]}, flat_call_id={call_ids[y,x]}, flat_call_group_id={flat_call_group_ids[y,x]}, flat_call_group_thread_id={flat_call_group_thread_ids[y,x]}")
#        print(f"call_id={call_ids[y,x]}, call_group_id={call_group_ids[y,x]}, call_group_thread_id={call_group_thread_ids[y,x]}, flat_call_id={call_ids[y,x]}, flat_call_group_id={flat_call_group_ids[y,x]}, flat_call_group_thread_id={flat_call_group_thread_ids[y,x]}")

### Below is in progress verification to automate checking of results ###

def default_call_id_to_1D_threadID(call_strides, call_id):
    expected_thread_id = 0
    for i in range(0, arr.size):
        expected_thread_id += call_id[i] * call_strides[i]
    return expected_thread_id

def shaped_call_id_to_1D_threadID(call_strides, call_id):
    # We want to do something like this here:
    # call-id[N] / group-dim[N] = group-id[N]
    # call-id[N] % group-dim[N] = group-thread-id[N]
    #
    # flat-group-id = group-id[0] * group-stride[0] + ....
    # flat-group-thread-id = group-thread-id[0] * group-thread-strid[0]
    #
    # flat-thread-id = Flat-group-id * 32 + flat-group-thread-id
    #
    # return flat-thread-id

    expected_thread_id = 0
    for i in range(0, arr.size):
        expected_thread_id += call_id[0] * call_strides[i]
    return expected_thread_id

# Should takes in an ND array where the base elements contain a 2D array containing 1D thread IDs
# that will be compared
# Need to pass some other stuff in here
def check_ids(thread_ids):
    # Get the shape of the array
    shape = thread_ids.shape
    
    # Create a list of indices, one for each dimension
    indices = [0] * len(shape)
    
    def traverse_recursive(dim=0):
        # Base case: if we've reached the last dimension, print the value
        if dim == len(shape) - 1:
            # Convert indices to tuple for indexing
            idx = tuple(indices)
            ids = result_ids[idx]

            # Compute the expected 

            # Do something with the ids
            #print(f"Index {idx}: {result_ids[idx]}")
            return
            
        # Recursively traverse each dimension
        for i in range(shape[dim]):
            indices[dim] = i
            traverse_recursive(dim + 1)
    
    traverse_recursive()
