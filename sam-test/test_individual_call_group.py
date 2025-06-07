#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import slangpy as spy
from slangpy.slangpy import Shape

device = spy.create_device()
module = spy.Module.load_from_file(device, 'test_call_group_simple.slang')

print('Testing individual call group functions with call_group_shape...')

try:
    # Test call_id with call_group_shape((2, 2))
    print('\n--- call_id with call_group_shape((2, 2)) on grid (4, 4) ---')
    call_id_result = module.test_call_id_2d.call_group_shape(Shape((2, 2)))(
        spy.grid((4, 4)),
        spy.call_id(),
        _result='numpy'
    )
    print(f'Call ID result shape: {call_id_result.shape}')
    for y in range(4):
        for x in range(4):
            print(f'  [{y},{x}]: call_id={call_id_result[y,x]}')

    # Test call_group_id with call_group_shape((2, 2))
    print('\n--- call_group_id with call_group_shape((2, 2)) on grid (4, 4) ---')
    call_group_id_result = module.test_call_group_id_2d.call_group_shape(Shape((2, 2)))(
        spy.grid((4, 4)),
        spy.call_group_id(),
        _result='numpy'
    )
    print(f'Call Group ID result shape: {call_group_id_result.shape}')
    for y in range(4):
        for x in range(4):
            print(f'  [{y},{x}]: call_group_id={call_group_id_result[y,x]}')

    # Test call_group_thread_id with call_group_shape((2, 2))
    print('\n--- call_group_thread_id with call_group_shape((2, 2)) on grid (4, 4) ---')
    call_group_thread_id_result = module.test_call_group_thread_id_2d.call_group_shape(Shape((2, 2)))(
        spy.grid((4, 4)),
        spy.call_group_thread_id(),
        _result='numpy'
    )
    print(f'Call Group Thread ID result shape: {call_group_thread_id_result.shape}')
    for y in range(4):
        for x in range(4):
            print(f'  [{y},{x}]: call_group_thread_id={call_group_thread_id_result[y,x]}')

    print('\n--- Analysis ---')
    print('Expected for call_group_shape((2, 2)) with grid (4, 4):')
    print('- 4 call groups total: [0,0], [0,1], [1,0], [1,1]')
    print('- Each group has 4 threads in 2x2 arrangement')
    print('- Call group thread IDs should range from [0,0] to [1,1] within each group')

    print('\n✓ All individual tests completed successfully!')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
