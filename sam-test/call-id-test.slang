import "slangpy";

// Currently a bit challenging to get a struct back. Not enough of
// a slang expert yet to know how to quickly and easily use this
// Also struggling to use generics here
/*
struct thread_id_info<let N : int>
{
    // A thread's id over the entire call shape
    // Might be better named as call_thread_id to align with the
    // convention used for call_group_thread_id
    int[N] call_id;

    // A group's id over the entire call shape
    uint[N] call_group_id;

    // A thread's id within the call group
    uint[N] call_group_thread_id;

    // 1-dimensional version of call_id
    uint flat_call_id;

    // 1-dimensional version of call_group_id
    uint flat_call_group_id;

    // 1-dimensional version of call_group_thread_id
    uint flat_call_group_thread_id;
}*/

// can't seem to use int2 or int[2] anywhere
struct thread_id_info_2d
{
    // A thread's id over the entire call shape
    // Might be better named as call_thread_id to align with the
    // convention used for call_group_thread_id
    // Call ID components (separate scalars instead of int[2])
    int call_id_x;
    int call_id_y;

    // A group's id over the entire call shape
    // Call group ID components (separate scalars instead of uint[2])
    uint call_group_id_x;
    uint call_group_id_y;

    // A thread's id within the call group
    // Call group thread ID components (separate scalars instead of uint[2])
    uint call_group_thread_id_x;
    uint call_group_thread_id_y;

    // 1-dimensional version of call_id
    uint flat_call_id;

    // 1-dimensional version of call_group_id
    uint flat_call_group_id;

    // 1-dimensional version of call_group_thread_id
    uint flat_call_group_thread_id;
}

// Just return call-id for now
uint2 test2D(uint2 val)
{
    //return uint2(3, 5);
    // test
    return val;
    //return call_data._group_dim;
}

// Return the call_id and thread_id
uint3 test2D_thread(uint2 call_id, uint3 thread_id)
{
    return uint3(call_id.x, call_id.y, thread_id.x);
}

// Return a bunch of thread id info
/*
// Might be able to make a generic version of this
// Note: There appears to be a current limitation in slangpy
// that prevents using something like RWNDBuffer<int[2], 2> call_ids,
// here
void test2D_thread_id_info(uint2 grid_cell,
                           RWNDBuffer<int, 3> call_ids,
                           RWNDBuffer<uint, 3> call_group_ids,
                           RWNDBuffer<uint, 3> call_group_thread_ids,
                           RWNDBuffer<uint, 2> flat_call_ids,
                           RWNDBuffer<uint, 2> flat_call_group_ids,
                           RWNDBuffer<uint, 2> flat_call_group_thread_ids)
{
    // Technically grid_cell should already be the call id
    int[2] call_id = get_call_id<2>();
    call_ids[uint3(grid_cell, 0)] = call_id[0];
    call_ids[uint3(grid_cell, 1)] = call_id[1];

    uint[2] call_group_id = get_call_group_id<2>();
    call_group_ids[uint3(grid_cell, 0)] = call_group_id[0];
    call_group_ids[uint3(grid_cell, 1)] = call_group_id[1];

    uint[2] call_group_thread_id = get_call_group_thread_id<2>();
    call_group_thread_ids[uint3(grid_cell, 0)] = call_group_thread_id[0];
    call_group_thread_ids[uint3(grid_cell, 1)] = call_group_thread_id[1];

    flat_call_ids[grid_cell] = get_flat_call_id();
    flat_call_group_ids[grid_cell] = get_flat_call_group_id();
    flat_call_group_thread_ids[grid_cell] = get_flat_call_group_thread_id();
}

void test2D_thread_id_info_test(uint2 grid_cell,
                           RWNDBuffer<uint, 2> flat_call_id,
                           RWNDBuffer<uint, 2> flat_call_group_id,
                           RWNDBuffer<uint, 2> flat_call_group_thread_id)
{
    int[2] call_id;
    get_call_id<2>(call_id);
    flat_call_id[grid_cell] = get_flat_call_id();
    flat_call_group_id[grid_cell] = get_flat_call_group_id();
    flat_call_group_thread_id[grid_cell] = get_flat_call_group_thread_id();
}


void test2D_thread_id_info_test_struct(uint2 grid_cell,
                                       RWNDBuffer<thread_id_info_2d, 2> thread_id_infos)
{
    thread_id_info_2d info;

    int[2] call_id = get_call_id<2>();
    info.call_id_x = call_id[0];
    info.call_id_y = call_id[1];

    thread_id_infos[grid_cell] = info;
}
*/

uint3 test3D(uint3 val)
{
    //return uint2(3, 5);
    // test
    return val;
}

uint4 test4D(uint4 val)
{
    //return uint2(3, 5);
    // test
    return val;
}

/*
Test test2(int2 val)
{
    Test t = {};
    t.grid_cell = val;

    t.flattened_call_thread_id = dispatchThreadID.x; // SV_DispatchThreadID.x
    t.flattened_call_group_thread_id = dispatchThreadID.x % 32; // "SV_GroupThreadID.x"
    t.flattened_call_group_id = dispatchThreadID.x / 32; // "SV_GroupID.x"

    t.call_group_thread_id = (((dispatchThreadID.x / 32)/call_data._grid_stride) % call_data._grid_dim);
    t.call_group_id = (((dispatchThreadID.x / 32)/call_data._grid_stride) % call_data._grid_dim);
    // Can do this because it is already computed for us, but it's calculated as
    // (((dispatchThreadID.x / 32)/call_data._grid_stride) % call_data._grid_dim) * call_data._group_dim + (((dispatchThreadID.x % 32)/call_data._group_stride) % call_data._group_dim)
    t.call_thread_id = call_id;
    
    return t;
}
*/
