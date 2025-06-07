import slangpy;

struct CallGroupInfo2D {
    int2 call_id;
    uint2 call_group_id;
    uint2 call_group_thread_id;
    uint flat_call_group_id;
    uint flat_call_group_thread_id;
}

CallGroupInfo2D test_call_group_2d(uint2 grid_cell,
                                   int2 call_id_arg,
                                   uint2 call_group_id_arg,
                                   uint2 call_group_thread_id_arg,
                                   uint flat_call_group_id_arg,
                                   uint flat_call_group_thread_id_arg)
{
    CallGroupInfo2D result;
    result.call_id = call_id_arg;
    result.call_group_id = call_group_id_arg;
    result.call_group_thread_id = call_group_thread_id_arg;
    result.flat_call_group_id = flat_call_group_id_arg;
    result.flat_call_group_thread_id = flat_call_group_thread_id_arg;
    return result;
}

int2 test_call_id_2d(uint2 grid_cell, int2 call_id_arg)
{
    return call_id_arg;
}

uint2 test_call_group_id_2d(uint2 grid_cell, uint2 call_group_id_arg)
{
    return call_group_id_arg;
}

uint2 test_call_group_thread_id_2d(uint2 grid_cell, uint2 call_group_thread_id_arg)
{
    return call_group_thread_id_arg;
}

uint test_flat_call_group_id(uint2 grid_cell, uint flat_call_group_id_arg)
{
    return flat_call_group_id_arg;
}

uint test_flat_call_group_thread_id(uint2 grid_cell, uint flat_call_group_thread_id_arg)
{
    return flat_call_group_thread_id_arg;
}
