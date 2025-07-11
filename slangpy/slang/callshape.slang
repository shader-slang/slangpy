// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

implementing slangpy;

// Those are link-time constants, this constant can be generated during run-time
public extern static const int call_data_len;
public extern static const int call_group_size;
public extern static const int[call_data_len] call_group_strides;
public extern static const int[call_data_len] call_group_shape_vector;

static int[call_data_len] call_id;
static int[call_data_len] call_group_id;
static int[call_data_len] call_group_thread_id;

// The reason we design the struct this way is because user will have no idea
// about the value of the `call_data_len` when they are writing the code. Therefore,
// if they try to use the `call_id`, `call_group_id`, or `call_group_thread_id` directly,
// they will not know the type. So we use the interface to hide the details, so they will
// query the length and access the shape data from the interface.
internal enum CallShapeInfoFlag
{
    CallId,
    CallGroupId,
    CallGroupThreadId
}

public struct CallShapeInfo
{
    private int m_flag;

    // We don't want to expose the constructor to the user, so we make it internal.
    // External users should use the static methods to get the CallShapeInfo.
    internal __init(int flag)
    {
        m_flag = flag;
    }

    public property int dimensionality
    {
        get { return call_data_len; }
    }

    public property int[call_data_len] shape
    {
        get
        {
            switch (m_flag)
            {
                case CallShapeInfoFlag::CallId:
                    return call_id;
                case CallShapeInfoFlag::CallGroupId:
                    return call_group_id;
                case CallShapeInfoFlag::CallGroupThreadId:
                    return call_group_thread_id;
                default:
                    return {};
            }
        }
    }

    public static CallShapeInfo get_call_id()
    {
        CallShapeInfo call_shape_info = CallShapeInfo(CallShapeInfoFlag::CallId);
        return call_shape_info;
    }

    public static CallShapeInfo get_call_group_id()
    {
        CallShapeInfo call_shape_info = CallShapeInfo(CallShapeInfoFlag::CallGroupId);
        return call_shape_info;
    }

    public static CallShapeInfo get_call_group_thread_id()
    {
        CallShapeInfo call_shape_info = CallShapeInfo(CallShapeInfoFlag::CallGroupThreadId);
        return call_shape_info;
    }
}

// This function is used to initialize the thread local call shape info.
// If the call shape is not aligned to the call group shape, we will return false,
// Otherwise, it will return true.
// TODO: The reason we have to use generic here is because grid_stride/grid_dim/call_dim
// are the shader parameters and they cannot use link-time constant as array size because
// of the reflection doesn't support that (https://github.com/shader-slang/slang/pull/7067).
// Therefore we cannot use "const int[call_data_len]",  instead we have to use generic
// as generic parameter is also a link-time constant here.
public bool init_thread_local_call_shape_info<let N: int = call_data_len>(
    int flat_call_group_thread_id,
    int3 flat_call_group_id,
    int3 flat_call_thread_id,
    const int[N] grid_stride,
    const int[N] grid_dim,
    const int[N] call_dim)
{
    if (call_group_size != 1)
    {
        [unroll]
        for (int i = 0; i < N; i++)
        {
            call_group_thread_id[i] = (flat_call_group_thread_id / call_group_strides[i]) % call_group_shape_vector[i];
            call_group_id[i]        = (flat_call_group_id.x      / grid_stride[i])        % grid_dim[i];
            call_id[i]              = call_group_id[i] * call_group_shape_vector[i] + call_group_thread_id[i];
        }

        // The Slang compiler seems to have trouble unrolling the for loop above when it
        // contains the if statement in the below. Separate the if checks into a separate
        // loop for now to better facilitate unrolling and improve perf.
        [unroll]
        for (int i = 0; i < N; i++)
        {

            // In the event that the call shape is not aligned to the call group shape,
            // we use an aligned call shape to calculate the number of threads that we
            // need. However, that means that some threads will fall outside the call shape
            // and as a result will need to return early and be wasted.
            if (call_id[i] >= call_dim[i])
            {
                return false;
            }
        }
    }
    else
    {
        [unroll]
        for (int i = 0; i < N; i++)
        {
            call_id[i] = (flat_call_thread_id.x / grid_stride[i]) % grid_dim[i];
            call_group_thread_id[i] = 0;
            call_group_id[i] = call_id[i];
        }
    }
    return true;
}
