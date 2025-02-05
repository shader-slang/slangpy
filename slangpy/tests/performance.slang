// SPDX-License-Identifier: Apache-2.0
import slangpy;

float add(float a, float b) {
    return a + b;
}

void increment(float val, inout float total) {
    for (int i = 0; i < 64; i++)
        total += val;
}

struct AddKernelData {
    StructuredBuffer<float> a;
    StructuredBuffer<float> b;
    RWStructuredBuffer<float> res;
    int count;
}
ParameterBlock<AddKernelData> addKernelData;

[numthreads(32, 1, 1)]
void addkernel(uint3 thread: SV_DispatchThreadID) {
    if (thread.x < addKernelData.count) {
        addKernelData.res[thread.x] = add(addKernelData.a[thread.x], addKernelData.b[thread.x]);
    }
}

struct AddKernelWithShapesData {
    NDBuffer<float, 1> a;
    NDBuffer<float,1> b;
    RWNDBuffer<float,1> res;
    int count;
}
ParameterBlock<AddKernelWithShapesData> addKernelWithShapesData;

[numthreads(32, 1, 1)]
void addkernelWithShapes(uint3 thread: SV_DispatchThreadID) {
    if (thread.x < addKernelData.count) {
        addKernelData.res[thread.x] = add(addKernelData.a[thread.x], addKernelData.b[thread.x]);
    }
}

struct IncrementKernelWithShapesData {
    NDBuffer<float, 1> val;
    RWNDBuffer<float, 1> total;
    int count;
}
ParameterBlock<IncrementKernelWithShapesData> incrementKernelWithShapesData;

[numthreads(32, 1, 1)]
void incrementkernelWithShapes(uint3 thread: SV_DispatchThreadID) {
    if (thread.x < addKernelData.count) {
        float val = incrementKernelWithShapesData.val[thread.x];
        float total = incrementKernelWithShapesData.total[thread.x];
        increment(val, total);
        incrementKernelWithShapesData.total[thread.x] = total;
    }
}
