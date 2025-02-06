// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

public struct ThreadIdArg {

    public void load(Context0D context, out uint3 value) {
        value = context.thread_id;
    }
    public void store(Context0D context, in uint3 value) {};

    public void load(Context0D context, out uint2 value) {
        value = context.thread_id.xy;
    }
    public void store(Context0D context, in uint2 value) {};

    public void load(Context0D context, out uint1 value) {
        value = context.thread_id.x;
    }
    public void store(Context0D context, in uint1 value) {};

    public void load(Context0D context, out uint value) {
        value = context.thread_id.x;
    }
    public void store(Context0D context, in uint value) {};
}
