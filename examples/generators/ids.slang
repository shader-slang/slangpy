// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

int2 myfunc(int2 value) {
    return value;
}

int2 myfuncarray(int[2] value) {
    return int2(value[0],value[1]);
}

int3 myfunc3d(int3 value) {
    return value;
}
