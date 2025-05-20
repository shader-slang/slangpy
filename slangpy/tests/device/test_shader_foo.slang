// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Foo {
    uint a;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main_a(uint3 tid: SV_DispatchThreadID, uniform Foo foo) { }

[shader("compute")]
[numthreads(16, 8, 1)]
void main_b(uint3 tid: SV_DispatchThreadID, uniform Foo foo) { }

[shader("vertex")]
void main_vs() { }

[shader("fragment")]
void main_fs() { }
