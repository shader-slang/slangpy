// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
module slangpy;

// TODO: Disable warning 30856 for the entire slangpy module.
// This is a workaround for an issue in the Slang compiler:
// https://github.com/shader-slang/slang/issues/8166
// Once fixed, we should go back to locally disabling the warnings where needed.
// Searching for "// #pragma warning(disable: 30856)" should reveal the relevant instances.
// Edit: Even this workaround isn't reliable, so we currently disable the warning globally in shader.cpp
#pragma warning(disable: 30856)

__include "core.slang";
__include "randfloatarg.slang";
__include "wanghasharg.slang";
__include "threadidarg.slang";
__include "callidarg.slang";
__include "atomics.slang";
__include "staticarray.slang";
__include "tensor.slang";
__include "gridarg.slang";
__include "callshape.slang";
