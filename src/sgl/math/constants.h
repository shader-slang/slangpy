// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cmath>
#include <cstdint>
#include <cfloat>

// We should move away from M_ constants, but for now fix issues when compiling with latest clang
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif
