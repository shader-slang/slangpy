// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"

#define SGL_VERSION_MAJOR 0
#define SGL_VERSION_MINOR 25
#define SGL_VERSION_PATCH 0

#define SGL_VERSION                                                                                                    \
    SGL_TO_STRING(SGL_VERSION_MAJOR) "." SGL_TO_STRING(SGL_VERSION_MINOR) "." SGL_TO_STRING(SGL_VERSION_PATCH)

extern SGL_API const char* SGL_GIT_VERSION;

namespace sgl {

SGL_API void static_init();
SGL_API void static_shutdown();

} // namespace sgl
