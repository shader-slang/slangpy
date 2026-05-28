// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/core/data_type.h"

SGL_PY_EXPORT(core_data_type)
{
    using namespace sgl;

    nb::sgl_enum<DataType>(m, "DataType", D(DataType));
}
