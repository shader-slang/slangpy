// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/query.h"

namespace sgl {
SGL_DICT_TO_DESC_BEGIN(QueryPoolDesc)
SGL_DICT_TO_DESC_FIELD(type, QueryType)
SGL_DICT_TO_DESC_FIELD(count, uint32_t)
SGL_DICT_TO_DESC_END()
} // namespace sgl

SGL_PY_EXPORT(device_query)
{
    using namespace sgl;

    nb::class_<QueryPoolDesc>(m, "QueryPoolDesc", D(QueryPoolDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](QueryPoolDesc* self, nb::dict dict) { new (self) QueryPoolDesc(dict_to_QueryPoolDesc(dict)); }
        )
        .def_rw("type", &QueryPoolDesc::type, D(QueryPoolDesc, type))
        .def_rw("count", &QueryPoolDesc::count, D(QueryPoolDesc, count));
    nb::implicitly_convertible<nb::dict, QueryPoolDesc>();

    nb::class_<QueryPool, DeviceChild>(m, "QueryPool", D(QueryPool))
        .def_prop_ro("desc", &QueryPool::desc, D(QueryPool, desc))
        .def("reset", &QueryPool::reset, D(QueryPool, reset))
        .def("get_result", &QueryPool::get_result, "index"_a, D(QueryPool, get_result))
        .def(
            "get_results",
            nb::overload_cast<uint32_t, uint32_t>(&QueryPool::get_results),
            "index"_a,
            "count"_a,
            D(QueryPool, get_results)
        )
        .def("get_timestamp_result", &QueryPool::get_timestamp_result, "index"_a, D(QueryPool, get_timestamp_result))
        .def(
            "get_timestamp_results",
            nb::overload_cast<uint32_t, uint32_t>(&QueryPool::get_timestamp_results),
            "index"_a,
            "count"_a,
            D(QueryPool, get_timestamp_results)
        );
}
