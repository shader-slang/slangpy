// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/core/object.h"

SGL_PY_EXPORT(core_object)
{
    using namespace sgl;
    object_init_py(
        [](PyObject* o) noexcept
        {
            nb::gil_scoped_acquire guard;
            Py_INCREF(o);
        },
        [](PyObject* o) noexcept
        {
            nb::gil_scoped_acquire guard;
            Py_DECREF(o);
        },
        [](PyObject* o) noexcept -> Py_ssize_t_
        {
            nb::gil_scoped_acquire guard;
            return Py_REFCNT(o);
        }
    );

    nb::class_<Object>(
        m,
        "Object",
        nb::intrusive_ptr<Object>([](Object* o, PyObject* po) noexcept { o->set_self_py(po); }),
        "Base class for all reference counted objects."
    )
#if SGL_ENABLE_OBJECT_TRACKING
        .def_static("report_live_objects", &Object::report_live_objects)
#endif
        .def("__repr__", &Object::to_string);
}
