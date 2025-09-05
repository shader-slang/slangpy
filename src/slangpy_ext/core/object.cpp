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
        .def_static(
            "report_live_objects",
            [](bool log_to_tty = true)
            {
                // We want to avoid creating new live objects by reporting objects, so instead of
                // creating bindings for LiveObjectInfo, convert each to a dictionary.
                auto live_objects = Object::report_live_objects(log_to_tty);
                nb::list result;
                for (const auto& info : live_objects) {
                    nb::dict obj_dict;
                    obj_dict["object"] = reinterpret_cast<uintptr_t>(info.object);
                    obj_dict["ref_count"] = info.ref_count;
                    obj_dict["self_py"] = reinterpret_cast<uintptr_t>(info.self_py);
                    obj_dict["class_name"] = info.class_name;
                    result.append(obj_dict);
                }
                return result;
            },
            "log_to_tty"_a = true,
            "Returns a list of dictionaries containing information about live objects"
        )
#endif
        .def("__repr__", &Object::to_string);
}
