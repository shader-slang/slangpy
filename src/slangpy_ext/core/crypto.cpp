// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/core/crypto.h"

SGL_PY_EXPORT(core_crypto)
{
    using namespace sgl;

    nb::class_<SHA1>(m, "SHA1", D(SHA1))
        .def(nb::init<>())
        .def(
            "__init__",
            [](SHA1* self, nb::bytes data)
            {
                new (self) SHA1(data.c_str(), data.size());
            },
            "data"_a,
            D(SHA1, SHA1, 2)
        )
        .def(nb::init<std::string_view>(), "str"_a, D(SHA1, SHA1, 3))
        .def(
            "update",
            [](SHA1& self, nb::bytes data)
            {
                self.update(data.c_str(), data.size());
                return self;
            },
            "data"_a,
            D(SHA1, update, 2)
        )
        .def(
            "update",
            [](SHA1& self, std::string_view str)
            {
                self.update(str);
                return self;
            },
            "str"_a,
            D(SHA1, update, 3)
        )
        .def(
            "digest",
            [](SHA1& self)
            {
                auto digest = self.digest();
                return nb::bytes((const char*)(digest.data()), digest.size());
            },
            D(SHA1, digest)
        )
        .def("hex_digest", &SHA1::hex_digest, D(SHA1, hex_digest));
}
