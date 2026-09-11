// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <Python.h>

#if defined(_WIN32)
int wmain(int argc, wchar_t* argv[])
{
    return Py_Main(argc, argv);
}
#else
int main(int argc, char* argv[])
{
    return Py_BytesMain(argc, argv);
}
#endif
