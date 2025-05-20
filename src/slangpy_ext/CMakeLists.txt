
# slangpy python package path
set(SLANGPY_PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../slangpy)

nanobind_add_module(slangpy_ext NB_STATIC LTO
    slangpy_ext.cpp
    app/app.cpp
    core/bitmap.cpp
    core/crypto.cpp
    core/data_struct.cpp
    core/data_type.cpp
    core/input.cpp
    core/logger.cpp
    core/object.cpp
    core/platform.cpp
    core/thread.cpp
    core/timer.cpp
    core/window.cpp
    device/buffer_cursor.cpp
    device/command.cpp
    device/coopvec.cpp
    device/cursor_utils.h
    device/device_resource.cpp
    device/device.cpp
    device/fence.cpp
    device/formats.cpp
    device/input_layout.cpp
    device/kernel.cpp
    device/native_handle.cpp
    device/pipeline.cpp
    device/query.cpp
    device/raytracing.cpp
    device/reflection.cpp
    device/resource.cpp
    device/sampler.cpp
    device/shader_cursor.cpp
    device/shader_object.cpp
    device/shader.cpp
    device/surface.cpp
    device/types.cpp
    math/matrix.cpp
    math/quaternion.cpp
    math/scalar.cpp
    math/vector.cpp
    ui/ui.cpp
    ui/widgets.cpp
    utils/renderdoc.cpp
    utils/slangpy.h
    utils/slangpy.cpp
    utils/slangpyvalue.h
    utils/slangpyvalue.cpp
    utils/slangpystridedbufferview.h
    utils/slangpystridedbufferview.cpp
    utils/slangpybuffer.h
    utils/slangpybuffer.cpp
    utils/slangpyfunction.h
    utils/slangpyfunction.cpp
    utils/slangpyresources.h
    utils/slangpyresources.cpp
    utils/slangpytensor.h
    utils/slangpytensor.cpp
    utils/tev.cpp
    utils/texture_loader.cpp
)

if(SGL_ENABLE_PCH)
    target_precompile_headers(slangpy_ext
        PRIVATE
            slangpy_ext_pch.h
    )
endif()

target_include_directories(slangpy_ext PRIVATE .)

target_link_libraries(slangpy_ext PRIVATE sgl)
set_target_properties(slangpy_ext PROPERTIES
    OUTPUT_NAME slangpy_ext
    ARCHIVE_OUTPUT_NAME slangpy_ext
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${SLANGPY_PACKAGE_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${SLANGPY_PACKAGE_DIR}
    PDB_OUTPUT_DIRECTORY_RELEASE ${SLANGPY_PACKAGE_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${SLANGPY_PACKAGE_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${SLANGPY_PACKAGE_DIR}
    PDB_OUTPUT_DIRECTORY_DEBUG ${SLANGPY_PACKAGE_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${SLANGPY_PACKAGE_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${SLANGPY_PACKAGE_DIR}
    PDB_OUTPUT_DIRECTORY_RELWITHDEBINFO ${SLANGPY_PACKAGE_DIR}
)

# Write a .build_dir file to the package directory.
# This is used by __init__.py to setup the correct DLL path in development mode.
add_custom_target(
    slangpy_ext_build_dir
    COMMAND echo "${SGL_OUTPUT_DIRECTORY}" > ${SLANGPY_PACKAGE_DIR}/.build_dir
)

add_dependencies(slangpy_ext slangpy_ext_build_dir)

# Post processes (and overwrites) a stub file.
# Note: The use of cached variable allows this to be called from other CMake scripts.
set(SGL_POST_PROCESS_STUB_PY ${CMAKE_CURRENT_LIST_DIR}/../../tools/postprocess_stub.py CACHE INTERNAL "")
function(postprocess_stub stub_file args)
    add_custom_command(
        APPEND
        OUTPUT ${stub_file}
        COMMAND ${Python_EXECUTABLE} ${SGL_POST_PROCESS_STUB_PY} --file=${stub_file} ${args} --quiet
        COMMENT "Post-processing stub ${stub_file}"
    )
endfunction()

# Generate main stub file.
nanobind_add_stub(
    slangpy_stub
    MODULE slangpy
    PYTHON_PATH ${SLANGPY_PACKAGE_DIR}/..
    OUTPUT ${SLANGPY_PACKAGE_DIR}/__init__.pyi
    DEPENDS slangpy_ext ${SLANGPY_PACKAGE_DIR}/__init__.py
    INCLUDE_PRIVATE # allow us to have functions/variables that start with _
)

# Post-process the main stub file.
postprocess_stub(${SLANGPY_PACKAGE_DIR}/__init__.pyi "")

# Install the main stub file.
install(FILES ${SLANGPY_PACKAGE_DIR}/__init__.pyi DESTINATION ${CMAKE_INSTALL_BINDIR})

# Generate submodule stub files.
foreach(submodule IN ITEMS platform thread math ui tev renderdoc slangpy)
    string(REPLACE "." "/" submodule_path ${submodule})
    nanobind_add_stub(
        slangpy_${submodule}_stub
        MODULE slangpy.${submodule}
        PYTHON_PATH ${SLANGPY_PACKAGE_DIR}/..
        OUTPUT ${SLANGPY_PACKAGE_DIR}/${submodule_path}/__init__.pyi
        DEPENDS slangpy_ext ${SLANGPY_PACKAGE_DIR}/${submodule_path}/__init__.py
        INCLUDE_PRIVATE # allow us to have functions/variables that start with _
    )

    # Post-process the submodule stub file.
    postprocess_stub(${SLANGPY_PACKAGE_DIR}/${submodule_path}/__init__.pyi "--submodule")

    # Install the submodule stub file.
    install(FILES ${SLANGPY_PACKAGE_DIR}/${submodule_path}/__init__.pyi DESTINATION ${CMAKE_INSTALL_BINDIR}/${submodule_path})

endforeach()

if(SGL_MASTER_PROJECT)
    set(MKDOC_ARGS "")

    # Add system include directories.
    foreach(value ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
        list(APPEND MKDOC_ARGS -I${value})
    endforeach()

    # Add target include directories.
    get_target_property(list sgl INCLUDE_DIRECTORIES)
    foreach(value ${list})
        list(APPEND MKDOC_ARGS -I${value})
    endforeach()
    get_target_property(list sgl INTERFACE_INCLUDE_DIRECTORIES)
    foreach(value ${list})
        list(APPEND MKDOC_ARGS -I${value})
    endforeach()

    # Add target include directories from dependencies.
    get_target_property(targets sgl INTERFACE_LINK_LIBRARIES)
    if(targets)
        foreach(value ${targets})
            # Skip generator expressions
            if(value MATCHES "^\\$<.*>$")
                # message(STATUS "Skipping target: ${value}")
                continue()
            endif()
            get_target_property(list ${value} INTERFACE_INCLUDE_DIRECTORIES)
            if(list)
                foreach(value ${list})
                    # Unwrap BUILD_INTERFACE generator expression
                    string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]+)>" "\\1" value "${value}")
                    # Skip generator expressions
                    if(value MATCHES "^\\$<.*>$")
                        # message(STATUS "Skipping path: ${value}")
                        continue()
                    endif()
                    list(APPEND MKDOC_ARGS -I${value})
                endforeach()
            endif()
        endforeach()
    endif()

    # Note: The current version of pybind11_mkdoc does not support C++20.
    # Running this will result in the following error:
    # ValueError: Unknown template argument kind 604
    # To fix this, we need to manually change the cindex.py file in the
    # clang package and add the following lines:
    # > # A concept declaration
    # > CursorKind.CONCEPT_DECL = CursorKind(604)
    add_custom_target(pydoc USES_TERMINAL COMMAND
        COMMAND ${Python_EXECUTABLE} -m pybind11_mkdoc -std=c++20 -stdlib=libc++
        ${MKDOC_ARGS}
        -I${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../sgl/app/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../sgl/core/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../sgl/device/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../sgl/math/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../sgl/ui/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../sgl/utils/*.h
        -o ${CMAKE_CURRENT_SOURCE_DIR}/py_doc.h
    )
endif()

# Install binaries.
install(
    TARGETS slangpy_ext
    LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
