# -----------------------------------------------------------------------------
# sgl unit tests
# -----------------------------------------------------------------------------

if(SGL_BUILD_TESTS)

    add_executable(sgl_tests)
    target_sources(sgl_tests PRIVATE
        sgl/sgl_tests.cpp
        sgl/testing.cpp
        sgl/core/test_dds_file.cpp
        sgl/core/test_enum.cpp
        sgl/core/test_file_system_watcher.cpp
        sgl/core/test_lmdb_cache.cpp
        sgl/core/test_maths.cpp
        sgl/core/test_memory_mapped_file.cpp
        sgl/core/test_object.cpp
        sgl/core/test_platform.cpp
        sgl/core/test_plugin.cpp
        sgl/core/test_short_vector.cpp
        sgl/core/test_static_vector.cpp
        sgl/core/test_stream.cpp
        sgl/core/test_string.cpp
        sgl/device/test_device.cpp
        sgl/device/test_hot_reload.cpp
        sgl/device/test_formats.cpp
        sgl/device/test_shader.cpp
        sgl/math/test_float16.cpp
        sgl/math/test_matrix.cpp
        sgl/math/test_quaternion.cpp
        sgl/math/test_vector.cpp
    )
    target_include_directories(sgl_tests BEFORE PRIVATE sgl)
    target_link_libraries(sgl_tests PRIVATE sgl header_only)
    target_compile_definitions(sgl_tests PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

    set_target_properties(sgl_tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${SGL_RUNTIME_OUTPUT_DIRECTORY}
        LIBRARY_OUTPUT_DIRECTORY ${SGL_LIBRARY_OUTPUT_DIRECTORY}
    )

    add_test(NAME sgl_tests COMMAND $<TARGET_FILE:sgl_tests>)

endif(SGL_BUILD_TESTS)
