add_subdirectory(simple_compute)
add_subdirectory(render_window)

# Ray tracing, barycentrics and the tev image viewer are not available with WebGPU in a browser.
if(NOT SGL_EMSCRIPTEN)
    add_subdirectory(pathtracer)
    add_subdirectory(raytracing_pipeline)
    add_subdirectory(raytracing)
    add_subdirectory(render_pipeline)
    add_subdirectory(tinybc)
endif()
