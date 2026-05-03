add_subdirectory(pathtracer)
add_subdirectory(raytracing_pipeline)
add_subdirectory(raytracing)
add_subdirectory(render_pipeline)
add_subdirectory(simple_compute)
add_subdirectory(tinybc)

# WASM example for Emscripten
if(SGL_EMSCRIPTEN)
    add_subdirectory(wasm)
endif()
