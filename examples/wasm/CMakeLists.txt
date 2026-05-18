# Render Pipeline Example
add_executable(render-pipeline-wasm
    ../render_pipeline/render_pipeline.cpp
)

target_compile_features(render-pipeline-wasm PRIVATE cxx_std_20)
target_link_libraries(render-pipeline-wasm PRIVATE sgl)
target_link_options(render-pipeline-wasm PRIVATE
    "SHELL:--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/../render_pipeline/render_pipeline.slang@render_pipeline.slang"
)
target_compile_definitions(render-pipeline-wasm PRIVATE SGL_EXAMPLE_DIR=".")

add_custom_command(TARGET render-pipeline-wasm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/../render_pipeline/render_pipeline.slang
        $<TARGET_FILE_DIR:render-pipeline-wasm>/render_pipeline.slang
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/render_pipeline.html
        $<TARGET_FILE_DIR:render-pipeline-wasm>/render_pipeline.html
)

# Simple Compute Example
add_executable(simple-compute-wasm
    ../simple_compute/simple_compute.cpp
)

target_compile_features(simple-compute-wasm PRIVATE cxx_std_20)
target_link_libraries(simple-compute-wasm PRIVATE sgl)
target_link_options(simple-compute-wasm PRIVATE
    "SHELL:--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/../simple_compute/simple_compute.slang@simple_compute.slang"
)
target_compile_definitions(simple-compute-wasm PRIVATE SGL_EXAMPLE_DIR=".")

add_custom_command(TARGET simple-compute-wasm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple_compute/simple_compute.slang
        $<TARGET_FILE_DIR:simple-compute-wasm>/simple_compute.slang
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/simple_compute.html
        $<TARGET_FILE_DIR:simple-compute-wasm>/simple_compute.html
)
