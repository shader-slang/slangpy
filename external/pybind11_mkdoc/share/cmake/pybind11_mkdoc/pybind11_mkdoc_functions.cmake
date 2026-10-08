# This function is used to run pybind11-mkdoc for the headers of a pybind11 module.
# In addition, this will also add target dependencies so the pybind11-mkdoc header
# file is generated before the pybind11 module. Also, this will automatically add
# the current binary directory to the pybind11 module's includes, so it can
# easily be included when compiling the module.
#
# The required parameters are:
# * OUTPUT - The name of the output file.
# * PYBIND11_MODULE - The pybind11 module target that these docs will be used for.
# * HEADERS - The header files to create docs for. These can be absolute paths or relative to the
#             current source directory.
#
# The optional parameters are:
# * EXTRA_ARGS - A list of arguments that will be added verbatim to the pybind11-mkdoc command.
#
# Example usage:
# pybind11_add_module(my_pybind11_module src/my_pybind11_module.cc)
# pybind11_mkdoc(
#     OUTPUT my_pybind11_module_doc.h
#     PYBIND11_MODULE my_pybind11_module
#     HEADERS
#         include/my_header_1.h
#         /absolute/path/to/header.h
# )
function(pybind11_mkdoc)
  set(options)
  set(oneValueArgs OUTPUT PYBIND11_MODULE)
  set(multiValueArgs HEADERS EXTRA_ARGS)
  cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}" "${multiValueArgs}")

  # Include directories for the pybind11 module
  set(prop "$<TARGET_PROPERTY:${arg_PYBIND11_MODULE},INCLUDE_DIRECTORIES>")

  # Remove the header file from the list
  set(HEADERS "")
  # Run through all the other arguments.
  foreach(header ${arg_HEADERS})
    if(IS_ABSOLUTE ${header})
      # If it is an absolute path, then add it as is.
      list(APPEND HEADERS ${header})
    else()
      # Otherwise, assume it is relative to the current source directory.
      list(APPEND HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/${header}")
    endif()
  endforeach()

  # Add a custom target and command for the full header file location that runs pybind11-mkdoc
  # We automatically include the source directory and build/include
  add_custom_target(pybind11_mkdoc_${arg_OUTPUT} DEPENDS ${arg_OUTPUT})

  add_custom_command(
    OUTPUT ${arg_OUTPUT}
    COMMAND ${Python_EXECUTABLE} -m pybind11_mkdoc ${arg_EXTRA_ARGS} -o ${arg_OUTPUT}
            "$<$<BOOL:${prop}>:-I$<JOIN:${prop},;-I>>" ${HEADERS}
    DEPENDS ${HEADERS}
    COMMAND_EXPAND_LISTS)

  # Add a dependency so that the pybind11-mkdoc command runs before we try to compile the pybind11 module
  add_dependencies(${arg_PYBIND11_MODULE} pybind11_mkdoc_${arg_OUTPUT})

  # Add the current binary directory to the pybind11 module so it can be included easily
  target_include_directories(${arg_PYBIND11_MODULE} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endfunction()
