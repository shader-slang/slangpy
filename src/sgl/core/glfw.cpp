// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "glfw.h"
#include "sgl/core/config.h"
#include "sgl/core/error.h"
#include "sgl/core/window.h"

#if SGL_HAS_GLFW
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace sgl::detail {

namespace {
    bool glfw_initialized = false;
    bool glfw_shutdown = false;
} // namespace

void ensure_glfw_initialized()
{
    SGL_CHECK(!glfw_shutdown, "GLFW has been shut down");
    if (!glfw_initialized) {
        SGL_CHECK(glfwInit() == GLFW_TRUE, "Failed to initialize GLFW");
        // Preserve the controller mapping previously installed by Window.
        // clang-format off
        const char* mappings = "03000000550900001472000000000000,NVIDIA Controller v01.04,a:b11,b:b10,back:b13,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b7,leftstick:b5,lefttrigger:a4,leftx:a0,lefty:a1,rightshoulder:b6,rightstick:b4,righttrigger:a5,rightx:a3,righty:a6,start:b3,x:b9,y:b8,platform:Windows,\n"
                "03000000550900001072000000000000,NVIDIA Shield,a:b9,b:b8,back:b11,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b5,leftstick:b3,lefttrigger:a3,leftx:a0,lefty:a1,rightshoulder:b4,rightstick:b2,righttrigger:a4,rightx:a2,righty:a5,start:b0,x:b7,y:b6,platform:Windows,\n"
                "030000005509000000b4000000000000,NVIDIA Virtual Gamepad,a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b8,lefttrigger:+a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b9,righttrigger:-a2,rightx:a3,righty:a4,start:b7,x:b2,y:b3,platform:Windows,";
        // clang-format on
        glfwUpdateGamepadMappings(mappings);
        glfw_initialized = true;
    }
}

void shutdown_glfw()
{
    glfw_shutdown = true;
    if (glfw_initialized) {
        Window::_destroy_all_windows();
        glfwTerminate();
        glfw_initialized = false;
    }
}

} // namespace sgl::detail

#endif
