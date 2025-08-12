/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define __EXPAND(x)                                      x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define __VA_SIZE(...)                                   __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __CAT1(a, b)                                     a ## b
#define __CAT2(a, b)                                     __CAT1(a, b)
#define __DOC1(n1)                                       __doc_##n1
#define __DOC2(n1, n2)                                   __doc_##n1##_##n2
#define __DOC3(n1, n2, n3)                               __doc_##n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4)                           __doc_##n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5)                       __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6)                   __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7)               __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                         __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif


static const char *__doc_GLFWwindow = R"doc()doc";

static const char *__doc_HWND = R"doc()doc";

static const char *__doc_ImFont = R"doc()doc";

static const char *__doc_ImGuiContext = R"doc()doc";

static const char *__doc_ImGui_PushFont = R"doc()doc";

static const char *__doc_SourceLocation =
R"doc(Source location information. This is similar to std::source_location
which unfortunately is still not well supported on all major C++
library implementations.)doc";

static const char *__doc_SourceLocation_column = R"doc()doc";

static const char *__doc_SourceLocation_current = R"doc()doc";

static const char *__doc_SourceLocation_file_name = R"doc()doc";

static const char *__doc_SourceLocation_function_name = R"doc()doc";

static const char *__doc_SourceLocation_line = R"doc()doc";

static const char *__doc_formatter = R"doc()doc";

static const char *__doc_formatter_2 = R"doc()doc";

static const char *__doc_formatter_format = R"doc()doc";

static const char *__doc_formatter_format_2 = R"doc()doc";

static const char *__doc_sgl_AccelerationStructure = R"doc()doc";

static const char *__doc_sgl_AccelerationStructure_2 = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildDesc = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildDescConverter = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildDescConverter_AccelerationStructureBuildDescConverter = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildDescConverter_rhi_build_inputs = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildDescConverter_rhi_desc = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildDesc_flags = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildDesc_inputs = R"doc(List of build inputs. All inputs must be of the same type.)doc";

static const char *__doc_sgl_AccelerationStructureBuildDesc_mode = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildDesc_motion_options = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildFlags = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildFlags_allow_compaction = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildFlags_allow_update = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildFlags_info = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildFlags_minimize_memory = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildFlags_none = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildFlags_prefer_fast_build = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildFlags_prefer_fast_trace = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputInstances = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputInstances_instance_buffer = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputInstances_instance_count = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputInstances_instance_stride = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputMotionOptions = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputMotionOptions_key_count = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputMotionOptions_time_end = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputMotionOptions_time_start = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputProceduralPrimitives = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputProceduralPrimitives_aabb_buffers = R"doc(List of AABB buffers, one for each motion step.)doc";

static const char *__doc_sgl_AccelerationStructureBuildInputProceduralPrimitives_aabb_stride = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputProceduralPrimitives_flags = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputProceduralPrimitives_primitive_count = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputTriangles = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputTriangles_flags = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputTriangles_index_buffer = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputTriangles_index_count = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputTriangles_index_format = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputTriangles_pre_transform_buffer =
R"doc(Optional buffer containing 3x4 transform matrix applied to each
vertex.)doc";

static const char *__doc_sgl_AccelerationStructureBuildInputTriangles_vertex_buffers = R"doc(List of vertex buffers, one for each motion step.)doc";

static const char *__doc_sgl_AccelerationStructureBuildInputTriangles_vertex_count = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputTriangles_vertex_format = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildInputTriangles_vertex_stride = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildMode = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildMode_build = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildMode_info = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureBuildMode_update = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureCopyMode = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureCopyMode_clone = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureCopyMode_compact = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureCopyMode_info = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureDesc = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureDesc_2 = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureDesc_label = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureDesc_size = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureGeometryFlags = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureGeometryFlags_info = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureGeometryFlags_no_duplicate_any_hit_invocation = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureGeometryFlags_none = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureGeometryFlags_opaque = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceDesc = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceDesc_acceleration_structure = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceDesc_flags = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceDesc_instance_contribution_to_hit_group_index = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceDesc_instance_id = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceDesc_instance_mask = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceDesc_transform = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceFlags = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceFlags_force_opaque = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceFlags_info = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceFlags_no_opaque = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceFlags_none = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceFlags_triangle_facing_cull_disable = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceFlags_triangle_front_counter_clockwise = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_2 = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_AccelerationStructureInstanceList = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_buffer = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_build_input_instances = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_class_name = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_instance_stride = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_m_buffer = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_m_dirty = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_m_instance_stride = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_m_instance_type = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_m_instances = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_resize = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_size = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_to_string = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_write = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureInstanceList_write_2 = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureQueryDesc = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureQueryDesc_first_query_index = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureQueryDesc_query_pool = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureQueryDesc_query_type = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureSizes = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureSizes_acceleration_structure_size = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureSizes_scratch_size = R"doc()doc";

static const char *__doc_sgl_AccelerationStructureSizes_update_scratch_size = R"doc()doc";

static const char *__doc_sgl_AccelerationStructure_AccelerationStructure = R"doc()doc";

static const char *__doc_sgl_AccelerationStructure_class_name = R"doc()doc";

static const char *__doc_sgl_AccelerationStructure_desc = R"doc()doc";

static const char *__doc_sgl_AccelerationStructure_handle = R"doc()doc";

static const char *__doc_sgl_AccelerationStructure_m_desc = R"doc()doc";

static const char *__doc_sgl_AccelerationStructure_m_rhi_acceleration_structure = R"doc()doc";

static const char *__doc_sgl_AccelerationStructure_rhi_acceleration_structure = R"doc()doc";

static const char *__doc_sgl_AccelerationStructure_to_string = R"doc()doc";

static const char *__doc_sgl_AdapterInfo = R"doc()doc";

static const char *__doc_sgl_AdapterInfo_device_id =
R"doc(Unique identifier for the physical device among devices from the
vendor (only available for D3D12 and Vulkan).)doc";

static const char *__doc_sgl_AdapterInfo_luid = R"doc(Logically unique identifier of the adapter.)doc";

static const char *__doc_sgl_AdapterInfo_name = R"doc(Descriptive name of the adapter.)doc";

static const char *__doc_sgl_AdapterInfo_to_string = R"doc()doc";

static const char *__doc_sgl_AdapterInfo_vendor_id =
R"doc(Unique identifier for the vendor (only available for D3D12 and
Vulkan).)doc";

static const char *__doc_sgl_AnyCursorValue =
R"doc(Dummy type to represent traits of an arbitrary value type usable by
cursors)doc";

static const char *__doc_sgl_App = R"doc()doc";

static const char *__doc_sgl_AppDesc = R"doc()doc";

static const char *__doc_sgl_AppDesc_device =
R"doc(Device to use for rendering. If not provided, a default device will be
created.)doc";

static const char *__doc_sgl_AppWindow = R"doc()doc";

static const char *__doc_sgl_AppWindow_2 = R"doc()doc";

static const char *__doc_sgl_AppWindowDesc = R"doc()doc";

static const char *__doc_sgl_AppWindowDesc_app = R"doc()doc";

static const char *__doc_sgl_AppWindowDesc_enable_vsync = R"doc(Enable/disable vertical synchronization.)doc";

static const char *__doc_sgl_AppWindowDesc_height = R"doc(Height of the window in pixels.)doc";

static const char *__doc_sgl_AppWindowDesc_mode = R"doc(Window mode.)doc";

static const char *__doc_sgl_AppWindowDesc_resizable = R"doc(Whether the window is resizable.)doc";

static const char *__doc_sgl_AppWindowDesc_surface_format = R"doc(Format of the swapchain images.)doc";

static const char *__doc_sgl_AppWindowDesc_title = R"doc(Title of the window.)doc";

static const char *__doc_sgl_AppWindowDesc_width = R"doc(Width of the window in pixels.)doc";

static const char *__doc_sgl_AppWindow_AppWindow = R"doc()doc";

static const char *__doc_sgl_AppWindow_RenderContext = R"doc()doc";

static const char *__doc_sgl_AppWindow_RenderContext_command_encoder = R"doc()doc";

static const char *__doc_sgl_AppWindow_RenderContext_surface_texture = R"doc()doc";

static const char *__doc_sgl_AppWindow_app = R"doc()doc";

static const char *__doc_sgl_AppWindow_class_name = R"doc()doc";

static const char *__doc_sgl_AppWindow_device = R"doc()doc";

static const char *__doc_sgl_AppWindow_handle_drop_files = R"doc()doc";

static const char *__doc_sgl_AppWindow_handle_gamepad_event = R"doc()doc";

static const char *__doc_sgl_AppWindow_handle_keyboard_event = R"doc()doc";

static const char *__doc_sgl_AppWindow_handle_mouse_event = R"doc()doc";

static const char *__doc_sgl_AppWindow_handle_resize = R"doc()doc";

static const char *__doc_sgl_AppWindow_m_app = R"doc()doc";

static const char *__doc_sgl_AppWindow_m_device = R"doc()doc";

static const char *__doc_sgl_AppWindow_m_surface = R"doc()doc";

static const char *__doc_sgl_AppWindow_m_surface_config = R"doc()doc";

static const char *__doc_sgl_AppWindow_m_ui_context = R"doc()doc";

static const char *__doc_sgl_AppWindow_m_window = R"doc()doc";

static const char *__doc_sgl_AppWindow_on_drop_files = R"doc()doc";

static const char *__doc_sgl_AppWindow_on_gamepad_event = R"doc()doc";

static const char *__doc_sgl_AppWindow_on_keyboard_event = R"doc()doc";

static const char *__doc_sgl_AppWindow_on_mouse_event = R"doc()doc";

static const char *__doc_sgl_AppWindow_on_resize = R"doc()doc";

static const char *__doc_sgl_AppWindow_render = R"doc()doc";

static const char *__doc_sgl_AppWindow_render_ui = R"doc()doc";

static const char *__doc_sgl_AppWindow_run_frame = R"doc()doc";

static const char *__doc_sgl_AppWindow_screen = R"doc()doc";

static const char *__doc_sgl_AppWindow_should_close = R"doc()doc";

static const char *__doc_sgl_App_App = R"doc()doc";

static const char *__doc_sgl_App_add_window = R"doc()doc";

static const char *__doc_sgl_App_class_name = R"doc()doc";

static const char *__doc_sgl_App_device = R"doc()doc";

static const char *__doc_sgl_App_m_device = R"doc()doc";

static const char *__doc_sgl_App_m_terminate = R"doc()doc";

static const char *__doc_sgl_App_m_windows = R"doc()doc";

static const char *__doc_sgl_App_remove_window = R"doc()doc";

static const char *__doc_sgl_App_run = R"doc()doc";

static const char *__doc_sgl_App_run_frame = R"doc()doc";

static const char *__doc_sgl_App_terminate = R"doc()doc";

static const char *__doc_sgl_AspectBlendDesc = R"doc()doc";

static const char *__doc_sgl_AspectBlendDesc_dst_factor = R"doc()doc";

static const char *__doc_sgl_AspectBlendDesc_op = R"doc()doc";

static const char *__doc_sgl_AspectBlendDesc_src_factor = R"doc()doc";

static const char *__doc_sgl_BaseReflectionIndexedList =
R"doc(Base class for read-only lazy evaluation list of search results. To
use it, the search function (e.g. children_of_kind) fills out the
indices of the children matching the search. The list can then be
accessed as usual, and only allocates/evaluates children as they're
requested.)doc";

static const char *__doc_sgl_BaseReflectionIndexedList_BaseReflectionIndexedList = R"doc()doc";

static const char *__doc_sgl_BaseReflectionIndexedList_BaseReflectionIndexedList_2 = R"doc()doc";

static const char *__doc_sgl_BaseReflectionIndexedList_BaseReflectionIndexedList_3 = R"doc()doc";

static const char *__doc_sgl_BaseReflectionIndexedList_m_indices = R"doc()doc";

static const char *__doc_sgl_BaseReflectionIndexedList_operator_array = R"doc(Index operator)doc";

static const char *__doc_sgl_BaseReflectionIndexedList_size = R"doc(Number of search results in list.)doc";

static const char *__doc_sgl_BaseReflectionList =
R"doc(Base class for read-only lazy evaluation list. This only maintains a
reference to the parent, and allocates/returns children on demand.)doc";

static const char *__doc_sgl_BaseReflectionList_BaseReflectionList = R"doc()doc";

static const char *__doc_sgl_BaseReflectionList_BaseReflectionList_2 = R"doc()doc";

static const char *__doc_sgl_BaseReflectionList_BaseReflectionList_3 = R"doc()doc";

static const char *__doc_sgl_BaseReflectionList_Iterator = R"doc()doc";

static const char *__doc_sgl_BaseReflectionList_Iterator_Iterator = R"doc()doc";

static const char *__doc_sgl_BaseReflectionList_Iterator_m_index = R"doc()doc";

static const char *__doc_sgl_BaseReflectionList_Iterator_m_list = R"doc()doc";

static const char *__doc_sgl_BaseReflectionList_Iterator_operator_eq = R"doc()doc";

static const char *__doc_sgl_BaseReflectionList_Iterator_operator_inc = R"doc()doc";

static const char *__doc_sgl_BaseReflectionList_Iterator_operator_inc_2 = R"doc()doc";

static const char *__doc_sgl_BaseReflectionList_Iterator_operator_mul = R"doc()doc";

static const char *__doc_sgl_BaseReflectionList_Iterator_operator_ne = R"doc()doc";

static const char *__doc_sgl_BaseReflectionList_begin = R"doc(Begin iterator)doc";

static const char *__doc_sgl_BaseReflectionList_end = R"doc(End iterator)doc";

static const char *__doc_sgl_BaseReflectionList_evaluate = R"doc(Evaluate and return a specific child in the parent object.)doc";

static const char *__doc_sgl_BaseReflectionList_m_owner = R"doc()doc";

static const char *__doc_sgl_BaseReflectionList_operator_array = R"doc(Index operator)doc";

static const char *__doc_sgl_BaseReflectionList_size = R"doc(Number of entries in list.)doc";

static const char *__doc_sgl_BaseReflectionObject = R"doc()doc";

static const char *__doc_sgl_BaseReflectionObjectImpl = R"doc()doc";

static const char *__doc_sgl_BaseReflectionObjectImpl_BaseReflectionObjectImpl = R"doc()doc";

static const char *__doc_sgl_BaseReflectionObjectImpl_hot_reload_invalidate = R"doc()doc";

static const char *__doc_sgl_BaseReflectionObjectImpl_m_target = R"doc()doc";

static const char *__doc_sgl_BaseReflectionObjectImpl_slang_target = R"doc()doc";

static const char *__doc_sgl_BaseReflectionObject_BaseReflectionObject = R"doc()doc";

static const char *__doc_sgl_BaseReflectionObject_hot_reload_invalidate = R"doc()doc";

static const char *__doc_sgl_BaseReflectionObject_is_valid = R"doc()doc";

static const char *__doc_sgl_BaseReflectionObject_m_owner = R"doc()doc";

static const char *__doc_sgl_Bitmap = R"doc()doc";

static const char *__doc_sgl_Bitmap_2 = R"doc()doc";

static const char *__doc_sgl_Bitmap_Bitmap = R"doc()doc";

static const char *__doc_sgl_Bitmap_Bitmap_2 = R"doc()doc";

static const char *__doc_sgl_Bitmap_Bitmap_3 = R"doc()doc";

static const char *__doc_sgl_Bitmap_Bitmap_4 = R"doc(Copy constructor.)doc";

static const char *__doc_sgl_Bitmap_Bitmap_5 = R"doc(Move constructor.)doc";

static const char *__doc_sgl_Bitmap_FileFormat = R"doc()doc";

static const char *__doc_sgl_Bitmap_FileFormat_auto = R"doc()doc";

static const char *__doc_sgl_Bitmap_FileFormat_bmp = R"doc()doc";

static const char *__doc_sgl_Bitmap_FileFormat_exr = R"doc()doc";

static const char *__doc_sgl_Bitmap_FileFormat_hdr = R"doc()doc";

static const char *__doc_sgl_Bitmap_FileFormat_info = R"doc()doc";

static const char *__doc_sgl_Bitmap_FileFormat_jpg = R"doc()doc";

static const char *__doc_sgl_Bitmap_FileFormat_png = R"doc()doc";

static const char *__doc_sgl_Bitmap_FileFormat_tga = R"doc()doc";

static const char *__doc_sgl_Bitmap_FileFormat_unknown = R"doc()doc";

static const char *__doc_sgl_Bitmap_PixelFormat = R"doc()doc";

static const char *__doc_sgl_Bitmap_PixelFormat_info = R"doc()doc";

static const char *__doc_sgl_Bitmap_PixelFormat_multi_channel = R"doc(Arbitrary multi-channel.)doc";

static const char *__doc_sgl_Bitmap_PixelFormat_r = R"doc(Red.)doc";

static const char *__doc_sgl_Bitmap_PixelFormat_rg = R"doc(Red + green.)doc";

static const char *__doc_sgl_Bitmap_PixelFormat_rgb = R"doc(RGB.)doc";

static const char *__doc_sgl_Bitmap_PixelFormat_rgba = R"doc(RGB + alpha.)doc";

static const char *__doc_sgl_Bitmap_PixelFormat_y = R"doc(Luminance only.)doc";

static const char *__doc_sgl_Bitmap_PixelFormat_ya = R"doc(Luminance + alpha.)doc";

static const char *__doc_sgl_Bitmap_buffer_size = R"doc(The total size of the bitmap in bytes.)doc";

static const char *__doc_sgl_Bitmap_bytes_per_pixel = R"doc(The number of bytes per pixel.)doc";

static const char *__doc_sgl_Bitmap_channel_count = R"doc(The number of channels in the bitmap.)doc";

static const char *__doc_sgl_Bitmap_channel_names = R"doc(The names of the channels in the bitmap.)doc";

static const char *__doc_sgl_Bitmap_check_required_format = R"doc()doc";

static const char *__doc_sgl_Bitmap_class_name = R"doc()doc";

static const char *__doc_sgl_Bitmap_clear = R"doc(Clears the bitmap to zeros.)doc";

static const char *__doc_sgl_Bitmap_component_type = R"doc(The component type.)doc";

static const char *__doc_sgl_Bitmap_convert = R"doc()doc";

static const char *__doc_sgl_Bitmap_convert_2 = R"doc()doc";

static const char *__doc_sgl_Bitmap_data = R"doc(The raw image data.)doc";

static const char *__doc_sgl_Bitmap_data_2 = R"doc()doc";

static const char *__doc_sgl_Bitmap_data_as = R"doc()doc";

static const char *__doc_sgl_Bitmap_data_as_2 = R"doc()doc";

static const char *__doc_sgl_Bitmap_detect_file_format = R"doc()doc";

static const char *__doc_sgl_Bitmap_empty = R"doc(True if bitmap is empty.)doc";

static const char *__doc_sgl_Bitmap_has_alpha = R"doc(Returns true if the bitmap has an alpha channel.)doc";

static const char *__doc_sgl_Bitmap_height = R"doc(The height of the bitmap in pixels.)doc";

static const char *__doc_sgl_Bitmap_m_component_type = R"doc()doc";

static const char *__doc_sgl_Bitmap_m_data = R"doc()doc";

static const char *__doc_sgl_Bitmap_m_height = R"doc()doc";

static const char *__doc_sgl_Bitmap_m_owns_data = R"doc()doc";

static const char *__doc_sgl_Bitmap_m_pixel_format = R"doc()doc";

static const char *__doc_sgl_Bitmap_m_pixel_struct = R"doc()doc";

static const char *__doc_sgl_Bitmap_m_srgb_gamma = R"doc()doc";

static const char *__doc_sgl_Bitmap_m_width = R"doc()doc";

static const char *__doc_sgl_Bitmap_operator_eq = R"doc(Equality operator.)doc";

static const char *__doc_sgl_Bitmap_operator_ne = R"doc(Inequality operator.)doc";

static const char *__doc_sgl_Bitmap_pixel_count = R"doc(The total number of pixels in the bitmap.)doc";

static const char *__doc_sgl_Bitmap_pixel_format = R"doc(The pixel format.)doc";

static const char *__doc_sgl_Bitmap_pixel_struct = R"doc(DataStruct describing the pixel layout.)doc";

static const char *__doc_sgl_Bitmap_read = R"doc()doc";

static const char *__doc_sgl_Bitmap_read_bmp = R"doc()doc";

static const char *__doc_sgl_Bitmap_read_exr = R"doc()doc";

static const char *__doc_sgl_Bitmap_read_hdr = R"doc()doc";

static const char *__doc_sgl_Bitmap_read_jpg = R"doc()doc";

static const char *__doc_sgl_Bitmap_read_multiple =
R"doc(Load a list of bitmaps from multiple paths. Uses multi-threading to
load bitmaps in parallel.)doc";

static const char *__doc_sgl_Bitmap_read_png = R"doc()doc";

static const char *__doc_sgl_Bitmap_read_stb = R"doc()doc";

static const char *__doc_sgl_Bitmap_read_tga = R"doc()doc";

static const char *__doc_sgl_Bitmap_rebuild_pixel_struct = R"doc()doc";

static const char *__doc_sgl_Bitmap_set_srgb_gamma =
R"doc(Set the sRGB gamma flag. Note that this does not convert the pixel
values, it only sets the flag and adjusts the pixel struct.)doc";

static const char *__doc_sgl_Bitmap_split =
R"doc(Split bitmap into multiple bitmaps, each containing the channels with
the same prefix.

For example, if the bitmap has channels `albedo.R`, `albedo.G`,
`albedo.B`, `normal.R`, `normal.G`, `normal.B`, this function will
return two bitmaps, one containing the channels `albedo.R`,
`albedo.G`, `albedo.B` and the other containing the channels
`normal.R`, `normal.G`, `normal.B`.

Common pixel formats (e.g. `y`, `rgb`, `rgba`) are automatically
detected and used for the split bitmaps.

Any channels that do not have a prefix will be returned in the bitmap
with the empty prefix.

Returns:
    Returns a list of (prefix, bitmap) pairs.)doc";

static const char *__doc_sgl_Bitmap_srgb_gamma = R"doc(True if the bitmap is in sRGB gamma space.)doc";

static const char *__doc_sgl_Bitmap_static_init = R"doc()doc";

static const char *__doc_sgl_Bitmap_static_shutdown = R"doc()doc";

static const char *__doc_sgl_Bitmap_to_string = R"doc()doc";

static const char *__doc_sgl_Bitmap_uint8_data = R"doc(The raw image data as uint8_t.)doc";

static const char *__doc_sgl_Bitmap_uint8_data_2 = R"doc()doc";

static const char *__doc_sgl_Bitmap_vflip = R"doc(Vertically flip the bitmap.)doc";

static const char *__doc_sgl_Bitmap_width = R"doc(The width of the bitmap in pixels.)doc";

static const char *__doc_sgl_Bitmap_write = R"doc()doc";

static const char *__doc_sgl_Bitmap_write_2 = R"doc()doc";

static const char *__doc_sgl_Bitmap_write_async = R"doc()doc";

static const char *__doc_sgl_Bitmap_write_bmp = R"doc()doc";

static const char *__doc_sgl_Bitmap_write_exr = R"doc()doc";

static const char *__doc_sgl_Bitmap_write_hdr = R"doc()doc";

static const char *__doc_sgl_Bitmap_write_jpg = R"doc()doc";

static const char *__doc_sgl_Bitmap_write_png = R"doc()doc";

static const char *__doc_sgl_Bitmap_write_tga = R"doc()doc";

static const char *__doc_sgl_BlendFactor = R"doc()doc";

static const char *__doc_sgl_BlendFactor_blend_color = R"doc()doc";

static const char *__doc_sgl_BlendFactor_dest_alpha = R"doc()doc";

static const char *__doc_sgl_BlendFactor_dest_color = R"doc()doc";

static const char *__doc_sgl_BlendFactor_info = R"doc()doc";

static const char *__doc_sgl_BlendFactor_inv_blend_color = R"doc()doc";

static const char *__doc_sgl_BlendFactor_inv_dest_alpha = R"doc()doc";

static const char *__doc_sgl_BlendFactor_inv_dest_color = R"doc()doc";

static const char *__doc_sgl_BlendFactor_inv_secondary_src_alpha = R"doc()doc";

static const char *__doc_sgl_BlendFactor_inv_secondary_src_color = R"doc()doc";

static const char *__doc_sgl_BlendFactor_inv_src_alpha = R"doc()doc";

static const char *__doc_sgl_BlendFactor_inv_src_color = R"doc()doc";

static const char *__doc_sgl_BlendFactor_one = R"doc()doc";

static const char *__doc_sgl_BlendFactor_secondary_src_alpha = R"doc()doc";

static const char *__doc_sgl_BlendFactor_secondary_src_color = R"doc()doc";

static const char *__doc_sgl_BlendFactor_src_alpha = R"doc()doc";

static const char *__doc_sgl_BlendFactor_src_alpha_saturate = R"doc()doc";

static const char *__doc_sgl_BlendFactor_src_color = R"doc()doc";

static const char *__doc_sgl_BlendFactor_zero = R"doc()doc";

static const char *__doc_sgl_BlendOp = R"doc()doc";

static const char *__doc_sgl_BlendOp_add = R"doc()doc";

static const char *__doc_sgl_BlendOp_info = R"doc()doc";

static const char *__doc_sgl_BlendOp_max = R"doc()doc";

static const char *__doc_sgl_BlendOp_min = R"doc()doc";

static const char *__doc_sgl_BlendOp_reverse_subtract = R"doc()doc";

static const char *__doc_sgl_BlendOp_subtract = R"doc()doc";

static const char *__doc_sgl_Blitter =
R"doc(Utility class for blitting textures and generating mipmaps.

Because compute shaders have limited support for UAV texture formats,
this class uses a graphics pipeline to blit a texture to a render
target.)doc";

static const char *__doc_sgl_Blitter_2 = R"doc()doc";

static const char *__doc_sgl_Blitter_Blitter = R"doc()doc";

static const char *__doc_sgl_Blitter_ProgramKey = R"doc()doc";

static const char *__doc_sgl_Blitter_ProgramKey_dst_type = R"doc()doc";

static const char *__doc_sgl_Blitter_ProgramKey_operator_le = R"doc()doc";

static const char *__doc_sgl_Blitter_ProgramKey_src_layout = R"doc()doc";

static const char *__doc_sgl_Blitter_ProgramKey_src_type = R"doc()doc";

static const char *__doc_sgl_Blitter_TextureDataType = R"doc()doc";

static const char *__doc_sgl_Blitter_TextureDataType_float = R"doc()doc";

static const char *__doc_sgl_Blitter_TextureDataType_int = R"doc()doc";

static const char *__doc_sgl_Blitter_TextureLayout = R"doc()doc";

static const char *__doc_sgl_Blitter_TextureLayout_texture_2d = R"doc()doc";

static const char *__doc_sgl_Blitter_TextureLayout_texture_2d_array = R"doc()doc";

static const char *__doc_sgl_Blitter_blit =
R"doc(Blit a texture (SRV) to another texture (RTV).

Blits the full extent of the source texture to the destination
texture.

Parameter ``command_encoder``:
    Command encoder.

Parameter ``dst``:
    View of the destination texture.

Parameter ``src``:
    View of the source texture.

Parameter ``filter``:
    Filtering mode to use.)doc";

static const char *__doc_sgl_Blitter_blit_2 = R"doc()doc";

static const char *__doc_sgl_Blitter_class_name = R"doc()doc";

static const char *__doc_sgl_Blitter_generate_mips =
R"doc(Generate mipmaps for a texture.

Repetatively blits the texture to a smaller mip level until the
smallest mip level is reached. The texture needs to have mip levels
pre-allocated and have usage flags for SRV and RTV. Supports both 2D
and 2D array textures.

Parameter ``command_encoder``:
    Command encoder.

Parameter ``texture``:
    Texture to generate mipmaps for.

Parameter ``layer``:
    Array layer to generate mipmaps for.)doc";

static const char *__doc_sgl_Blitter_get_compute_pipeline = R"doc()doc";

static const char *__doc_sgl_Blitter_get_compute_program = R"doc()doc";

static const char *__doc_sgl_Blitter_get_render_pipeline = R"doc()doc";

static const char *__doc_sgl_Blitter_get_render_program = R"doc()doc";

static const char *__doc_sgl_Blitter_m_compute_pipeline_cache = R"doc()doc";

static const char *__doc_sgl_Blitter_m_compute_program_cache = R"doc()doc";

static const char *__doc_sgl_Blitter_m_device = R"doc()doc";

static const char *__doc_sgl_Blitter_m_linear_sampler = R"doc()doc";

static const char *__doc_sgl_Blitter_m_point_sampler = R"doc()doc";

static const char *__doc_sgl_Blitter_m_render_pipeline_cache = R"doc()doc";

static const char *__doc_sgl_Blitter_m_render_program_cache = R"doc()doc";

static const char *__doc_sgl_Blob = R"doc()doc";

static const char *__doc_sgl_Buffer = R"doc()doc";

static const char *__doc_sgl_Buffer_2 = R"doc()doc";

static const char *__doc_sgl_BufferCursor =
R"doc(Represents a list of elements in a block of memory, and provides
simple interface to get a BufferElementCursor for each one. As this
can be the owner of its data, it is a ref counted object that elements
refer to.)doc";

static const char *__doc_sgl_BufferCursor_2 = R"doc()doc";

static const char *__doc_sgl_BufferCursor_BufferCursor = R"doc()doc";

static const char *__doc_sgl_BufferCursor_BufferCursor_2 =
R"doc(Create with none-owning view of specific block of memory. Number of
elements is inferred from the size of the block and the type layout.)doc";

static const char *__doc_sgl_BufferCursor_BufferCursor_3 =
R"doc(Create buffer + allocate space internally for a given number of
elements.)doc";

static const char *__doc_sgl_BufferCursor_BufferCursor_4 =
R"doc(Create as a view onto a buffer resource. Disable load_before_write to
prevent automatic loading of current buffer state before writing data
to it.)doc";

static const char *__doc_sgl_BufferCursor_BufferCursor_5 =
R"doc(Create as a view onto a section of a buffer resource. Disable
load_before_write to prevent automatic loading of current buffer state
before writing data to it.)doc";

static const char *__doc_sgl_BufferCursor_apply = R"doc(In case of GPU only buffers, pushes all data to the GPU.)doc";

static const char *__doc_sgl_BufferCursor_element_count = R"doc(Number of elements in the buffer.)doc";

static const char *__doc_sgl_BufferCursor_element_size = R"doc(Size of element.)doc";

static const char *__doc_sgl_BufferCursor_element_stride = R"doc(Stride of elements.)doc";

static const char *__doc_sgl_BufferCursor_element_type = R"doc(Get type of an element of the cursor.)doc";

static const char *__doc_sgl_BufferCursor_element_type_layout = R"doc(Get type layout of an element of the cursor.)doc";

static const char *__doc_sgl_BufferCursor_find_element = R"doc(Get element at a given index.)doc";

static const char *__doc_sgl_BufferCursor_is_loaded = R"doc(Check if internal buffer exists.)doc";

static const char *__doc_sgl_BufferCursor_load = R"doc(In case of GPU only buffers, loads all data from GPU.)doc";

static const char *__doc_sgl_BufferCursor_m_buffer = R"doc()doc";

static const char *__doc_sgl_BufferCursor_m_element_type_layout = R"doc()doc";

static const char *__doc_sgl_BufferCursor_m_load_before_write = R"doc()doc";

static const char *__doc_sgl_BufferCursor_m_offset = R"doc()doc";

static const char *__doc_sgl_BufferCursor_m_owner = R"doc()doc";

static const char *__doc_sgl_BufferCursor_m_resource = R"doc()doc";

static const char *__doc_sgl_BufferCursor_m_size = R"doc()doc";

static const char *__doc_sgl_BufferCursor_operator_array = R"doc(Index operator to get element at a given index.)doc";

static const char *__doc_sgl_BufferCursor_read_data = R"doc(Reads data from buffer (note: reads only from host memory).)doc";

static const char *__doc_sgl_BufferCursor_resource = R"doc(Get the resource this cursor represents (if any).)doc";

static const char *__doc_sgl_BufferCursor_size = R"doc(Size of whole buffer.)doc";

static const char *__doc_sgl_BufferCursor_write_data = R"doc(Write data to buffer (note: writes only to host memory).)doc";

static const char *__doc_sgl_BufferDesc = R"doc()doc";

static const char *__doc_sgl_BufferDesc_2 = R"doc()doc";

static const char *__doc_sgl_BufferDesc_data = R"doc(Initial data to upload to the buffer.)doc";

static const char *__doc_sgl_BufferDesc_data_size = R"doc(Size of the initial data in bytes.)doc";

static const char *__doc_sgl_BufferDesc_default_state = R"doc(Initial resource state.)doc";

static const char *__doc_sgl_BufferDesc_element_count =
R"doc(Buffer size in number of struct elements. Can be used instead of
``size``.)doc";

static const char *__doc_sgl_BufferDesc_format = R"doc(Buffer format. Used when creating typed buffer views.)doc";

static const char *__doc_sgl_BufferDesc_label = R"doc(Debug label.)doc";

static const char *__doc_sgl_BufferDesc_memory_type = R"doc(Memory type.)doc";

static const char *__doc_sgl_BufferDesc_resource_type_layout =
R"doc(Resource type layout of the buffer. Can be used instead of
``struct_size`` to specify the size of the struct.)doc";

static const char *__doc_sgl_BufferDesc_size = R"doc(Buffer size in bytes.)doc";

static const char *__doc_sgl_BufferDesc_struct_size = R"doc(Struct size in bytes.)doc";

static const char *__doc_sgl_BufferDesc_usage = R"doc(Resource usage flags.)doc";

static const char *__doc_sgl_BufferElementCursor =
R"doc(Represents a single element of a given type in a block of memory, and
provides read/write tools to access its members via reflection.)doc";

static const char *__doc_sgl_BufferElementCursor_2 = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_BufferElementCursor = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_BufferElementCursor_2 = R"doc(Create with none-owning view of specific block of memory)doc";

static const char *__doc_sgl_BufferElementCursor_as = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_find_element = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_find_field = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_get = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_get_array = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_get_matrix = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_get_scalar = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_get_vector = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_has_element = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_has_field = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_is_valid = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_m_buffer = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_m_offset = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_m_type_layout = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_offset = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_operator_array = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_operator_array_2 = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_operator_assign = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_read_data = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_set = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_set_array = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_set_data = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_set_matrix = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_set_offset = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_set_pointer = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_set_scalar = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_set_vector = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_slang_type_layout = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_to_string = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_type = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_type_layout = R"doc()doc";

static const char *__doc_sgl_BufferElementCursor_write_data = R"doc()doc";

static const char *__doc_sgl_BufferOffsetPair = R"doc()doc";

static const char *__doc_sgl_BufferOffsetPair_BufferOffsetPair = R"doc()doc";

static const char *__doc_sgl_BufferOffsetPair_BufferOffsetPair_2 = R"doc()doc";

static const char *__doc_sgl_BufferOffsetPair_BufferOffsetPair_3 = R"doc()doc";

static const char *__doc_sgl_BufferOffsetPair_buffer = R"doc()doc";

static const char *__doc_sgl_BufferOffsetPair_offset = R"doc()doc";

static const char *__doc_sgl_BufferRange = R"doc()doc";

static const char *__doc_sgl_BufferRange_offset = R"doc()doc";

static const char *__doc_sgl_BufferRange_operator_le = R"doc()doc";

static const char *__doc_sgl_BufferRange_size = R"doc()doc";

static const char *__doc_sgl_BufferRange_to_string = R"doc()doc";

static const char *__doc_sgl_BufferUsage = R"doc()doc";

static const char *__doc_sgl_BufferUsage_acceleration_structure = R"doc()doc";

static const char *__doc_sgl_BufferUsage_acceleration_structure_build_input = R"doc()doc";

static const char *__doc_sgl_BufferUsage_constant_buffer = R"doc()doc";

static const char *__doc_sgl_BufferUsage_copy_destination = R"doc()doc";

static const char *__doc_sgl_BufferUsage_copy_source = R"doc()doc";

static const char *__doc_sgl_BufferUsage_index_buffer = R"doc()doc";

static const char *__doc_sgl_BufferUsage_indirect_argument = R"doc()doc";

static const char *__doc_sgl_BufferUsage_info = R"doc()doc";

static const char *__doc_sgl_BufferUsage_none = R"doc()doc";

static const char *__doc_sgl_BufferUsage_shader_resource = R"doc()doc";

static const char *__doc_sgl_BufferUsage_shader_table = R"doc()doc";

static const char *__doc_sgl_BufferUsage_shared = R"doc()doc";

static const char *__doc_sgl_BufferUsage_unordered_access = R"doc()doc";

static const char *__doc_sgl_BufferUsage_vertex_buffer = R"doc()doc";

static const char *__doc_sgl_BufferView = R"doc()doc";

static const char *__doc_sgl_BufferView_2 = R"doc()doc";

static const char *__doc_sgl_BufferViewDesc = R"doc()doc";

static const char *__doc_sgl_BufferViewDesc_2 = R"doc()doc";

static const char *__doc_sgl_BufferViewDesc_format = R"doc()doc";

static const char *__doc_sgl_BufferViewDesc_label = R"doc()doc";

static const char *__doc_sgl_BufferViewDesc_range = R"doc()doc";

static const char *__doc_sgl_BufferView_BufferView = R"doc()doc";

static const char *__doc_sgl_BufferView_buffer = R"doc()doc";

static const char *__doc_sgl_BufferView_class_name = R"doc()doc";

static const char *__doc_sgl_BufferView_desc = R"doc()doc";

static const char *__doc_sgl_BufferView_format = R"doc()doc";

static const char *__doc_sgl_BufferView_label = R"doc()doc";

static const char *__doc_sgl_BufferView_m_buffer = R"doc()doc";

static const char *__doc_sgl_BufferView_m_desc = R"doc()doc";

static const char *__doc_sgl_BufferView_native_handle = R"doc(Get the native buffer view handle.)doc";

static const char *__doc_sgl_BufferView_range = R"doc()doc";

static const char *__doc_sgl_BufferView_to_string = R"doc()doc";

static const char *__doc_sgl_Buffer_Buffer = R"doc()doc";

static const char *__doc_sgl_Buffer_class_name = R"doc()doc";

static const char *__doc_sgl_Buffer_create_view = R"doc()doc";

static const char *__doc_sgl_Buffer_cuda_memory =
R"doc(Returns a pointer to the CUDA memory. This is only supported if the
buffer was created with ResourceUsage::shared and the device has CUDA
interop enabled.)doc";

static const char *__doc_sgl_Buffer_descriptor_handle_ro = R"doc(Get bindless descriptor handle for read access.)doc";

static const char *__doc_sgl_Buffer_descriptor_handle_rw = R"doc(Get bindless descriptor handle for read-write access.)doc";

static const char *__doc_sgl_Buffer_desc = R"doc()doc";

static const char *__doc_sgl_Buffer_device_address = R"doc()doc";

static const char *__doc_sgl_Buffer_format = R"doc()doc";

static const char *__doc_sgl_Buffer_get_data =
R"doc(Get buffer data to host memory. \note If the buffer is in device local
memory, this will wait until the data is copied back to host memory.

Parameter ``data``:
    Data buffer to read to.

Parameter ``size``:
    Size of the data in bytes.

Parameter ``offset``:
    Offset in the buffer to read from.)doc";

static const char *__doc_sgl_Buffer_get_element = R"doc()doc";

static const char *__doc_sgl_Buffer_get_elements = R"doc()doc";

static const char *__doc_sgl_Buffer_is_mapped = R"doc(Returns true if buffer is currently mapped.)doc";

static const char *__doc_sgl_Buffer_m_cuda_memory = R"doc()doc";

static const char *__doc_sgl_Buffer_m_desc = R"doc()doc";

static const char *__doc_sgl_Buffer_m_mapped_ptr = R"doc()doc";

static const char *__doc_sgl_Buffer_m_rhi_buffer = R"doc()doc";

static const char *__doc_sgl_Buffer_map =
R"doc(Map the whole buffer. Only available for buffers created with
``MemoryType::upload`` or ``MemoryType::read_back``.)doc";

static const char *__doc_sgl_Buffer_map_2 = R"doc()doc";

static const char *__doc_sgl_Buffer_memory_type = R"doc()doc";

static const char *__doc_sgl_Buffer_memory_usage = R"doc()doc";

static const char *__doc_sgl_Buffer_rhi_buffer = R"doc()doc";

static const char *__doc_sgl_Buffer_rhi_resource = R"doc()doc";

static const char *__doc_sgl_Buffer_set_data =
R"doc(Set buffer data from host memory.

Parameter ``data``:
    Data to write.

Parameter ``size``:
    Size of the data in bytes.

Parameter ``offset``:
    Offset in the buffer to write to.)doc";

static const char *__doc_sgl_Buffer_set_element = R"doc()doc";

static const char *__doc_sgl_Buffer_set_elements = R"doc()doc";

static const char *__doc_sgl_Buffer_shared_handle =
R"doc(Get the shared resource handle. Note: Buffer must be created with the
``BufferUsage::shared`` usage flag.)doc";

static const char *__doc_sgl_Buffer_size = R"doc()doc";

static const char *__doc_sgl_Buffer_struct_size = R"doc()doc";

static const char *__doc_sgl_Buffer_to_string = R"doc()doc";

static const char *__doc_sgl_Buffer_unmap = R"doc(Unmap the buffer.)doc";

static const char *__doc_sgl_ColorTargetDesc = R"doc()doc";

static const char *__doc_sgl_ColorTargetDesc_alpha = R"doc()doc";

static const char *__doc_sgl_ColorTargetDesc_color = R"doc()doc";

static const char *__doc_sgl_ColorTargetDesc_enable_blend = R"doc()doc";

static const char *__doc_sgl_ColorTargetDesc_format = R"doc()doc";

static const char *__doc_sgl_ColorTargetDesc_logic_op = R"doc()doc";

static const char *__doc_sgl_ColorTargetDesc_write_mask = R"doc()doc";

static const char *__doc_sgl_CommandBuffer = R"doc()doc";

static const char *__doc_sgl_CommandBuffer_2 = R"doc()doc";

static const char *__doc_sgl_CommandBuffer_CommandBuffer = R"doc()doc";

static const char *__doc_sgl_CommandBuffer_class_name = R"doc()doc";

static const char *__doc_sgl_CommandBuffer_m_cuda_interop_buffers = R"doc()doc";

static const char *__doc_sgl_CommandBuffer_m_rhi_command_buffer = R"doc()doc";

static const char *__doc_sgl_CommandBuffer_rhi_command_buffer = R"doc()doc";

static const char *__doc_sgl_CommandBuffer_to_string = R"doc()doc";

static const char *__doc_sgl_CommandEncoder = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_2 = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_CommandEncoder = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_begin_compute_pass = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_begin_ray_tracing_pass = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_begin_render_pass = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_blit =
R"doc(Blit a texture view.

Blits the full extent of the source texture to the destination
texture.

Parameter ``dst``:
    View of the destination texture.

Parameter ``src``:
    View of the source texture.

Parameter ``filter``:
    Filtering mode to use.)doc";

static const char *__doc_sgl_CommandEncoder_blit_2 =
R"doc(Blit a texture.

Blits the full extent of the source texture to the destination
texture.

Parameter ``dst``:
    Destination texture.

Parameter ``src``:
    Source texture.

Parameter ``filter``:
    Filtering mode to use.)doc";

static const char *__doc_sgl_CommandEncoder_build_acceleration_structure = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_class_name = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_clear_buffer = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_clear_texture_depth_stencil = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_clear_texture_float = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_clear_texture_sint = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_clear_texture_uint = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_copy_acceleration_structure = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_copy_buffer =
R"doc(Copy a buffer region.

Parameter ``dst``:
    Destination buffer.

Parameter ``dst_offset``:
    Destination offset in bytes.

Parameter ``src``:
    Source buffer.

Parameter ``src_offset``:
    Source offset in bytes.

Parameter ``size``:
    Size in bytes.)doc";

static const char *__doc_sgl_CommandEncoder_copy_buffer_to_texture =
R"doc(Copy a buffer to a texture.

Parameter ``dst``:
    Destination texture.

Parameter ``dst_layer``:
    Destination layer.

Parameter ``dst_mip``:
    Destination mip level.

Parameter ``dst_offset``:
    Destination offset in texels.

Parameter ``src``:
    Source buffer.

Parameter ``src_offset``:
    Source offset in bytes.

Parameter ``src_size``:
    Size in bytes.

Parameter ``src_row_pitch``:
    Source row stride in bytes.

Parameter ``extent``:
    Extent in texels (-1 for maximum possible extent).)doc";

static const char *__doc_sgl_CommandEncoder_copy_texture =
R"doc(Copy a texture region.

Parameter ``dst``:
    Destination texture.

Parameter ``dst_subresource_range``:
    Destination subresource range.

Parameter ``dst_offset``:
    Destination offset in texels.

Parameter ``src``:
    Source texture.

Parameter ``src_subresource_range``:
    Source subresource range.

Parameter ``src_offset``:
    Source offset in texels.

Parameter ``extent``:
    Size in texels (-1 for maximum possible size).)doc";

static const char *__doc_sgl_CommandEncoder_copy_texture_2 =
R"doc(Copy a texture region.

Parameter ``dst``:
    Destination texture.

Parameter ``dst_layer``:
    Destination layer.

Parameter ``dst_mip``:
    Destination mip level.

Parameter ``dst_offset``:
    Destination offset in texels.

Parameter ``src``:
    Source texture.

Parameter ``src_layer``:
    Source layer.

Parameter ``src_mip``:
    Source mip level.

Parameter ``src_offset``:
    Source offset in texels.

Parameter ``extent``:
    Size in texels (-1 for maximum possible size).)doc";

static const char *__doc_sgl_CommandEncoder_copy_texture_to_buffer =
R"doc(Copy a texture to a buffer.

Parameter ``dst``:
    Destination buffer.

Parameter ``dst_offset``:
    Destination offset in bytes.

Parameter ``dst_size``:
    Destination size in bytes.

Parameter ``dst_row_pitch``:
    Destination row stride in bytes.

Parameter ``src``:
    Source texture.

Parameter ``src_layer``:
    Source layer.

Parameter ``src_mip``:
    Source mip level.

Parameter ``src_offset``:
    Source offset in texels.

Parameter ``extent``:
    Extent in texels (-1 for maximum possible extent).)doc";

static const char *__doc_sgl_CommandEncoder_deserialize_acceleration_structure = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_finish = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_generate_mips = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_get_root_object = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_global_barrier =
R"doc(Insert a global barrier that ensures all previous writes are visible
to subsequent reads. Note: This is not necessary for typical bindings,
as state management is automatic, however global barriers are useful
for cross-api synchronization (eg 2 slangpy devices constructed from
the same native handle), or as brute force tools for synchronizing
pointer/bindless operations.)doc";

static const char *__doc_sgl_CommandEncoder_insert_debug_marker =
R"doc(Insert a debug marker.

Parameter ``name``:
    Name of the marker.

Parameter ``color``:
    Color of the marker.)doc";

static const char *__doc_sgl_CommandEncoder_m_compute_pass_encoder = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_m_cuda_interop_buffers = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_m_open = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_m_ray_tracing_pass_encoder = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_m_render_pass_encoder = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_m_rhi_command_encoder = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_m_root_object = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_native_handle = R"doc(Get the command encoder handle.)doc";

static const char *__doc_sgl_CommandEncoder_pop_debug_group = R"doc(Pop a debug group.)doc";

static const char *__doc_sgl_CommandEncoder_push_debug_group = R"doc(Push a debug group.)doc";

static const char *__doc_sgl_CommandEncoder_query_acceleration_structure_properties = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_resolve_query = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_rhi_command_encoder = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_serialize_acceleration_structure = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_set_buffer_state =
R"doc(Transition resource state of a buffer and add a barrier if state has
changed.

Parameter ``buffer``:
    Buffer

Parameter ``state``:
    New state)doc";

static const char *__doc_sgl_CommandEncoder_set_texture_state =
R"doc(Transition resource state of a texture and add a barrier if state has
changed.

Parameter ``texture``:
    Texture

Parameter ``state``:
    New state)doc";

static const char *__doc_sgl_CommandEncoder_set_texture_state_2 =
R"doc(Transition resource state of texture sub-resources and add a barrier
if state has changed.

Parameter ``texture``:
    Texture

Parameter ``range``:
    Subresource range

Parameter ``state``:
    New state)doc";

static const char *__doc_sgl_CommandEncoder_to_string = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_upload_buffer_data =
R"doc(Upload host memory to a buffer.

Parameter ``buffer``:
    Buffer to write to.

Parameter ``offset``:
    Buffer offset in bytes.

Parameter ``size``:
    Number of bytes to write.

Parameter ``data``:
    Host memory to write.)doc";

static const char *__doc_sgl_CommandEncoder_upload_texture_data = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_upload_texture_data_2 = R"doc()doc";

static const char *__doc_sgl_CommandEncoder_write_timestamp =
R"doc(Write a timestamp.

Parameter ``query_pool``:
    Query pool.

Parameter ``index``:
    Index of the query.)doc";

static const char *__doc_sgl_CommandQueue = R"doc()doc";

static const char *__doc_sgl_CommandQueueType = R"doc()doc";

static const char *__doc_sgl_CommandQueueType_graphics = R"doc()doc";

static const char *__doc_sgl_CommandQueueType_info = R"doc()doc";

static const char *__doc_sgl_ComparisonFunc = R"doc()doc";

static const char *__doc_sgl_ComparisonFunc_always = R"doc()doc";

static const char *__doc_sgl_ComparisonFunc_equal = R"doc()doc";

static const char *__doc_sgl_ComparisonFunc_greater = R"doc()doc";

static const char *__doc_sgl_ComparisonFunc_greater_equal = R"doc()doc";

static const char *__doc_sgl_ComparisonFunc_info = R"doc()doc";

static const char *__doc_sgl_ComparisonFunc_less = R"doc()doc";

static const char *__doc_sgl_ComparisonFunc_less_equal = R"doc()doc";

static const char *__doc_sgl_ComparisonFunc_never = R"doc()doc";

static const char *__doc_sgl_ComparisonFunc_not_equal = R"doc()doc";

static const char *__doc_sgl_ComputeKernel = R"doc()doc";

static const char *__doc_sgl_ComputeKernel_2 = R"doc()doc";

static const char *__doc_sgl_ComputeKernelDesc = R"doc()doc";

static const char *__doc_sgl_ComputeKernelDesc_2 = R"doc()doc";

static const char *__doc_sgl_ComputeKernelDesc_program = R"doc()doc";

static const char *__doc_sgl_ComputeKernel_ComputeKernel = R"doc()doc";

static const char *__doc_sgl_ComputeKernel_class_name = R"doc()doc";

static const char *__doc_sgl_ComputeKernel_dispatch = R"doc()doc";

static const char *__doc_sgl_ComputeKernel_m_pipeline = R"doc()doc";

static const char *__doc_sgl_ComputeKernel_m_thread_group_size = R"doc()doc";

static const char *__doc_sgl_ComputeKernel_pipeline = R"doc()doc";

static const char *__doc_sgl_ComputePassEncoder = R"doc()doc";

static const char *__doc_sgl_ComputePassEncoder_2 = R"doc()doc";

static const char *__doc_sgl_ComputePassEncoder_bind_pipeline = R"doc()doc";

static const char *__doc_sgl_ComputePassEncoder_bind_pipeline_2 = R"doc()doc";

static const char *__doc_sgl_ComputePassEncoder_class_name = R"doc()doc";

static const char *__doc_sgl_ComputePassEncoder_dispatch = R"doc()doc";

static const char *__doc_sgl_ComputePassEncoder_dispatch_compute = R"doc()doc";

static const char *__doc_sgl_ComputePassEncoder_dispatch_compute_indirect = R"doc()doc";

static const char *__doc_sgl_ComputePassEncoder_end = R"doc()doc";

static const char *__doc_sgl_ComputePassEncoder_m_rhi_compute_pass_encoder = R"doc()doc";

static const char *__doc_sgl_ComputePassEncoder_m_thread_group_size = R"doc()doc";

static const char *__doc_sgl_ComputePipeline = R"doc()doc";

static const char *__doc_sgl_ComputePipeline_2 = R"doc(Compute pipeline.)doc";

static const char *__doc_sgl_ComputePipelineDesc = R"doc()doc";

static const char *__doc_sgl_ComputePipelineDesc_2 = R"doc()doc";

static const char *__doc_sgl_ComputePipelineDesc_label = R"doc()doc";

static const char *__doc_sgl_ComputePipelineDesc_program = R"doc()doc";

static const char *__doc_sgl_ComputePipeline_ComputePipeline = R"doc()doc";

static const char *__doc_sgl_ComputePipeline_desc = R"doc()doc";

static const char *__doc_sgl_ComputePipeline_m_desc = R"doc()doc";

static const char *__doc_sgl_ComputePipeline_m_rhi_pipeline = R"doc()doc";

static const char *__doc_sgl_ComputePipeline_m_thread_group_size = R"doc()doc";

static const char *__doc_sgl_ComputePipeline_native_handle = R"doc(Get the native pipeline handle.)doc";

static const char *__doc_sgl_ComputePipeline_recreate = R"doc()doc";

static const char *__doc_sgl_ComputePipeline_rhi_pipeline = R"doc()doc";

static const char *__doc_sgl_ComputePipeline_thread_group_size =
R"doc(Thread group size. Used to determine the number of thread groups to
dispatch.)doc";

static const char *__doc_sgl_ComputePipeline_to_string = R"doc()doc";

static const char *__doc_sgl_ConsoleLoggerOutput = R"doc()doc";

static const char *__doc_sgl_ConsoleLoggerOutput_2 =
R"doc(Logger output that writes to the console. Error messages are printed
to stderr, all other messages to stdout. Messages are optionally
colored.)doc";

static const char *__doc_sgl_ConsoleLoggerOutput_ConsoleLoggerOutput = R"doc()doc";

static const char *__doc_sgl_ConsoleLoggerOutput_enable_ansi_control_sequences = R"doc()doc";

static const char *__doc_sgl_ConsoleLoggerOutput_m_colored = R"doc()doc";

static const char *__doc_sgl_ConsoleLoggerOutput_to_string = R"doc()doc";

static const char *__doc_sgl_ConsoleLoggerOutput_write = R"doc()doc";

static const char *__doc_sgl_ConstructorRefGuard =
R"doc(Helper class to protect objects from being deleted before the
constructor has finished.

When objects are constructed, they have a reference count of zero. If
during the constructor, a reference to the object is taken and later
released, the object will be deleted before the constructor has
finished.

This helper class will take the initial reference to the object during
construction, and avoid any other references to the object from
deallocating it before the constructor has finished.

Template parameter ``T``:
    Object type.)doc";

static const char *__doc_sgl_ConstructorRefGuard_ConstructorRefGuard = R"doc()doc";

static const char *__doc_sgl_ConstructorRefGuard_ConstructorRefGuard_2 = R"doc()doc";

static const char *__doc_sgl_ConstructorRefGuard_ConstructorRefGuard_3 = R"doc()doc";

static const char *__doc_sgl_ConstructorRefGuard_m_obj = R"doc()doc";

static const char *__doc_sgl_ConstructorRefGuard_operator_assign = R"doc()doc";

static const char *__doc_sgl_ConstructorRefGuard_operator_assign_2 = R"doc()doc";

static const char *__doc_sgl_CoopVec = R"doc()doc";

static const char *__doc_sgl_CoopVec_2 = R"doc()doc";

static const char *__doc_sgl_CoopVecMatrixDesc = R"doc()doc";

static const char *__doc_sgl_CoopVecMatrixDesc_cols = R"doc()doc";

static const char *__doc_sgl_CoopVecMatrixDesc_element_type = R"doc()doc";

static const char *__doc_sgl_CoopVecMatrixDesc_layout = R"doc()doc";

static const char *__doc_sgl_CoopVecMatrixDesc_offset = R"doc()doc";

static const char *__doc_sgl_CoopVecMatrixDesc_rows = R"doc()doc";

static const char *__doc_sgl_CoopVecMatrixDesc_size = R"doc()doc";

static const char *__doc_sgl_CoopVecMatrixLayout = R"doc()doc";

static const char *__doc_sgl_CoopVecMatrixLayout_column_major = R"doc()doc";

static const char *__doc_sgl_CoopVecMatrixLayout_inferencing_optimal = R"doc()doc";

static const char *__doc_sgl_CoopVecMatrixLayout_info = R"doc()doc";

static const char *__doc_sgl_CoopVecMatrixLayout_row_major = R"doc()doc";

static const char *__doc_sgl_CoopVecMatrixLayout_training_optimal = R"doc()doc";

static const char *__doc_sgl_CoopVec_CoopVec = R"doc()doc";

static const char *__doc_sgl_CoopVec_align_matrix_offset = R"doc()doc";

static const char *__doc_sgl_CoopVec_align_vector_offset = R"doc()doc";

static const char *__doc_sgl_CoopVec_class_name = R"doc()doc";

static const char *__doc_sgl_CoopVec_convert_matrix_device = R"doc()doc";

static const char *__doc_sgl_CoopVec_convert_matrix_device_2 = R"doc()doc";

static const char *__doc_sgl_CoopVec_convert_matrix_device_3 = R"doc()doc";

static const char *__doc_sgl_CoopVec_convert_matrix_host = R"doc()doc";

static const char *__doc_sgl_CoopVec_create_matrix_desc = R"doc()doc";

static const char *__doc_sgl_CoopVec_m_device = R"doc()doc";

static const char *__doc_sgl_CoopVec_query_matrix_size = R"doc()doc";

static const char *__doc_sgl_CullMode = R"doc()doc";

static const char *__doc_sgl_CullMode_back = R"doc()doc";

static const char *__doc_sgl_CullMode_front = R"doc()doc";

static const char *__doc_sgl_CullMode_info = R"doc()doc";

static const char *__doc_sgl_CullMode_none = R"doc()doc";

static const char *__doc_sgl_CursorMode = R"doc(Mouse cursor modes.)doc";

static const char *__doc_sgl_CursorMode_disabled = R"doc(The cursor is hidden and locked to the window.)doc";

static const char *__doc_sgl_CursorMode_hidden = R"doc(The cursor is hidden when over the window.)doc";

static const char *__doc_sgl_CursorMode_info = R"doc()doc";

static const char *__doc_sgl_CursorMode_normal = R"doc(The cursor is visible and behaves normally.)doc";

static const char *__doc_sgl_DDSFile = R"doc(Helper class for loading DDS files.)doc";

static const char *__doc_sgl_DDSFile_DDSFile = R"doc()doc";

static const char *__doc_sgl_DDSFile_DDSFile_2 = R"doc()doc";

static const char *__doc_sgl_DDSFile_DDSFile_3 = R"doc()doc";

static const char *__doc_sgl_DDSFile_DDSFile_4 = R"doc()doc";

static const char *__doc_sgl_DDSFile_TextureType = R"doc()doc";

static const char *__doc_sgl_DDSFile_TextureType_info = R"doc()doc";

static const char *__doc_sgl_DDSFile_TextureType_texture_1d = R"doc()doc";

static const char *__doc_sgl_DDSFile_TextureType_texture_2d = R"doc()doc";

static const char *__doc_sgl_DDSFile_TextureType_texture_3d = R"doc()doc";

static const char *__doc_sgl_DDSFile_TextureType_texture_cube = R"doc()doc";

static const char *__doc_sgl_DDSFile_array_size = R"doc()doc";

static const char *__doc_sgl_DDSFile_bits_per_pixel_or_block = R"doc()doc";

static const char *__doc_sgl_DDSFile_block_height = R"doc()doc";

static const char *__doc_sgl_DDSFile_block_width = R"doc()doc";

static const char *__doc_sgl_DDSFile_class_name = R"doc()doc";

static const char *__doc_sgl_DDSFile_compressed = R"doc()doc";

static const char *__doc_sgl_DDSFile_data = R"doc()doc";

static const char *__doc_sgl_DDSFile_decode_header = R"doc()doc";

static const char *__doc_sgl_DDSFile_depth = R"doc()doc";

static const char *__doc_sgl_DDSFile_detect_dds_file = R"doc()doc";

static const char *__doc_sgl_DDSFile_dxgi_format = R"doc()doc";

static const char *__doc_sgl_DDSFile_get_subresource_data =
R"doc(Get a pointer to the start of the data for the specified mip and
slice.

Parameter ``mip``:
    Mip level.

Parameter ``slice``:
    Slice index (array index, face index or volume slice index).

Returns:
    Pointer to the start of the data.)doc";

static const char *__doc_sgl_DDSFile_get_subresource_pitch = R"doc()doc";

static const char *__doc_sgl_DDSFile_height = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_array_size = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_bits_per_pixel_or_block = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_block_height = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_block_width = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_compressed = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_data = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_depth = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_dxgi_format = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_header_size = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_height = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_mip_count = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_row_pitch = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_size = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_slice_pitch = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_srgb = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_type = R"doc()doc";

static const char *__doc_sgl_DDSFile_m_width = R"doc()doc";

static const char *__doc_sgl_DDSFile_mip_count = R"doc()doc";

static const char *__doc_sgl_DDSFile_operator_assign = R"doc()doc";

static const char *__doc_sgl_DDSFile_operator_assign_2 = R"doc()doc";

static const char *__doc_sgl_DDSFile_resource_data = R"doc()doc";

static const char *__doc_sgl_DDSFile_resource_size = R"doc()doc";

static const char *__doc_sgl_DDSFile_row_pitch = R"doc()doc";

static const char *__doc_sgl_DDSFile_size = R"doc()doc";

static const char *__doc_sgl_DDSFile_slice_pitch = R"doc()doc";

static const char *__doc_sgl_DDSFile_srgb = R"doc()doc";

static const char *__doc_sgl_DDSFile_to_string = R"doc()doc";

static const char *__doc_sgl_DDSFile_type = R"doc()doc";

static const char *__doc_sgl_DDSFile_width = R"doc()doc";

static const char *__doc_sgl_DataStruct =
R"doc(Structured data definition.

This class is used to describe a structured data type layout. It is
used by the DataStructConverter class to convert between different
layouts.)doc";

static const char *__doc_sgl_DataStruct_2 = R"doc()doc";

static const char *__doc_sgl_DataStructConverter =
R"doc(Data struct converter.

This helper class can be used to convert between structs with
different layouts.)doc";

static const char *__doc_sgl_DataStructConverter_2 = R"doc()doc";

static const char *__doc_sgl_DataStructConverter_DataStructConverter =
R"doc(Constructor.

Parameter ``src``:
    Source struct definition.

Parameter ``dst``:
    Destination struct definition.)doc";

static const char *__doc_sgl_DataStructConverter_class_name = R"doc()doc";

static const char *__doc_sgl_DataStructConverter_convert =
R"doc(Convert data from source struct to destination struct.

Parameter ``src``:
    Source data.

Parameter ``dst``:
    Destination data.

Parameter ``count``:
    Number of structs to convert.)doc";

static const char *__doc_sgl_DataStructConverter_dst = R"doc(The destination struct definition.)doc";

static const char *__doc_sgl_DataStructConverter_m_dst = R"doc()doc";

static const char *__doc_sgl_DataStructConverter_m_src = R"doc()doc";

static const char *__doc_sgl_DataStructConverter_src = R"doc(The source struct definition.)doc";

static const char *__doc_sgl_DataStructConverter_to_string = R"doc()doc";

static const char *__doc_sgl_DataStruct_ByteOrder = R"doc(Byte order.)doc";

static const char *__doc_sgl_DataStruct_ByteOrder_big_endian = R"doc()doc";

static const char *__doc_sgl_DataStruct_ByteOrder_host = R"doc()doc";

static const char *__doc_sgl_DataStruct_ByteOrder_info = R"doc()doc";

static const char *__doc_sgl_DataStruct_ByteOrder_little_endian = R"doc()doc";

static const char *__doc_sgl_DataStruct_DataStruct =
R"doc(Constructor.

Parameter ``pack``:
    If true, the struct will be packed.

Parameter ``byte_order``:
    Byte order of the struct.)doc";

static const char *__doc_sgl_DataStruct_Field = R"doc(Struct field.)doc";

static const char *__doc_sgl_DataStruct_Field_blend =
R"doc(List of blend weights and names. If set, the ``DataStructConverter``
will blend fields from the source struct with the specified weights to
generate the destination field. Blending is done in linear space.)doc";

static const char *__doc_sgl_DataStruct_Field_default_value = R"doc(Default value.)doc";

static const char *__doc_sgl_DataStruct_Field_flags = R"doc(Field flags.)doc";

static const char *__doc_sgl_DataStruct_Field_is_float = R"doc(Check if the field is a floating point type.)doc";

static const char *__doc_sgl_DataStruct_Field_is_integer = R"doc(Check if the field is an integer type.)doc";

static const char *__doc_sgl_DataStruct_Field_is_signed = R"doc(Check if the field is a signed type.)doc";

static const char *__doc_sgl_DataStruct_Field_is_unsigned = R"doc(Check if the field is an unsigned type.)doc";

static const char *__doc_sgl_DataStruct_Field_name = R"doc(Name of the field.)doc";

static const char *__doc_sgl_DataStruct_Field_offset = R"doc(Offset of the field in bytes.)doc";

static const char *__doc_sgl_DataStruct_Field_operator_eq = R"doc(Equality operator.)doc";

static const char *__doc_sgl_DataStruct_Field_operator_ne = R"doc(Inequality operator.)doc";

static const char *__doc_sgl_DataStruct_Field_size = R"doc(Size of the field in bytes.)doc";

static const char *__doc_sgl_DataStruct_Field_to_string = R"doc()doc";

static const char *__doc_sgl_DataStruct_Field_type = R"doc(Type of the field.)doc";

static const char *__doc_sgl_DataStruct_Flags = R"doc(Struct field flags.)doc";

static const char *__doc_sgl_DataStruct_Flags_default =
R"doc(Use default when field is not present in the source struct during
conversion.)doc";

static const char *__doc_sgl_DataStruct_Flags_info = R"doc()doc";

static const char *__doc_sgl_DataStruct_Flags_none = R"doc()doc";

static const char *__doc_sgl_DataStruct_Flags_normalized = R"doc(Integer fields represent normalized values in the range [0, 1].)doc";

static const char *__doc_sgl_DataStruct_Flags_srgb_gamma = R"doc(Field encodes an sRGB gamma-corrected value.)doc";

static const char *__doc_sgl_DataStruct_Type = R"doc(Struct field type.)doc";

static const char *__doc_sgl_DataStruct_Type_float16 = R"doc()doc";

static const char *__doc_sgl_DataStruct_Type_float32 = R"doc()doc";

static const char *__doc_sgl_DataStruct_Type_float64 = R"doc()doc";

static const char *__doc_sgl_DataStruct_Type_info = R"doc()doc";

static const char *__doc_sgl_DataStruct_Type_int16 = R"doc()doc";

static const char *__doc_sgl_DataStruct_Type_int32 = R"doc()doc";

static const char *__doc_sgl_DataStruct_Type_int64 = R"doc()doc";

static const char *__doc_sgl_DataStruct_Type_int8 = R"doc()doc";

static const char *__doc_sgl_DataStruct_Type_uint16 = R"doc()doc";

static const char *__doc_sgl_DataStruct_Type_uint32 = R"doc()doc";

static const char *__doc_sgl_DataStruct_Type_uint64 = R"doc()doc";

static const char *__doc_sgl_DataStruct_Type_uint8 = R"doc()doc";

static const char *__doc_sgl_DataStruct_alignment = R"doc(The alignment of the struct in bytes.)doc";

static const char *__doc_sgl_DataStruct_append = R"doc(Append a field to the struct.)doc";

static const char *__doc_sgl_DataStruct_append_2 =
R"doc(Append a field to the struct.

Parameter ``name``:
    Name of the field.

Parameter ``type``:
    Type of the field.

Parameter ``flags``:
    Field flags.

Parameter ``default_value``:
    Default value.

Parameter ``blend``:
    List of blend weights/names.

Returns:
    Reference to the struct.)doc";

static const char *__doc_sgl_DataStruct_begin = R"doc(Field iterator (begin).)doc";

static const char *__doc_sgl_DataStruct_begin_2 = R"doc(Const field iterator (begin).)doc";

static const char *__doc_sgl_DataStruct_byte_order = R"doc(The byte order of the struct.)doc";

static const char *__doc_sgl_DataStruct_class_name = R"doc()doc";

static const char *__doc_sgl_DataStruct_end = R"doc(Field iterator (end).)doc";

static const char *__doc_sgl_DataStruct_end_2 = R"doc(Const field iterator (end).)doc";

static const char *__doc_sgl_DataStruct_field = R"doc(Access field by name. Throws if field is not found.)doc";

static const char *__doc_sgl_DataStruct_field_2 = R"doc(Access field by name. Throws if field is not found.)doc";

static const char *__doc_sgl_DataStruct_field_count = R"doc(The number of fields in the struct.)doc";

static const char *__doc_sgl_DataStruct_has_field = R"doc(Check if a field with the specified name exists.)doc";

static const char *__doc_sgl_DataStruct_host_byte_order = R"doc(The host byte order.)doc";

static const char *__doc_sgl_DataStruct_is_float = R"doc(Check if ``type`` is a floating point type.)doc";

static const char *__doc_sgl_DataStruct_is_integer = R"doc(Check if ``type`` is an integer type.)doc";

static const char *__doc_sgl_DataStruct_is_signed = R"doc(Check if ``type`` is a signed type.)doc";

static const char *__doc_sgl_DataStruct_is_unsigned = R"doc(Check if ``type`` is an unsigned type.)doc";

static const char *__doc_sgl_DataStruct_m_byte_order = R"doc()doc";

static const char *__doc_sgl_DataStruct_m_fields = R"doc()doc";

static const char *__doc_sgl_DataStruct_m_pack = R"doc()doc";

static const char *__doc_sgl_DataStruct_operator_array = R"doc(Access field by index.)doc";

static const char *__doc_sgl_DataStruct_operator_array_2 = R"doc(Access field by index.)doc";

static const char *__doc_sgl_DataStruct_operator_eq = R"doc(Equality operator.)doc";

static const char *__doc_sgl_DataStruct_operator_ne = R"doc(Inequality operator.)doc";

static const char *__doc_sgl_DataStruct_size = R"doc(The size of the struct in bytes (with padding).)doc";

static const char *__doc_sgl_DataStruct_to_string = R"doc()doc";

static const char *__doc_sgl_DataStruct_type_range = R"doc(Get the numeric range of a type.)doc";

static const char *__doc_sgl_DataStruct_type_size = R"doc(Get the size of a type in bytes.)doc";

static const char *__doc_sgl_DataType = R"doc()doc";

static const char *__doc_sgl_DataType_bool = R"doc()doc";

static const char *__doc_sgl_DataType_float16 = R"doc()doc";

static const char *__doc_sgl_DataType_float32 = R"doc()doc";

static const char *__doc_sgl_DataType_float64 = R"doc()doc";

static const char *__doc_sgl_DataType_info = R"doc()doc";

static const char *__doc_sgl_DataType_int16 = R"doc()doc";

static const char *__doc_sgl_DataType_int32 = R"doc()doc";

static const char *__doc_sgl_DataType_int64 = R"doc()doc";

static const char *__doc_sgl_DataType_int8 = R"doc()doc";

static const char *__doc_sgl_DataType_uint16 = R"doc()doc";

static const char *__doc_sgl_DataType_uint32 = R"doc()doc";

static const char *__doc_sgl_DataType_uint64 = R"doc()doc";

static const char *__doc_sgl_DataType_uint8 = R"doc()doc";

static const char *__doc_sgl_DataType_void = R"doc()doc";

static const char *__doc_sgl_DebugConsoleLoggerOutput = R"doc()doc";

static const char *__doc_sgl_DebugConsoleLoggerOutput_2 = R"doc(Logger output that writes to the debug console (Windows only).)doc";

static const char *__doc_sgl_DebugConsoleLoggerOutput_to_string = R"doc()doc";

static const char *__doc_sgl_DebugConsoleLoggerOutput_write = R"doc()doc";

static const char *__doc_sgl_DebugLogger = R"doc()doc";

static const char *__doc_sgl_DebugLogger_DebugLogger = R"doc()doc";

static const char *__doc_sgl_DebugLogger_get = R"doc()doc";

static const char *__doc_sgl_DebugLogger_get_messages = R"doc()doc";

static const char *__doc_sgl_DebugLogger_handleMessage = R"doc()doc";

static const char *__doc_sgl_DebugLogger_m_logger = R"doc()doc";

static const char *__doc_sgl_DebugLogger_m_message_count = R"doc()doc";

static const char *__doc_sgl_DebugLogger_m_messages = R"doc()doc";

static const char *__doc_sgl_DebugLogger_m_mutex = R"doc()doc";

static const char *__doc_sgl_DebugLogger_message_count = R"doc()doc";

static const char *__doc_sgl_DebugPrinter = R"doc()doc";

static const char *__doc_sgl_DebugPrinter_2 =
R"doc(Debug printer.

This class implements host-side support for shader debug printing.)doc";

static const char *__doc_sgl_DebugPrinter_DebugPrinter = R"doc()doc";

static const char *__doc_sgl_DebugPrinter_add_hashed_strings =
R"doc(Add a map of hashed strings to the printer. This needs to be called
for any shader that uses debug printing.)doc";

static const char *__doc_sgl_DebugPrinter_bind = R"doc()doc";

static const char *__doc_sgl_DebugPrinter_flush = R"doc(Flush the print buffer and output any messages to stdout.)doc";

static const char *__doc_sgl_DebugPrinter_flush_device = R"doc()doc";

static const char *__doc_sgl_DebugPrinter_flush_to_string = R"doc(Flush the print buffer and output any messages as a string.)doc";

static const char *__doc_sgl_DebugPrinter_m_buffer = R"doc()doc";

static const char *__doc_sgl_DebugPrinter_m_device = R"doc()doc";

static const char *__doc_sgl_DebugPrinter_m_hashed_strings = R"doc()doc";

static const char *__doc_sgl_DebugPrinter_m_readback_buffer = R"doc()doc";

static const char *__doc_sgl_DeclReflection = R"doc()doc";

static const char *__doc_sgl_DeclReflection_2 = R"doc()doc";

static const char *__doc_sgl_DeclReflectionChildList = R"doc()doc";

static const char *__doc_sgl_DeclReflectionChildList_2 = R"doc(DeclReflection lazy child list evaluation.)doc";

static const char *__doc_sgl_DeclReflectionChildList_DeclReflectionChildList = R"doc()doc";

static const char *__doc_sgl_DeclReflectionChildList_evaluate = R"doc(Get a specific search result.)doc";

static const char *__doc_sgl_DeclReflectionChildList_size = R"doc(Number of entries in list.)doc";

static const char *__doc_sgl_DeclReflectionIndexedChildList = R"doc()doc";

static const char *__doc_sgl_DeclReflectionIndexedChildList_2 = R"doc(DeclReflection lazy search result evaluation.)doc";

static const char *__doc_sgl_DeclReflectionIndexedChildList_DeclReflectionIndexedChildList = R"doc()doc";

static const char *__doc_sgl_DeclReflectionIndexedChildList_evaluate = R"doc(Get a specific search result.)doc";

static const char *__doc_sgl_DeclReflection_DeclReflection = R"doc()doc";

static const char *__doc_sgl_DeclReflection_Kind = R"doc(Different kinds of decl slang can return.)doc";

static const char *__doc_sgl_DeclReflection_Kind_func = R"doc()doc";

static const char *__doc_sgl_DeclReflection_Kind_generic = R"doc()doc";

static const char *__doc_sgl_DeclReflection_Kind_info = R"doc()doc";

static const char *__doc_sgl_DeclReflection_Kind_module = R"doc()doc";

static const char *__doc_sgl_DeclReflection_Kind_struct = R"doc()doc";

static const char *__doc_sgl_DeclReflection_Kind_unsupported = R"doc()doc";

static const char *__doc_sgl_DeclReflection_Kind_variable = R"doc()doc";

static const char *__doc_sgl_DeclReflection_as_function = R"doc(Get function corresponding to this decl ref.)doc";

static const char *__doc_sgl_DeclReflection_as_type = R"doc(Get type corresponding to this decl ref.)doc";

static const char *__doc_sgl_DeclReflection_as_variable = R"doc(Get variable corresponding to this decl ref.)doc";

static const char *__doc_sgl_DeclReflection_child = R"doc(Get a child by index.)doc";

static const char *__doc_sgl_DeclReflection_child_count = R"doc(Get number of children.)doc";

static const char *__doc_sgl_DeclReflection_children = R"doc(List of children of this cursor.)doc";

static const char *__doc_sgl_DeclReflection_children_of_kind = R"doc(List of children of this cursor of a specific kind.)doc";

static const char *__doc_sgl_DeclReflection_find_children_of_kind =
R"doc(Finds all children of a specific kind with a given name. Note: Only
supported for types, functions and variables.)doc";

static const char *__doc_sgl_DeclReflection_find_first_child_of_kind =
R"doc(Finds the first child of a specific kind with a given name. Note: Only
supported for types, functions and variables.)doc";

static const char *__doc_sgl_DeclReflection_kind = R"doc(Decl kind (struct/function/module/generic/variable).)doc";

static const char *__doc_sgl_DeclReflection_name =
R"doc(Get the name of this decl (if it is of a kind that has a name). Note:
Only supported for types, functions and variables.)doc";

static const char *__doc_sgl_DeclReflection_operator_array = R"doc(Index operator to get nth child.)doc";

static const char *__doc_sgl_DeclReflection_to_string = R"doc(Description as string.)doc";

static const char *__doc_sgl_DepthStencilDesc = R"doc()doc";

static const char *__doc_sgl_DepthStencilDesc_back_face = R"doc()doc";

static const char *__doc_sgl_DepthStencilDesc_depth_func = R"doc()doc";

static const char *__doc_sgl_DepthStencilDesc_depth_test_enable = R"doc()doc";

static const char *__doc_sgl_DepthStencilDesc_depth_write_enable = R"doc()doc";

static const char *__doc_sgl_DepthStencilDesc_format = R"doc()doc";

static const char *__doc_sgl_DepthStencilDesc_front_face = R"doc()doc";

static const char *__doc_sgl_DepthStencilDesc_stencil_enable = R"doc()doc";

static const char *__doc_sgl_DepthStencilDesc_stencil_read_mask = R"doc()doc";

static const char *__doc_sgl_DepthStencilDesc_stencil_write_mask = R"doc()doc";

static const char *__doc_sgl_DepthStencilOpDesc = R"doc()doc";

static const char *__doc_sgl_DepthStencilOpDesc_stencil_depth_fail_op = R"doc()doc";

static const char *__doc_sgl_DepthStencilOpDesc_stencil_fail_op = R"doc()doc";

static const char *__doc_sgl_DepthStencilOpDesc_stencil_func = R"doc()doc";

static const char *__doc_sgl_DepthStencilOpDesc_stencil_pass_op = R"doc()doc";

static const char *__doc_sgl_DescriptorHandle = R"doc()doc";

static const char *__doc_sgl_DescriptorHandleType = R"doc()doc";

static const char *__doc_sgl_DescriptorHandleType_acceleration_structure = R"doc()doc";

static const char *__doc_sgl_DescriptorHandleType_buffer = R"doc()doc";

static const char *__doc_sgl_DescriptorHandleType_info = R"doc()doc";

static const char *__doc_sgl_DescriptorHandleType_rw_buffer = R"doc()doc";

static const char *__doc_sgl_DescriptorHandleType_rw_texture = R"doc()doc";

static const char *__doc_sgl_DescriptorHandleType_sampler = R"doc()doc";

static const char *__doc_sgl_DescriptorHandleType_texture = R"doc()doc";

static const char *__doc_sgl_DescriptorHandleType_undefined = R"doc()doc";

static const char *__doc_sgl_DescriptorHandle_DescriptorHandle = R"doc()doc";

static const char *__doc_sgl_DescriptorHandle_is_valid = R"doc()doc";

static const char *__doc_sgl_DescriptorHandle_operator_bool = R"doc()doc";

static const char *__doc_sgl_DescriptorHandle_to_string = R"doc()doc";

static const char *__doc_sgl_DescriptorHandle_type = R"doc()doc";

static const char *__doc_sgl_DescriptorHandle_value = R"doc()doc";

static const char *__doc_sgl_Device = R"doc()doc";

static const char *__doc_sgl_Device_2 = R"doc()doc";

static const char *__doc_sgl_DeviceDesc = R"doc()doc";

static const char *__doc_sgl_DeviceDesc_2 = R"doc()doc";

static const char *__doc_sgl_DeviceDesc_adapter_luid = R"doc(Adapter LUID to select adapter on which the device will be created.)doc";

static const char *__doc_sgl_DeviceDesc_compiler_options = R"doc(Compiler options (used for default slang session).)doc";

static const char *__doc_sgl_DeviceDesc_enable_compilation_reports = R"doc(Enable compilation reports.)doc";

static const char *__doc_sgl_DeviceDesc_enable_cuda_interop = R"doc(Enable CUDA interoperability.)doc";

static const char *__doc_sgl_DeviceDesc_enable_debug_layers = R"doc(Enable debug layers.)doc";

static const char *__doc_sgl_DeviceDesc_enable_hot_reload =
R"doc(Enable automatic shader reload in response to file changes. Note:
Currently windows and linux only.)doc";

static const char *__doc_sgl_DeviceDesc_enable_print = R"doc(Enable device side printing (adds performance overhead).)doc";

static const char *__doc_sgl_DeviceDesc_existing_device_handles =
R"doc(Native device handles for initializing with externally created device.
Currenlty only used for CUDA interoperability.)doc";

static const char *__doc_sgl_DeviceDesc_shader_cache_path =
R"doc(Path to the shader cache directory (optional). If a relative path is
used, the cache is stored in the application data directory.)doc";

static const char *__doc_sgl_DeviceDesc_type = R"doc(The type of the device.)doc";

static const char *__doc_sgl_DeviceInfo = R"doc()doc";

static const char *__doc_sgl_DeviceInfo_adapter_luid = R"doc(The logically unique identifier of the graphics adapter.)doc";

static const char *__doc_sgl_DeviceInfo_adapter_name = R"doc(The name of the graphics adapter.)doc";

static const char *__doc_sgl_DeviceInfo_api_name = R"doc(The name of the graphics API being used by this device.)doc";

static const char *__doc_sgl_DeviceInfo_limits = R"doc(Limits of the device.)doc";

static const char *__doc_sgl_DeviceInfo_timestamp_frequency =
R"doc(The frequency of the timestamp counter. To resolve a timestamp to
seconds, divide by this value.)doc";

static const char *__doc_sgl_DeviceInfo_type = R"doc(The type of the device.)doc";

static const char *__doc_sgl_DeviceLimits = R"doc()doc";

static const char *__doc_sgl_DeviceLimits_max_compute_dispatch_thread_groups = R"doc(Maximum number of thread groups per dimension in a single dispatch.)doc";

static const char *__doc_sgl_DeviceLimits_max_compute_thread_group_size = R"doc(Maximum dimensions of a thread group.)doc";

static const char *__doc_sgl_DeviceLimits_max_compute_threads_per_group = R"doc(Maximum number of threads per thread group.)doc";

static const char *__doc_sgl_DeviceLimits_max_framebuffer_dimensions = R"doc(Maximum framebuffer dimensions.)doc";

static const char *__doc_sgl_DeviceLimits_max_shader_visible_samplers = R"doc(Maximum samplers visible in a shader stage.)doc";

static const char *__doc_sgl_DeviceLimits_max_texture_dimension_1d = R"doc(Maximum dimension for 1D textures.)doc";

static const char *__doc_sgl_DeviceLimits_max_texture_dimension_2d = R"doc(Maximum dimensions for 2D textures.)doc";

static const char *__doc_sgl_DeviceLimits_max_texture_dimension_3d = R"doc(Maximum dimensions for 3D textures.)doc";

static const char *__doc_sgl_DeviceLimits_max_texture_dimension_cube = R"doc(Maximum dimensions for cube textures.)doc";

static const char *__doc_sgl_DeviceLimits_max_texture_layers = R"doc(Maximum number of texture layers.)doc";

static const char *__doc_sgl_DeviceLimits_max_vertex_input_element_offset = R"doc(Maximum offset of a vertex input element in the vertex stream.)doc";

static const char *__doc_sgl_DeviceLimits_max_vertex_input_elements = R"doc(Maximum number of vertex input elements in a graphics pipeline.)doc";

static const char *__doc_sgl_DeviceLimits_max_vertex_stream_stride = R"doc(Maximum stride of a vertex stream.)doc";

static const char *__doc_sgl_DeviceLimits_max_vertex_streams = R"doc(Maximum number of vertex streams in a graphics pipeline.)doc";

static const char *__doc_sgl_DeviceLimits_max_viewport_dimensions = R"doc(Maximum viewport dimensions.)doc";

static const char *__doc_sgl_DeviceLimits_max_viewports = R"doc(Maximum number of viewports per pipeline.)doc";

static const char *__doc_sgl_DeviceResource = R"doc()doc";

static const char *__doc_sgl_DeviceResource_DeviceResource = R"doc()doc";

static const char *__doc_sgl_DeviceResource_MemoryUsage = R"doc()doc";

static const char *__doc_sgl_DeviceResource_MemoryUsage_device = R"doc(The amount of memory in bytes used on the device.)doc";

static const char *__doc_sgl_DeviceResource_MemoryUsage_host = R"doc(The amount of memory in bytes used on the host.)doc";

static const char *__doc_sgl_DeviceResource_class_name = R"doc()doc";

static const char *__doc_sgl_DeviceResource_device = R"doc()doc";

static const char *__doc_sgl_DeviceResource_m_device = R"doc()doc";

static const char *__doc_sgl_DeviceResource_memory_usage = R"doc(The memory usage by this resource.)doc";

static const char *__doc_sgl_DeviceType = R"doc()doc";

static const char *__doc_sgl_DeviceType_automatic = R"doc()doc";

static const char *__doc_sgl_DeviceType_cpu = R"doc()doc";

static const char *__doc_sgl_DeviceType_cuda = R"doc()doc";

static const char *__doc_sgl_DeviceType_d3d12 = R"doc()doc";

static const char *__doc_sgl_DeviceType_info = R"doc()doc";

static const char *__doc_sgl_DeviceType_metal = R"doc()doc";

static const char *__doc_sgl_DeviceType_vulkan = R"doc()doc";

static const char *__doc_sgl_DeviceType_wgpu = R"doc()doc";

static const char *__doc_sgl_Device_Device = R"doc()doc";

static const char *__doc_sgl_Device_blitter = R"doc()doc";

static const char *__doc_sgl_Device_class_name = R"doc()doc";

static const char *__doc_sgl_Device_close =
R"doc(Close the device.

This function should be called before the device is released. It waits
for all pending work to be completed and releases internal resources,
removing all cyclic references that might prevent the device from
being destroyed. After closing the device, no new resources must be
created and no new work must be submitted.

\note The Python extension will automatically close all open devices
when the interpreter is terminated through an `atexit` handler. If a
device is to be destroyed at runtime, it must be closed explicitly.)doc";

static const char *__doc_sgl_Device_close_all_devices = R"doc(Close all open devices.)doc";

static const char *__doc_sgl_Device_create = R"doc()doc";

static const char *__doc_sgl_Device_create_acceleration_structure = R"doc()doc";

static const char *__doc_sgl_Device_create_acceleration_structure_instance_list = R"doc()doc";

static const char *__doc_sgl_Device_create_buffer =
R"doc(Create a new buffer.

Parameter ``size``:
    Buffer size in bytes.

Parameter ``element_count``:
    Buffer size in number of struct elements. Can be used instead of
    ``size``.

Parameter ``struct_size``:
    Struct size in bytes.

Parameter ``resource_type_layout``:
    Resource type layout of the buffer. Can be used instead of
    ``struct_size`` to specify the size of the struct.

Parameter ``format``:
    Buffer format. Used when creating typed buffer views.

Parameter ``initial_state``:
    Initial resource state.

Parameter ``usage``:
    Resource usage flags.

Parameter ``memory_type``:
    Memory type.

Parameter ``label``:
    Debug label.

Parameter ``data``:
    Initial data to upload to the buffer.

Parameter ``data_size``:
    Size of the initial data in bytes.

Returns:
    New buffer object.)doc";

static const char *__doc_sgl_Device_create_buffer_view = R"doc()doc";

static const char *__doc_sgl_Device_create_command_encoder = R"doc()doc";

static const char *__doc_sgl_Device_create_compute_kernel = R"doc()doc";

static const char *__doc_sgl_Device_create_compute_pipeline = R"doc()doc";

static const char *__doc_sgl_Device_create_fence =
R"doc(Create a new fence.

Parameter ``initial_value``:
    Initial fence value.

Parameter ``shared``:
    Create a shared fence.

Returns:
    New fence object.)doc";

static const char *__doc_sgl_Device_create_input_layout =
R"doc(Create a new input layout.

Parameter ``input_elements``:
    List of input elements (see InputElementDesc for details).

Parameter ``vertex_streams``:
    List of vertex streams (see VertexStreamDesc for details).

Returns:
    New input layout object.)doc";

static const char *__doc_sgl_Device_create_query_pool =
R"doc(Create a new query pool.

Parameter ``type``:
    Query type.

Parameter ``count``:
    Number of queries in the pool.

Returns:
    New query pool object.)doc";

static const char *__doc_sgl_Device_create_ray_tracing_pipeline = R"doc()doc";

static const char *__doc_sgl_Device_create_render_pipeline = R"doc()doc";

static const char *__doc_sgl_Device_create_root_shader_object = R"doc()doc";

static const char *__doc_sgl_Device_create_sampler =
R"doc(Create a new sampler.

Parameter ``min_filter``:
    Minification filter.

Parameter ``mag_filter``:
    Magnification filter.

Parameter ``mip_filter``:
    Mip-map filter.

Parameter ``reduction_op``:
    Reduction operation.

Parameter ``address_u``:
    Texture addressing mode for the U coordinate.

Parameter ``address_v``:
    Texture addressing mode for the V coordinate.

Parameter ``address_w``:
    Texture addressing mode for the W coordinate.

Parameter ``mip_lod_bias``:
    Mip-map LOD bias.

Parameter ``max_anisotropy``:
    Maximum anisotropy.

Parameter ``comparison_func``:
    Comparison function.

Parameter ``border_color``:
    Border color.

Parameter ``min_lod``:
    Minimum LOD level.

Parameter ``max_lod``:
    Maximum LOD level.

Parameter ``label``:
    Debug label.

Returns:
    New sampler object.)doc";

static const char *__doc_sgl_Device_create_shader_object = R"doc()doc";

static const char *__doc_sgl_Device_create_shader_object_2 = R"doc()doc";

static const char *__doc_sgl_Device_create_shader_table = R"doc()doc";

static const char *__doc_sgl_Device_create_slang_session =
R"doc(Create a new slang session.

Parameter ``compiler_options``:
    Compiler options (see SlangCompilerOptions for details).

Returns:
    New slang session object.)doc";

static const char *__doc_sgl_Device_create_surface =
R"doc(Create a new surface.

Parameter ``window``:
    Window to create the surface for.

Returns:
    New surface object.)doc";

static const char *__doc_sgl_Device_create_surface_2 =
R"doc(Create a new surface.

Parameter ``window_handle``:
    Native window handle to create the surface for.

Returns:
    New surface object.)doc";

static const char *__doc_sgl_Device_create_texture =
R"doc(Create a new texture.

Parameter ``type``:
    Texture type.

Parameter ``format``:
    Texture format.

Parameter ``width``:
    Width in pixels.

Parameter ``height``:
    Height in pixels.

Parameter ``depth``:
    Depth in pixels.

Parameter ``array_length``:
    Array length.

Parameter ``mip_count``:
    Mip level count. Number of mip levels (ALL_MIPS for all mip
    levels).

Parameter ``sample_count``:
    Number of samples for multisampled textures.

Parameter ``quality``:
    Quality level for multisampled textures.

Parameter ``usage``:
    Resource usage.

Parameter ``memory_type``:
    Memory type.

Parameter ``label``:
    Debug label.

Parameter ``data``:
    Initial data.

Returns:
    New texture object.)doc";

static const char *__doc_sgl_Device_create_texture_from_resource = R"doc()doc";

static const char *__doc_sgl_Device_create_texture_view = R"doc()doc";

static const char *__doc_sgl_Device_cuda_device = R"doc()doc";

static const char *__doc_sgl_Device_debug_printer = R"doc()doc";

static const char *__doc_sgl_Device_desc = R"doc()doc";

static const char *__doc_sgl_Device_enable_agility_sdk =
R"doc(Try to enable D3D12 Agility SDK at runtime. Note: This must be called
before creating a device to have any effect.

Prefer adding SGL_EXPORT_AGILITY_SDK to the main translation unit of
executables to tag the application binary to load the D3D12 Agility
SDK.

When using sgl through as a Python extension tagging the main
application (Python interpreter) is not possible. The alternative is
to use the D3D12SDKConfiguration API introduced in Windows SDK 20348.
This however requires "Developer Mode" to be enabled and the executed
Python interpreter to be stored on the same drive as the sgl library.

Returns:
    Return true if D3D12 Agility SDK was successfully enabled.)doc";

static const char *__doc_sgl_Device_enumerate_adapters = R"doc(Enumerates all available adapters of a given device type.)doc";

static const char *__doc_sgl_Device_features = R"doc(List of features supported by the device.)doc";

static const char *__doc_sgl_Device_flush_print = R"doc(Block and flush all shader side debug print output.)doc";

static const char *__doc_sgl_Device_flush_print_to_string = R"doc(Block and flush all shader side debug print output to a string.)doc";

static const char *__doc_sgl_Device_get_acceleration_structure_sizes =
R"doc(Query the device for buffer sizes required for acceleration structure
builds.

Parameter ``desc``:
    Acceleration structure build description.

Returns:
    Acceleration structure sizes.)doc";

static const char *__doc_sgl_Device_get_format_support = R"doc(Returns the supported resource states for a given format.)doc";

static const char *__doc_sgl_Device_get_native_command_queue_handle =
R"doc(Returns the native API handle for the command queue: - D3D12:
ID3D12CommandQueue* - Vulkan: VkQueue (Vulkan))doc";

static const char *__doc_sgl_Device_get_or_create_coop_vec = R"doc(Get coop vec instance)doc";

static const char *__doc_sgl_Device_global_session = R"doc()doc";

static const char *__doc_sgl_Device_has_feature = R"doc(Check if the device supports a given feature.)doc";

static const char *__doc_sgl_Device_hot_reload = R"doc()doc";

static const char *__doc_sgl_Device_info = R"doc(Device information.)doc";

static const char *__doc_sgl_Device_is_submit_finished =
R"doc(Check if a submission is finished executing.

Parameter ``id``:
    Submission ID.

Returns:
    True if the submission is finished executing.)doc";

static const char *__doc_sgl_Device_link_program = R"doc()doc";

static const char *__doc_sgl_Device_load_module = R"doc()doc";

static const char *__doc_sgl_Device_load_module_from_source = R"doc()doc";

static const char *__doc_sgl_Device_load_program = R"doc()doc";

static const char *__doc_sgl_Device_m_blitter = R"doc()doc";

static const char *__doc_sgl_Device_m_closed = R"doc()doc";

static const char *__doc_sgl_Device_m_coop_vec = R"doc()doc";

static const char *__doc_sgl_Device_m_cuda_device = R"doc()doc";

static const char *__doc_sgl_Device_m_cuda_semaphore = R"doc()doc";

static const char *__doc_sgl_Device_m_debug_printer = R"doc()doc";

static const char *__doc_sgl_Device_m_desc = R"doc()doc";

static const char *__doc_sgl_Device_m_device_close_callbacks = R"doc(List of callbacks for shutdown event)doc";

static const char *__doc_sgl_Device_m_features = R"doc()doc";

static const char *__doc_sgl_Device_m_global_fence = R"doc()doc";

static const char *__doc_sgl_Device_m_global_session = R"doc()doc";

static const char *__doc_sgl_Device_m_hot_reload = R"doc()doc";

static const char *__doc_sgl_Device_m_info = R"doc()doc";

static const char *__doc_sgl_Device_m_rhi_device = R"doc()doc";

static const char *__doc_sgl_Device_m_rhi_graphics_queue = R"doc()doc";

static const char *__doc_sgl_Device_m_shader_cache_enabled = R"doc()doc";

static const char *__doc_sgl_Device_m_shader_cache_path = R"doc()doc";

static const char *__doc_sgl_Device_m_shader_hot_reload_callbacks = R"doc(List of callbacks for hot reload event)doc";

static const char *__doc_sgl_Device_m_slang_session = R"doc()doc";

static const char *__doc_sgl_Device_m_supported_shader_model = R"doc()doc";

static const char *__doc_sgl_Device_m_supports_cuda_interop = R"doc()doc";

static const char *__doc_sgl_Device_m_wait_global_fence = R"doc()doc";

static const char *__doc_sgl_Device_native_handles = R"doc(Get the native device handles.)doc";

static const char *__doc_sgl_Device_on_hot_reload = R"doc(Called by hot reload system after reload occurs, to trigger the hooks.)doc";

static const char *__doc_sgl_Device_read_buffer_data =
R"doc(Read buffer data to host memory. \note This will wait until the data
is copied back to host memory.

Parameter ``buffer``:
    Buffer to read from.

Parameter ``data``:
    Data to buffer to read to.

Parameter ``size``:
    Size of the data in bytes.

Parameter ``offset``:
    Offset in the buffer to read from.)doc";

static const char *__doc_sgl_Device_read_texture_data =
R"doc(Read texture data to host memory. \note This will wait until the data
is copied back to host memory.

Parameter ``texture``:
    Texture to read from.

Parameter ``layer``:
    Layer index.

Parameter ``mip``:
    Mip level.

Returns:
    Subresource data in host memory.)doc";

static const char *__doc_sgl_Device_register_device_close_callback = R"doc(Register a device close callback, called at start of device close.)doc";

static const char *__doc_sgl_Device_register_shader_hot_reload_callback =
R"doc(Register a hot reload hook, called immediately after any module is
reloaded.)doc";

static const char *__doc_sgl_Device_reload_all_programs = R"doc()doc";

static const char *__doc_sgl_Device_report_live_objects =
R"doc(Report live objects in the rhi layer. This is useful for checking
clean shutdown with all resources released properly.)doc";

static const char *__doc_sgl_Device_rhi_device = R"doc()doc";

static const char *__doc_sgl_Device_rhi_graphics_queue = R"doc()doc";

static const char *__doc_sgl_Device_shader_cache_stats = R"doc(Shader cache statistics.)doc";

static const char *__doc_sgl_Device_slang_session = R"doc(Default slang session.)doc";

static const char *__doc_sgl_Device_submit_command_buffer =
R"doc(Submit a command buffer to the device.

The returned submission ID can be used to wait for the submission to
complete.

Parameter ``command_buffer``:
    Command buffer to submit.

Parameter ``queue``:
    Command queue to submit to.

Returns:
    Submission ID.)doc";

static const char *__doc_sgl_Device_submit_command_buffers =
R"doc(Submit a list of command buffers to the device.

The returned submission ID can be used to wait for the submission to
complete.

The wait fence values are optional. If not provided, the fence values
will be set to AUTO, which means waiting for the last signaled value.

The signal fence values are optional. If not provided, the fence
values will be set to AUTO, which means incrementing the last signaled
value by 1. *

Parameter ``command_buffers``:
    List of command buffers to submit.

Parameter ``wait_fences``:
    List of fences to wait for before executing the command buffers.

Parameter ``wait_fence_values``:
    List of fence values to wait for before executing the command
    buffers.

Parameter ``signal_fences``:
    List of fences to signal after executing the command buffers.

Parameter ``signal_fence_values``:
    List of fence values to signal after executing the command
    buffers.

Parameter ``queue``:
    Command queue to submit to.

Parameter ``cuda_stream``:
    On none-CUDA backends, when interop is enabled, this is the stream
    to sync with before/after submission (assuming any resources are
    shared with CUDA) and use for internal copies. If not specified,
    sync will happen with the NULL (default) CUDA stream. On CUDA
    backends, this is the CUDA stream to use for the submission. If
    not specified, the default stream of the command queue will be
    used, which for CommandQueueType::graphics is the NULL stream. It
    is an error to specify a stream for none-CUDA backends that have
    interop disabled.

Returns:
    Submission ID.)doc";

static const char *__doc_sgl_Device_supported_shader_model = R"doc(The highest shader model supported by the device.)doc";

static const char *__doc_sgl_Device_supports_cuda_interop = R"doc(True if the device supports CUDA interoperability.)doc";

static const char *__doc_sgl_Device_sync_to_cuda =
R"doc(Synchronize CUDA -> device.

This signals a shared CUDA semaphore from the CUDA stream and then
waits for the signal on the command queue.

Parameter ``cuda_stream``:
    CUDA stream)doc";

static const char *__doc_sgl_Device_sync_to_device =
R"doc(Synchronize device -> CUDA.

This waits for a shared CUDA semaphore on the CUDA stream, making sure
all commands on the device have completed.

Parameter ``cuda_stream``:
    CUDA stream)doc";

static const char *__doc_sgl_Device_to_string = R"doc()doc";

static const char *__doc_sgl_Device_type = R"doc(Type of the graphics API used by this device.)doc";

static const char *__doc_sgl_Device_upload_buffer_data =
R"doc(Upload host memory to buffer.

Parameter ``buffer``:
    Buffer to write to.

Parameter ``offset``:
    Offset in the buffer to write to.

Parameter ``size``:
    Size of the data in bytes.

Parameter ``data``:
    Data to write.)doc";

static const char *__doc_sgl_Device_upload_texture_data =
R"doc(Upload host memory to texture.

Parameter ``texture``:
    Texture to write to.

Parameter ``subresource``:
    Subresource index.

Parameter ``subresource_data``:
    Subresource data.)doc";

static const char *__doc_sgl_Device_upload_texture_data_2 = R"doc()doc";

static const char *__doc_sgl_Device_wait = R"doc(Wait for all device work to complete.)doc";

static const char *__doc_sgl_Device_wait_for_idle =
R"doc(Wait for the command queue to be idle.

Parameter ``queue``:
    Command queue to wait for.)doc";

static const char *__doc_sgl_Device_wait_for_submit =
R"doc(Wait for a submission to finish execution.

Parameter ``id``:
    Submission ID.)doc";

static const char *__doc_sgl_DrawArguments = R"doc()doc";

static const char *__doc_sgl_DrawArguments_instance_count = R"doc()doc";

static const char *__doc_sgl_DrawArguments_start_index_location = R"doc()doc";

static const char *__doc_sgl_DrawArguments_start_instance_location = R"doc()doc";

static const char *__doc_sgl_DrawArguments_start_vertex_location = R"doc()doc";

static const char *__doc_sgl_DrawArguments_vertex_count = R"doc()doc";

static const char *__doc_sgl_EOFException = R"doc()doc";

static const char *__doc_sgl_EOFException_2 = R"doc()doc";

static const char *__doc_sgl_EOFException_EOFException = R"doc()doc";

static const char *__doc_sgl_EOFException_gcount = R"doc()doc";

static const char *__doc_sgl_EOFException_m_gcount = R"doc()doc";

static const char *__doc_sgl_EntryPointLayout = R"doc()doc";

static const char *__doc_sgl_EntryPointLayout_2 = R"doc()doc";

static const char *__doc_sgl_EntryPointLayoutParameterList = R"doc()doc";

static const char *__doc_sgl_EntryPointLayoutParameterList_2 = R"doc(EntryPointLayout lazy parameter list evaluation.)doc";

static const char *__doc_sgl_EntryPointLayoutParameterList_EntryPointLayoutParameterList = R"doc()doc";

static const char *__doc_sgl_EntryPointLayoutParameterList_evaluate = R"doc(Get a specific child.)doc";

static const char *__doc_sgl_EntryPointLayoutParameterList_size = R"doc(Number of entries in list.)doc";

static const char *__doc_sgl_EntryPointLayout_EntryPointLayout = R"doc()doc";

static const char *__doc_sgl_EntryPointLayout_compute_thread_group_size = R"doc()doc";

static const char *__doc_sgl_EntryPointLayout_from_slang = R"doc()doc";

static const char *__doc_sgl_EntryPointLayout_get_parameter_by_index = R"doc()doc";

static const char *__doc_sgl_EntryPointLayout_name = R"doc()doc";

static const char *__doc_sgl_EntryPointLayout_name_override = R"doc()doc";

static const char *__doc_sgl_EntryPointLayout_parameter_count = R"doc()doc";

static const char *__doc_sgl_EntryPointLayout_parameters = R"doc()doc";

static const char *__doc_sgl_EntryPointLayout_stage = R"doc()doc";

static const char *__doc_sgl_EntryPointLayout_to_string = R"doc()doc";

static const char *__doc_sgl_EntryPointLayout_uses_any_sample_rate_input = R"doc()doc";

static const char *__doc_sgl_ExceptionDiagnosticFlags = R"doc()doc";

static const char *__doc_sgl_ExceptionDiagnosticFlags_break_debugger = R"doc(Break into debugger when throwing an exception.)doc";

static const char *__doc_sgl_ExceptionDiagnosticFlags_log = R"doc(Log exception message.)doc";

static const char *__doc_sgl_ExceptionDiagnosticFlags_none = R"doc()doc";

static const char *__doc_sgl_Feature = R"doc()doc";

static const char *__doc_sgl_Feature_acceleration_structure = R"doc()doc";

static const char *__doc_sgl_Feature_acceleration_structure_linear_swept_spheres = R"doc()doc";

static const char *__doc_sgl_Feature_acceleration_structure_spheres = R"doc()doc";

static const char *__doc_sgl_Feature_argument_buffer_tier2 = R"doc()doc";

static const char *__doc_sgl_Feature_atomic_float = R"doc()doc";

static const char *__doc_sgl_Feature_atomic_half = R"doc()doc";

static const char *__doc_sgl_Feature_atomic_int64 = R"doc()doc";

static const char *__doc_sgl_Feature_barycentrics = R"doc()doc";

static const char *__doc_sgl_Feature_bindless = R"doc()doc";

static const char *__doc_sgl_Feature_conservative_rasterization = R"doc()doc";

static const char *__doc_sgl_Feature_conservative_rasterization1 = R"doc()doc";

static const char *__doc_sgl_Feature_conservative_rasterization2 = R"doc()doc";

static const char *__doc_sgl_Feature_conservative_rasterization3 = R"doc()doc";

static const char *__doc_sgl_Feature_cooperative_matrix = R"doc()doc";

static const char *__doc_sgl_Feature_cooperative_vector = R"doc()doc";

static const char *__doc_sgl_Feature_count = R"doc()doc";

static const char *__doc_sgl_Feature_custom_border_color = R"doc()doc";

static const char *__doc_sgl_Feature_double = R"doc()doc";

static const char *__doc_sgl_Feature_fragment_shading_rate = R"doc()doc";

static const char *__doc_sgl_Feature_half = R"doc()doc";

static const char *__doc_sgl_Feature_hardware_device = R"doc()doc";

static const char *__doc_sgl_Feature_info = R"doc()doc";

static const char *__doc_sgl_Feature_int16 = R"doc()doc";

static const char *__doc_sgl_Feature_int64 = R"doc()doc";

static const char *__doc_sgl_Feature_mesh_shader = R"doc()doc";

static const char *__doc_sgl_Feature_multi_view = R"doc()doc";

static const char *__doc_sgl_Feature_parameter_block = R"doc()doc";

static const char *__doc_sgl_Feature_pipeline_cache = R"doc()doc";

static const char *__doc_sgl_Feature_pointer = R"doc()doc";

static const char *__doc_sgl_Feature_programmable_sample_positions1 = R"doc()doc";

static const char *__doc_sgl_Feature_programmable_sample_positions2 = R"doc()doc";

static const char *__doc_sgl_Feature_rasterization = R"doc()doc";

static const char *__doc_sgl_Feature_rasterizer_ordered_views = R"doc()doc";

static const char *__doc_sgl_Feature_ray_query = R"doc()doc";

static const char *__doc_sgl_Feature_ray_tracing = R"doc()doc";

static const char *__doc_sgl_Feature_ray_tracing_validation = R"doc()doc";

static const char *__doc_sgl_Feature_realtime_clock = R"doc()doc";

static const char *__doc_sgl_Feature_sampler_feedback = R"doc()doc";

static const char *__doc_sgl_Feature_shader_execution_reordering = R"doc()doc";

static const char *__doc_sgl_Feature_shader_resource_min_lod = R"doc()doc";

static const char *__doc_sgl_Feature_sm_5_1 = R"doc()doc";

static const char *__doc_sgl_Feature_sm_6_0 = R"doc()doc";

static const char *__doc_sgl_Feature_sm_6_1 = R"doc()doc";

static const char *__doc_sgl_Feature_sm_6_2 = R"doc()doc";

static const char *__doc_sgl_Feature_sm_6_3 = R"doc()doc";

static const char *__doc_sgl_Feature_sm_6_4 = R"doc()doc";

static const char *__doc_sgl_Feature_sm_6_5 = R"doc()doc";

static const char *__doc_sgl_Feature_sm_6_6 = R"doc()doc";

static const char *__doc_sgl_Feature_sm_6_7 = R"doc()doc";

static const char *__doc_sgl_Feature_sm_6_8 = R"doc()doc";

static const char *__doc_sgl_Feature_sm_6_9 = R"doc()doc";

static const char *__doc_sgl_Feature_software_device = R"doc()doc";

static const char *__doc_sgl_Feature_surface = R"doc()doc";

static const char *__doc_sgl_Feature_timestamp_query = R"doc()doc";

static const char *__doc_sgl_Feature_wave_ops = R"doc()doc";

static const char *__doc_sgl_Fence = R"doc(Fence.)doc";

static const char *__doc_sgl_Fence_2 = R"doc()doc";

static const char *__doc_sgl_FenceDesc = R"doc(Fence descriptor.)doc";

static const char *__doc_sgl_FenceDesc_2 = R"doc()doc";

static const char *__doc_sgl_FenceDesc_initial_value = R"doc(Initial fence value.)doc";

static const char *__doc_sgl_FenceDesc_shared = R"doc(Create a shared fence.)doc";

static const char *__doc_sgl_Fence_Fence =
R"doc(Constructor. Do not use directly, instead use
``Device::create_fence``.)doc";

static const char *__doc_sgl_Fence_class_name = R"doc()doc";

static const char *__doc_sgl_Fence_current_value = R"doc(Returns the currently signaled value on the device.)doc";

static const char *__doc_sgl_Fence_desc = R"doc()doc";

static const char *__doc_sgl_Fence_m_desc = R"doc()doc";

static const char *__doc_sgl_Fence_m_rhi_fence = R"doc()doc";

static const char *__doc_sgl_Fence_m_signaled_value = R"doc()doc";

static const char *__doc_sgl_Fence_native_handle = R"doc(Get the native fence handle.)doc";

static const char *__doc_sgl_Fence_rhi_fence = R"doc()doc";

static const char *__doc_sgl_Fence_shared_handle =
R"doc(Get the shared fence handle. Note: Fence must be created with the
``FenceDesc::shared`` flag.)doc";

static const char *__doc_sgl_Fence_signal =
R"doc(Signal the fence. This signals the fence from the host.

Parameter ``value``:
    The value to signal. If ``AUTO``, the signaled value will be auto-
    incremented.

Returns:
    The signaled value.)doc";

static const char *__doc_sgl_Fence_signaled_value = R"doc(Returns the last signaled value on the device.)doc";

static const char *__doc_sgl_Fence_to_string = R"doc()doc";

static const char *__doc_sgl_Fence_update_signaled_value =
R"doc(Updates or increments the signaled value. This is used before
signaling a fence (from the host, on the device or from an external
source), to update the internal state. The passed value is stored, or
if ``AUTO``, the last signaled value is auto-incremented by one. The
returned value is what the caller should signal to the fence.

Parameter ``value``:
    The value to signal. If ``AUTO``, the signaled value will be auto-
    incremented.

Returns:
    The value to signal to the fence.)doc";

static const char *__doc_sgl_Fence_wait =
R"doc(Wait for the fence to be signaled on the host. Blocks the host until
the fence reaches or exceeds the specified value.

Parameter ``value``:
    The value to wait for. If ``AUTO``, wait for the last signaled
    value.

Parameter ``timeout_ns``:
    The timeout in nanoseconds. If ``TIMEOUT_INFINITE``, the function
    will block indefinitely.)doc";

static const char *__doc_sgl_FileLoggerOutput = R"doc()doc";

static const char *__doc_sgl_FileLoggerOutput_2 = R"doc(Logger output that writes to a file.)doc";

static const char *__doc_sgl_FileLoggerOutput_FileLoggerOutput = R"doc()doc";

static const char *__doc_sgl_FileLoggerOutput_m_file = R"doc()doc";

static const char *__doc_sgl_FileLoggerOutput_m_path = R"doc()doc";

static const char *__doc_sgl_FileLoggerOutput_path = R"doc()doc";

static const char *__doc_sgl_FileLoggerOutput_to_string = R"doc()doc";

static const char *__doc_sgl_FileLoggerOutput_write = R"doc()doc";

static const char *__doc_sgl_FileStream = R"doc()doc";

static const char *__doc_sgl_FileStream_2 = R"doc()doc";

static const char *__doc_sgl_FileStream_FileStream = R"doc()doc";

static const char *__doc_sgl_FileStream_Mode = R"doc()doc";

static const char *__doc_sgl_FileStream_Mode_info = R"doc()doc";

static const char *__doc_sgl_FileStream_Mode_read = R"doc()doc";

static const char *__doc_sgl_FileStream_Mode_read_write = R"doc()doc";

static const char *__doc_sgl_FileStream_Mode_write = R"doc()doc";

static const char *__doc_sgl_FileStream_class_name = R"doc()doc";

static const char *__doc_sgl_FileStream_close = R"doc()doc";

static const char *__doc_sgl_FileStream_flush = R"doc()doc";

static const char *__doc_sgl_FileStream_is_open = R"doc()doc";

static const char *__doc_sgl_FileStream_is_readable = R"doc()doc";

static const char *__doc_sgl_FileStream_is_writable = R"doc()doc";

static const char *__doc_sgl_FileStream_m_mode = R"doc()doc";

static const char *__doc_sgl_FileStream_m_path = R"doc()doc";

static const char *__doc_sgl_FileStream_m_stream = R"doc()doc";

static const char *__doc_sgl_FileStream_mode = R"doc()doc";

static const char *__doc_sgl_FileStream_path = R"doc()doc";

static const char *__doc_sgl_FileStream_read = R"doc()doc";

static const char *__doc_sgl_FileStream_seek = R"doc()doc";

static const char *__doc_sgl_FileStream_size = R"doc()doc";

static const char *__doc_sgl_FileStream_tell = R"doc()doc";

static const char *__doc_sgl_FileStream_to_string = R"doc()doc";

static const char *__doc_sgl_FileStream_truncate = R"doc()doc";

static const char *__doc_sgl_FileStream_write = R"doc()doc";

static const char *__doc_sgl_FileSystemWatchDesc = R"doc(Init options for FileSystemWatcher.)doc";

static const char *__doc_sgl_FileSystemWatchDesc_2 = R"doc()doc";

static const char *__doc_sgl_FileSystemWatchDesc_directory = R"doc(Directory to monitor.)doc";

static const char *__doc_sgl_FileSystemWatchEvent =
R"doc(Data reported on a given file system event to a file monitored by
FileSystemWatcher.)doc";

static const char *__doc_sgl_FileSystemWatchEvent_2 = R"doc()doc";

static const char *__doc_sgl_FileSystemWatchEvent_absolute_path = R"doc(Absolute path of file in file system)doc";

static const char *__doc_sgl_FileSystemWatchEvent_change = R"doc(Change type.)doc";

static const char *__doc_sgl_FileSystemWatchEvent_path = R"doc(Path of file that has changed relative to watcher.)doc";

static const char *__doc_sgl_FileSystemWatchEvent_time = R"doc(System time change was recorded.)doc";

static const char *__doc_sgl_FileSystemWatchState = R"doc()doc";

static const char *__doc_sgl_FileSystemWatcher =
R"doc(Monitors directories for changes and calls a callback when they're
detected. The watcher automatically queues up changes until disk has
been idle for a period. Relies on regular polling of update().)doc";

static const char *__doc_sgl_FileSystemWatcher_2 = R"doc()doc";

static const char *__doc_sgl_FileSystemWatcherChange = R"doc(Types of file system event that can be reported.)doc";

static const char *__doc_sgl_FileSystemWatcherChange_added = R"doc()doc";

static const char *__doc_sgl_FileSystemWatcherChange_invalid = R"doc()doc";

static const char *__doc_sgl_FileSystemWatcherChange_modified = R"doc()doc";

static const char *__doc_sgl_FileSystemWatcherChange_removed = R"doc()doc";

static const char *__doc_sgl_FileSystemWatcher_FileSystemWatcher = R"doc()doc";

static const char *__doc_sgl_FileSystemWatcher_FileSystemWatcher_2 =
R"doc(FileSystemWatch move constructor is deleted to allow for map of unique
ptrs.)doc";

static const char *__doc_sgl_FileSystemWatcher_add_watch = R"doc(Add watch of a new directory.)doc";

static const char *__doc_sgl_FileSystemWatcher_class_name = R"doc()doc";

static const char *__doc_sgl_FileSystemWatcher_delay = R"doc(Delay period before queued events are output.)doc";

static const char *__doc_sgl_FileSystemWatcher_m_last_event = R"doc(Time last event was recorded.)doc";

static const char *__doc_sgl_FileSystemWatcher_m_next_id = R"doc(Next unique id to be assigned to a given watch.)doc";

static const char *__doc_sgl_FileSystemWatcher_m_on_change = R"doc(Watch event callback.)doc";

static const char *__doc_sgl_FileSystemWatcher_m_output_delay_ms = R"doc(Delay before outputting events in milliseconds.)doc";

static const char *__doc_sgl_FileSystemWatcher_m_queued_events = R"doc(Events reported since last call to watch event callback.)doc";

static const char *__doc_sgl_FileSystemWatcher_m_queued_events_mutex = R"doc(Mutex to protect the queued events.)doc";

static const char *__doc_sgl_FileSystemWatcher_m_stop_thread = R"doc()doc";

static const char *__doc_sgl_FileSystemWatcher_m_thread = R"doc(Thread to poll for file system changes.)doc";

static const char *__doc_sgl_FileSystemWatcher_m_watches = R"doc(Map of id->watch.)doc";

static const char *__doc_sgl_FileSystemWatcher_m_watches_mutex = R"doc(Mutex to protect the watch map.)doc";

static const char *__doc_sgl_FileSystemWatcher_notify_change = R"doc(Internal function called when OS reports an event.)doc";

static const char *__doc_sgl_FileSystemWatcher_on_change = R"doc(Get callback for file system events.)doc";

static const char *__doc_sgl_FileSystemWatcher_remove_watch = R"doc(Remove existing watch.)doc";

static const char *__doc_sgl_FileSystemWatcher_remove_watch_2 = R"doc(Remove existing watch.)doc";

static const char *__doc_sgl_FileSystemWatcher_set_delay = R"doc(Set delay period before queued events are output.)doc";

static const char *__doc_sgl_FileSystemWatcher_set_on_change = R"doc(Set callback for file system events.)doc";

static const char *__doc_sgl_FileSystemWatcher_stop_watch = R"doc(Releases OS monitoring for a given watch.)doc";

static const char *__doc_sgl_FileSystemWatcher_thread_func = R"doc()doc";

static const char *__doc_sgl_FileSystemWatcher_update = R"doc(Update function to poll the watcher + report events.)doc";

static const char *__doc_sgl_FileSystemWatcher_watch_count = R"doc(Get number of active watches)doc";

static const char *__doc_sgl_FillMode = R"doc()doc";

static const char *__doc_sgl_FillMode_info = R"doc()doc";

static const char *__doc_sgl_FillMode_solid = R"doc()doc";

static const char *__doc_sgl_FillMode_wireframe = R"doc()doc";

static const char *__doc_sgl_Format = R"doc(Resource formats.)doc";

static const char *__doc_sgl_FormatChannels = R"doc()doc";

static const char *__doc_sgl_FormatChannels_a = R"doc()doc";

static const char *__doc_sgl_FormatChannels_b = R"doc()doc";

static const char *__doc_sgl_FormatChannels_g = R"doc()doc";

static const char *__doc_sgl_FormatChannels_info = R"doc()doc";

static const char *__doc_sgl_FormatChannels_none = R"doc()doc";

static const char *__doc_sgl_FormatChannels_r = R"doc()doc";

static const char *__doc_sgl_FormatChannels_rg = R"doc()doc";

static const char *__doc_sgl_FormatChannels_rgb = R"doc()doc";

static const char *__doc_sgl_FormatChannels_rgba = R"doc()doc";

static const char *__doc_sgl_FormatInfo = R"doc(Resource format information.)doc";

static const char *__doc_sgl_FormatInfo_block_height = R"doc(Block height for compressed formats (1 for uncompressed formats).)doc";

static const char *__doc_sgl_FormatInfo_block_width = R"doc(Block width for compressed formats (1 for uncompressed formats).)doc";

static const char *__doc_sgl_FormatInfo_bytes_per_block = R"doc(Number of bytes per block (compressed) or pixel (uncompressed).)doc";

static const char *__doc_sgl_FormatInfo_channel_bit_count = R"doc(Number of bits per channel.)doc";

static const char *__doc_sgl_FormatInfo_channel_count = R"doc(Number of channels.)doc";

static const char *__doc_sgl_FormatInfo_dxgi_format = R"doc(DXGI format.)doc";

static const char *__doc_sgl_FormatInfo_format = R"doc(Resource format.)doc";

static const char *__doc_sgl_FormatInfo_get_channel_bits = R"doc(Get the number of bits for the specified channels.)doc";

static const char *__doc_sgl_FormatInfo_get_channels = R"doc(Get the channels for the format (only for color formats).)doc";

static const char *__doc_sgl_FormatInfo_has_equal_channel_bits = R"doc(Check if all channels have the same number of bits.)doc";

static const char *__doc_sgl_FormatInfo_is_compressed = R"doc(True if format is compressed.)doc";

static const char *__doc_sgl_FormatInfo_is_depth = R"doc(True if format has a depth component.)doc";

static const char *__doc_sgl_FormatInfo_is_depth_stencil = R"doc(True if format has a depth or stencil component.)doc";

static const char *__doc_sgl_FormatInfo_is_float_format = R"doc(True if format is floating point.)doc";

static const char *__doc_sgl_FormatInfo_is_integer_format = R"doc(True if format is integer.)doc";

static const char *__doc_sgl_FormatInfo_is_normalized_format = R"doc(True if format is normalized.)doc";

static const char *__doc_sgl_FormatInfo_is_srgb_format = R"doc(True if format is sRGB.)doc";

static const char *__doc_sgl_FormatInfo_is_stencil = R"doc(True if format has a stencil component.)doc";

static const char *__doc_sgl_FormatInfo_name = R"doc(Format name.)doc";

static const char *__doc_sgl_FormatInfo_slang_format = R"doc(Slang format.)doc";

static const char *__doc_sgl_FormatInfo_to_string = R"doc()doc";

static const char *__doc_sgl_FormatInfo_type = R"doc(Format type (typeless, float, unorm, unorm_srgb, snorm, uint, sint).)doc";

static const char *__doc_sgl_FormatInfo_vk_format = R"doc(Vulkan format.)doc";

static const char *__doc_sgl_FormatSupport = R"doc()doc";

static const char *__doc_sgl_FormatSupport_blendable = R"doc()doc";

static const char *__doc_sgl_FormatSupport_buffer = R"doc()doc";

static const char *__doc_sgl_FormatSupport_copy_destination = R"doc()doc";

static const char *__doc_sgl_FormatSupport_copy_source = R"doc()doc";

static const char *__doc_sgl_FormatSupport_depth_stencil = R"doc()doc";

static const char *__doc_sgl_FormatSupport_index_buffer = R"doc()doc";

static const char *__doc_sgl_FormatSupport_info = R"doc()doc";

static const char *__doc_sgl_FormatSupport_multisampling = R"doc()doc";

static const char *__doc_sgl_FormatSupport_none = R"doc()doc";

static const char *__doc_sgl_FormatSupport_render_target = R"doc()doc";

static const char *__doc_sgl_FormatSupport_resolvable = R"doc()doc";

static const char *__doc_sgl_FormatSupport_shader_atomic = R"doc()doc";

static const char *__doc_sgl_FormatSupport_shader_load = R"doc()doc";

static const char *__doc_sgl_FormatSupport_shader_sample = R"doc()doc";

static const char *__doc_sgl_FormatSupport_shader_uav_load = R"doc()doc";

static const char *__doc_sgl_FormatSupport_shader_uav_store = R"doc()doc";

static const char *__doc_sgl_FormatSupport_texture = R"doc()doc";

static const char *__doc_sgl_FormatSupport_vertex_buffer = R"doc()doc";

static const char *__doc_sgl_FormatType = R"doc(Resource format types.)doc";

static const char *__doc_sgl_FormatType_float = R"doc(Floating-point formats.)doc";

static const char *__doc_sgl_FormatType_info = R"doc()doc";

static const char *__doc_sgl_FormatType_sint = R"doc(Signed integer formats.)doc";

static const char *__doc_sgl_FormatType_snorm = R"doc(Signed normalized formats.)doc";

static const char *__doc_sgl_FormatType_uint = R"doc(Unsigned integer formats.)doc";

static const char *__doc_sgl_FormatType_unknown = R"doc(Unknown format.)doc";

static const char *__doc_sgl_FormatType_unorm = R"doc(Unsigned normalized formats.)doc";

static const char *__doc_sgl_FormatType_unorm_srgb = R"doc(Unsigned normalized SRGB formats.)doc";

static const char *__doc_sgl_Format_b5g6r5_unorm = R"doc()doc";

static const char *__doc_sgl_Format_bc1_unorm = R"doc()doc";

static const char *__doc_sgl_Format_bc1_unorm_srgb = R"doc()doc";

static const char *__doc_sgl_Format_bc2_unorm = R"doc()doc";

static const char *__doc_sgl_Format_bc2_unorm_srgb = R"doc()doc";

static const char *__doc_sgl_Format_bc3_unorm = R"doc()doc";

static const char *__doc_sgl_Format_bc3_unorm_srgb = R"doc()doc";

static const char *__doc_sgl_Format_bc4_snorm = R"doc()doc";

static const char *__doc_sgl_Format_bc4_unorm = R"doc()doc";

static const char *__doc_sgl_Format_bc5_snorm = R"doc()doc";

static const char *__doc_sgl_Format_bc5_unorm = R"doc()doc";

static const char *__doc_sgl_Format_bc6h_sfloat = R"doc()doc";

static const char *__doc_sgl_Format_bc6h_ufloat = R"doc()doc";

static const char *__doc_sgl_Format_bc7_unorm = R"doc()doc";

static const char *__doc_sgl_Format_bc7_unorm_srgb = R"doc()doc";

static const char *__doc_sgl_Format_bgr5a1_unorm = R"doc()doc";

static const char *__doc_sgl_Format_bgra4_unorm = R"doc()doc";

static const char *__doc_sgl_Format_bgra8_unorm = R"doc()doc";

static const char *__doc_sgl_Format_bgra8_unorm_srgb = R"doc()doc";

static const char *__doc_sgl_Format_bgrx8_unorm = R"doc()doc";

static const char *__doc_sgl_Format_bgrx8_unorm_srgb = R"doc()doc";

static const char *__doc_sgl_Format_count = R"doc()doc";

static const char *__doc_sgl_Format_d16_unorm = R"doc()doc";

static const char *__doc_sgl_Format_d32_float = R"doc()doc";

static const char *__doc_sgl_Format_d32_float_s8_uint = R"doc()doc";

static const char *__doc_sgl_Format_info = R"doc()doc";

static const char *__doc_sgl_Format_r11g11b10_float = R"doc()doc";

static const char *__doc_sgl_Format_r16_float = R"doc()doc";

static const char *__doc_sgl_Format_r16_sint = R"doc()doc";

static const char *__doc_sgl_Format_r16_snorm = R"doc()doc";

static const char *__doc_sgl_Format_r16_uint = R"doc()doc";

static const char *__doc_sgl_Format_r16_unorm = R"doc()doc";

static const char *__doc_sgl_Format_r32_float = R"doc()doc";

static const char *__doc_sgl_Format_r32_sint = R"doc()doc";

static const char *__doc_sgl_Format_r32_uint = R"doc()doc";

static const char *__doc_sgl_Format_r64_sint = R"doc()doc";

static const char *__doc_sgl_Format_r64_uint = R"doc()doc";

static const char *__doc_sgl_Format_r8_sint = R"doc()doc";

static const char *__doc_sgl_Format_r8_snorm = R"doc()doc";

static const char *__doc_sgl_Format_r8_uint = R"doc()doc";

static const char *__doc_sgl_Format_r8_unorm = R"doc()doc";

static const char *__doc_sgl_Format_rg16_float = R"doc()doc";

static const char *__doc_sgl_Format_rg16_sint = R"doc()doc";

static const char *__doc_sgl_Format_rg16_snorm = R"doc()doc";

static const char *__doc_sgl_Format_rg16_uint = R"doc()doc";

static const char *__doc_sgl_Format_rg16_unorm = R"doc()doc";

static const char *__doc_sgl_Format_rg32_float = R"doc()doc";

static const char *__doc_sgl_Format_rg32_sint = R"doc()doc";

static const char *__doc_sgl_Format_rg32_uint = R"doc()doc";

static const char *__doc_sgl_Format_rg8_sint = R"doc()doc";

static const char *__doc_sgl_Format_rg8_snorm = R"doc()doc";

static const char *__doc_sgl_Format_rg8_uint = R"doc()doc";

static const char *__doc_sgl_Format_rg8_unorm = R"doc()doc";

static const char *__doc_sgl_Format_rgb10a2_uint = R"doc()doc";

static const char *__doc_sgl_Format_rgb10a2_unorm = R"doc()doc";

static const char *__doc_sgl_Format_rgb32_float = R"doc()doc";

static const char *__doc_sgl_Format_rgb32_sint = R"doc()doc";

static const char *__doc_sgl_Format_rgb32_uint = R"doc()doc";

static const char *__doc_sgl_Format_rgb9e5_ufloat = R"doc()doc";

static const char *__doc_sgl_Format_rgba16_float = R"doc()doc";

static const char *__doc_sgl_Format_rgba16_sint = R"doc()doc";

static const char *__doc_sgl_Format_rgba16_snorm = R"doc()doc";

static const char *__doc_sgl_Format_rgba16_uint = R"doc()doc";

static const char *__doc_sgl_Format_rgba16_unorm = R"doc()doc";

static const char *__doc_sgl_Format_rgba32_float = R"doc()doc";

static const char *__doc_sgl_Format_rgba32_sint = R"doc()doc";

static const char *__doc_sgl_Format_rgba32_uint = R"doc()doc";

static const char *__doc_sgl_Format_rgba8_sint = R"doc()doc";

static const char *__doc_sgl_Format_rgba8_snorm = R"doc()doc";

static const char *__doc_sgl_Format_rgba8_uint = R"doc()doc";

static const char *__doc_sgl_Format_rgba8_unorm = R"doc()doc";

static const char *__doc_sgl_Format_rgba8_unorm_srgb = R"doc()doc";

static const char *__doc_sgl_Format_undefined = R"doc()doc";

static const char *__doc_sgl_FrontFaceMode = R"doc()doc";

static const char *__doc_sgl_FrontFaceMode_clockwise = R"doc()doc";

static const char *__doc_sgl_FrontFaceMode_counter_clockwise = R"doc()doc";

static const char *__doc_sgl_FrontFaceMode_info = R"doc()doc";

static const char *__doc_sgl_FunctionReflection = R"doc()doc";

static const char *__doc_sgl_FunctionReflection_2 = R"doc()doc";

static const char *__doc_sgl_FunctionReflectionOverloadList = R"doc()doc";

static const char *__doc_sgl_FunctionReflectionOverloadList_2 = R"doc(FunctionReflection lazy overload list evaluation.)doc";

static const char *__doc_sgl_FunctionReflectionOverloadList_FunctionReflectionOverloadList = R"doc()doc";

static const char *__doc_sgl_FunctionReflectionOverloadList_evaluate = R"doc(Get a specific child.)doc";

static const char *__doc_sgl_FunctionReflectionOverloadList_size = R"doc(Number of entries in list.)doc";

static const char *__doc_sgl_FunctionReflectionParameterList = R"doc()doc";

static const char *__doc_sgl_FunctionReflectionParameterList_2 = R"doc(FunctionReflection lazy parameter list evaluation.)doc";

static const char *__doc_sgl_FunctionReflectionParameterList_FunctionReflectionParameterList = R"doc()doc";

static const char *__doc_sgl_FunctionReflectionParameterList_evaluate = R"doc(Get a specific child.)doc";

static const char *__doc_sgl_FunctionReflectionParameterList_size = R"doc(Number of entries in list.)doc";

static const char *__doc_sgl_FunctionReflection_FunctionReflection = R"doc()doc";

static const char *__doc_sgl_FunctionReflection_get_overload_by_index = R"doc(Get a given overload of this function.)doc";

static const char *__doc_sgl_FunctionReflection_get_parameter_by_index = R"doc(Get a single parameter.)doc";

static const char *__doc_sgl_FunctionReflection_has_modifier = R"doc(Check if the function has a given modifier (e.g. 'differentiable').)doc";

static const char *__doc_sgl_FunctionReflection_is_overloaded =
R"doc(Check whether this function object represents a group of overloaded
functions, accessible via the overloads list.)doc";

static const char *__doc_sgl_FunctionReflection_name = R"doc(Function name.)doc";

static const char *__doc_sgl_FunctionReflection_overload_count = R"doc(Get number of available overloads of this function.)doc";

static const char *__doc_sgl_FunctionReflection_overloads = R"doc(List of all overloads of this function.)doc";

static const char *__doc_sgl_FunctionReflection_parameter_count = R"doc(Number of function parameters.)doc";

static const char *__doc_sgl_FunctionReflection_parameters = R"doc(List of all function parameters.)doc";

static const char *__doc_sgl_FunctionReflection_return_type = R"doc(Function return type.)doc";

static const char *__doc_sgl_FunctionReflection_specialize_with_arg_types =
R"doc(Specialize a generic or interface based function with a set of
concrete argument types. Calling on a none-generic/interface function
will simply validate all argument types can be implicitly converted to
their respective parameter types. Where a function contains multiple
overloads, specialize will identify the correct overload based on the
arguments.)doc";

static const char *__doc_sgl_GamepadButton = R"doc(Gamepad buttons.)doc";

static const char *__doc_sgl_GamepadButton_a = R"doc()doc";

static const char *__doc_sgl_GamepadButton_b = R"doc()doc";

static const char *__doc_sgl_GamepadButton_back = R"doc()doc";

static const char *__doc_sgl_GamepadButton_down = R"doc()doc";

static const char *__doc_sgl_GamepadButton_guide = R"doc()doc";

static const char *__doc_sgl_GamepadButton_info = R"doc()doc";

static const char *__doc_sgl_GamepadButton_left = R"doc()doc";

static const char *__doc_sgl_GamepadButton_left_bumper = R"doc()doc";

static const char *__doc_sgl_GamepadButton_left_thumb = R"doc()doc";

static const char *__doc_sgl_GamepadButton_right = R"doc()doc";

static const char *__doc_sgl_GamepadButton_right_bumper = R"doc()doc";

static const char *__doc_sgl_GamepadButton_right_thumb = R"doc()doc";

static const char *__doc_sgl_GamepadButton_start = R"doc()doc";

static const char *__doc_sgl_GamepadButton_up = R"doc()doc";

static const char *__doc_sgl_GamepadButton_x = R"doc()doc";

static const char *__doc_sgl_GamepadButton_y = R"doc()doc";

static const char *__doc_sgl_GamepadEvent = R"doc()doc";

static const char *__doc_sgl_GamepadEvent_2 = R"doc(Gamepad event.)doc";

static const char *__doc_sgl_GamepadEventType = R"doc(Gamepad event types.)doc";

static const char *__doc_sgl_GamepadEventType_button_down = R"doc(Gamepad button was pressed.)doc";

static const char *__doc_sgl_GamepadEventType_button_up = R"doc(Gamepad button was released.)doc";

static const char *__doc_sgl_GamepadEventType_connect = R"doc(Gamepad was connected.)doc";

static const char *__doc_sgl_GamepadEventType_disconnect = R"doc(Gamepad was disconnected.)doc";

static const char *__doc_sgl_GamepadEventType_info = R"doc()doc";

static const char *__doc_sgl_GamepadEvent_button = R"doc(The gamepad button that was pressed/released.)doc";

static const char *__doc_sgl_GamepadEvent_is_button_down = R"doc(Returns true if this event is a gamepad button down event.)doc";

static const char *__doc_sgl_GamepadEvent_is_button_up = R"doc(Returns true if this event is a gamepad button up event.)doc";

static const char *__doc_sgl_GamepadEvent_is_connect = R"doc(Returns true if this event is a gamepad connect event.)doc";

static const char *__doc_sgl_GamepadEvent_is_disconnect = R"doc(Returns true if this event is a gamepad disconnect event.)doc";

static const char *__doc_sgl_GamepadEvent_to_string = R"doc()doc";

static const char *__doc_sgl_GamepadEvent_type = R"doc(The event type.)doc";

static const char *__doc_sgl_GamepadState = R"doc()doc";

static const char *__doc_sgl_GamepadState_2 = R"doc(Gamepad state.)doc";

static const char *__doc_sgl_GamepadState_buttons = R"doc(Bitfield of gamepad buttons (see GamepadButton).)doc";

static const char *__doc_sgl_GamepadState_is_button_down = R"doc(Returns true if the specified button is down.)doc";

static const char *__doc_sgl_GamepadState_left_trigger = R"doc(Value of the left analog trigger.)doc";

static const char *__doc_sgl_GamepadState_left_x = R"doc(X-axis of the left analog stick.)doc";

static const char *__doc_sgl_GamepadState_left_y = R"doc(Y-axis of the left analog stick.)doc";

static const char *__doc_sgl_GamepadState_right_trigger = R"doc(Value of the right analog trigger.)doc";

static const char *__doc_sgl_GamepadState_right_x = R"doc(X-axis of the right analog stick.)doc";

static const char *__doc_sgl_GamepadState_right_y = R"doc(Y-axis of the right analog stick.)doc";

static const char *__doc_sgl_GamepadState_to_string = R"doc()doc";

static const char *__doc_sgl_HitGroupDesc = R"doc()doc";

static const char *__doc_sgl_HitGroupDesc_any_hit_entry_point = R"doc()doc";

static const char *__doc_sgl_HitGroupDesc_closest_hit_entry_point = R"doc()doc";

static const char *__doc_sgl_HitGroupDesc_hit_group_name = R"doc()doc";

static const char *__doc_sgl_HitGroupDesc_intersection_entry_point = R"doc()doc";

static const char *__doc_sgl_HotReload = R"doc()doc";

static const char *__doc_sgl_HotReload_2 =
R"doc(Shader hot reload management, detects when relevant slang files have
been editor and triggers session recreates as necessary.)doc";

static const char *__doc_sgl_HotReload_HotReload = R"doc()doc";

static const char *__doc_sgl_HotReload_auto_detect_changes = R"doc()doc";

static const char *__doc_sgl_HotReload_auto_detect_delay = R"doc()doc";

static const char *__doc_sgl_HotReload_class_name = R"doc()doc";

static const char *__doc_sgl_HotReload_clear_file_watches = R"doc(Exclusively for testing, erase all existing file watches)doc";

static const char *__doc_sgl_HotReload_has_reloaded = R"doc()doc";

static const char *__doc_sgl_HotReload_last_build_failed =
R"doc(Return true if last attempt to recreate sessions failed with
exception.)doc";

static const char *__doc_sgl_HotReload_m_all_slang_sessions = R"doc()doc";

static const char *__doc_sgl_HotReload_m_auto_detect_changes = R"doc()doc";

static const char *__doc_sgl_HotReload_m_device = R"doc()doc";

static const char *__doc_sgl_HotReload_m_file_system_watcher = R"doc()doc";

static const char *__doc_sgl_HotReload_m_has_reloaded = R"doc()doc";

static const char *__doc_sgl_HotReload_m_last_build_failed = R"doc()doc";

static const char *__doc_sgl_HotReload_m_watched_paths = R"doc()doc";

static const char *__doc_sgl_HotReload_on_file_system_event = R"doc()doc";

static const char *__doc_sgl_HotReload_on_session_modules_changed = R"doc()doc";

static const char *__doc_sgl_HotReload_recreate_all_sessions =
R"doc(Force immediate recreation of all registered sessions and any
modules/programs they've loaded/linked.)doc";

static const char *__doc_sgl_HotReload_register_slang_session = R"doc()doc";

static const char *__doc_sgl_HotReload_reset_reloaded = R"doc()doc";

static const char *__doc_sgl_HotReload_set_auto_detect_changes = R"doc()doc";

static const char *__doc_sgl_HotReload_set_auto_detect_delay = R"doc()doc";

static const char *__doc_sgl_HotReload_unregister_slang_session = R"doc()doc";

static const char *__doc_sgl_HotReload_update = R"doc(Updates internal file system monitor for change detection.)doc";

static const char *__doc_sgl_HotReload_update_watched_paths_for_session = R"doc()doc";

static const char *__doc_sgl_IndexFormat = R"doc()doc";

static const char *__doc_sgl_IndexFormat_info = R"doc()doc";

static const char *__doc_sgl_IndexFormat_uint16 = R"doc()doc";

static const char *__doc_sgl_IndexFormat_uint32 = R"doc()doc";

static const char *__doc_sgl_IndirectDispatchArguments = R"doc()doc";

static const char *__doc_sgl_IndirectDispatchArguments_thread_group_count_x = R"doc()doc";

static const char *__doc_sgl_IndirectDispatchArguments_thread_group_count_y = R"doc()doc";

static const char *__doc_sgl_IndirectDispatchArguments_thread_group_count_z = R"doc()doc";

static const char *__doc_sgl_IndirectDrawArguments = R"doc()doc";

static const char *__doc_sgl_IndirectDrawArguments_instance_count = R"doc()doc";

static const char *__doc_sgl_IndirectDrawArguments_start_instance_location = R"doc()doc";

static const char *__doc_sgl_IndirectDrawArguments_start_vertex_location = R"doc()doc";

static const char *__doc_sgl_IndirectDrawArguments_vertex_count_per_instance = R"doc()doc";

static const char *__doc_sgl_IndirectDrawIndexedArguments = R"doc()doc";

static const char *__doc_sgl_IndirectDrawIndexedArguments_base_vertex_location = R"doc()doc";

static const char *__doc_sgl_IndirectDrawIndexedArguments_index_count_per_instance = R"doc()doc";

static const char *__doc_sgl_IndirectDrawIndexedArguments_instance_count = R"doc()doc";

static const char *__doc_sgl_IndirectDrawIndexedArguments_start_index_location = R"doc()doc";

static const char *__doc_sgl_IndirectDrawIndexedArguments_start_instance_location = R"doc()doc";

static const char *__doc_sgl_InputElementDesc = R"doc()doc";

static const char *__doc_sgl_InputElementDesc_buffer_slot_index = R"doc(The index of the vertex stream to fetch this element's data from.)doc";

static const char *__doc_sgl_InputElementDesc_format = R"doc(The format of the data being fetched for this element.)doc";

static const char *__doc_sgl_InputElementDesc_offset =
R"doc(The offset in bytes of this element from the start of the
corresponding chunk of vertex stream data.)doc";

static const char *__doc_sgl_InputElementDesc_semantic_index =
R"doc(The index of the corresponding parameter in shader code. Only needed
if multiple parameters share a semantic name.)doc";

static const char *__doc_sgl_InputElementDesc_semantic_name = R"doc(The name of the corresponding parameter in shader code.)doc";

static const char *__doc_sgl_InputLayout = R"doc()doc";

static const char *__doc_sgl_InputLayout_2 = R"doc()doc";

static const char *__doc_sgl_InputLayoutDesc = R"doc()doc";

static const char *__doc_sgl_InputLayoutDesc_2 = R"doc()doc";

static const char *__doc_sgl_InputLayoutDesc_input_elements = R"doc()doc";

static const char *__doc_sgl_InputLayoutDesc_vertex_streams = R"doc()doc";

static const char *__doc_sgl_InputLayout_InputLayout = R"doc()doc";

static const char *__doc_sgl_InputLayout_class_name = R"doc()doc";

static const char *__doc_sgl_InputLayout_desc = R"doc()doc";

static const char *__doc_sgl_InputLayout_m_desc = R"doc()doc";

static const char *__doc_sgl_InputLayout_m_rhi_input_layout = R"doc()doc";

static const char *__doc_sgl_InputLayout_rhi_input_layout = R"doc()doc";

static const char *__doc_sgl_InputLayout_to_string = R"doc()doc";

static const char *__doc_sgl_InputSlotClass = R"doc()doc";

static const char *__doc_sgl_InputSlotClass_info = R"doc()doc";

static const char *__doc_sgl_InputSlotClass_per_instance = R"doc()doc";

static const char *__doc_sgl_InputSlotClass_per_vertex = R"doc()doc";

static const char *__doc_sgl_Kernel = R"doc()doc";

static const char *__doc_sgl_Kernel_2 = R"doc()doc";

static const char *__doc_sgl_Kernel_Kernel = R"doc()doc";

static const char *__doc_sgl_Kernel_class_name = R"doc()doc";

static const char *__doc_sgl_Kernel_m_program = R"doc()doc";

static const char *__doc_sgl_Kernel_program = R"doc()doc";

static const char *__doc_sgl_Kernel_reflection = R"doc()doc";

static const char *__doc_sgl_KeyCode = R"doc(Keyboard key codes.)doc";

static const char *__doc_sgl_KeyCode_a = R"doc()doc";

static const char *__doc_sgl_KeyCode_apostrophe = R"doc()doc";

static const char *__doc_sgl_KeyCode_b = R"doc()doc";

static const char *__doc_sgl_KeyCode_backslash = R"doc()doc";

static const char *__doc_sgl_KeyCode_backspace = R"doc()doc";

static const char *__doc_sgl_KeyCode_c = R"doc()doc";

static const char *__doc_sgl_KeyCode_caps_lock = R"doc()doc";

static const char *__doc_sgl_KeyCode_comma = R"doc()doc";

static const char *__doc_sgl_KeyCode_d = R"doc()doc";

static const char *__doc_sgl_KeyCode_delete = R"doc()doc";

static const char *__doc_sgl_KeyCode_down = R"doc()doc";

static const char *__doc_sgl_KeyCode_e = R"doc()doc";

static const char *__doc_sgl_KeyCode_end = R"doc()doc";

static const char *__doc_sgl_KeyCode_enter = R"doc()doc";

static const char *__doc_sgl_KeyCode_equal = R"doc()doc";

static const char *__doc_sgl_KeyCode_escape = R"doc()doc";

static const char *__doc_sgl_KeyCode_f = R"doc()doc";

static const char *__doc_sgl_KeyCode_f1 = R"doc()doc";

static const char *__doc_sgl_KeyCode_f10 = R"doc()doc";

static const char *__doc_sgl_KeyCode_f11 = R"doc()doc";

static const char *__doc_sgl_KeyCode_f12 = R"doc()doc";

static const char *__doc_sgl_KeyCode_f2 = R"doc()doc";

static const char *__doc_sgl_KeyCode_f3 = R"doc()doc";

static const char *__doc_sgl_KeyCode_f4 = R"doc()doc";

static const char *__doc_sgl_KeyCode_f5 = R"doc()doc";

static const char *__doc_sgl_KeyCode_f6 = R"doc()doc";

static const char *__doc_sgl_KeyCode_f7 = R"doc()doc";

static const char *__doc_sgl_KeyCode_f8 = R"doc()doc";

static const char *__doc_sgl_KeyCode_f9 = R"doc()doc";

static const char *__doc_sgl_KeyCode_g = R"doc()doc";

static const char *__doc_sgl_KeyCode_grave_accent = R"doc()doc";

static const char *__doc_sgl_KeyCode_h = R"doc()doc";

static const char *__doc_sgl_KeyCode_home = R"doc()doc";

static const char *__doc_sgl_KeyCode_i = R"doc()doc";

static const char *__doc_sgl_KeyCode_info = R"doc()doc";

static const char *__doc_sgl_KeyCode_insert = R"doc()doc";

static const char *__doc_sgl_KeyCode_j = R"doc()doc";

static const char *__doc_sgl_KeyCode_k = R"doc()doc";

static const char *__doc_sgl_KeyCode_key0 = R"doc()doc";

static const char *__doc_sgl_KeyCode_key1 = R"doc()doc";

static const char *__doc_sgl_KeyCode_key2 = R"doc()doc";

static const char *__doc_sgl_KeyCode_key3 = R"doc()doc";

static const char *__doc_sgl_KeyCode_key4 = R"doc()doc";

static const char *__doc_sgl_KeyCode_key5 = R"doc()doc";

static const char *__doc_sgl_KeyCode_key6 = R"doc()doc";

static const char *__doc_sgl_KeyCode_key7 = R"doc()doc";

static const char *__doc_sgl_KeyCode_key8 = R"doc()doc";

static const char *__doc_sgl_KeyCode_key9 = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad0 = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad1 = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad2 = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad3 = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad4 = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad5 = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad6 = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad7 = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad8 = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad9 = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad_add = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad_del = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad_divide = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad_enter = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad_equal = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad_multiply = R"doc()doc";

static const char *__doc_sgl_KeyCode_keypad_subtract = R"doc()doc";

static const char *__doc_sgl_KeyCode_l = R"doc()doc";

static const char *__doc_sgl_KeyCode_left = R"doc()doc";

static const char *__doc_sgl_KeyCode_left_alt = R"doc()doc";

static const char *__doc_sgl_KeyCode_left_bracket = R"doc()doc";

static const char *__doc_sgl_KeyCode_left_control = R"doc()doc";

static const char *__doc_sgl_KeyCode_left_shift = R"doc()doc";

static const char *__doc_sgl_KeyCode_left_super = R"doc()doc";

static const char *__doc_sgl_KeyCode_m = R"doc()doc";

static const char *__doc_sgl_KeyCode_menu = R"doc()doc";

static const char *__doc_sgl_KeyCode_minus = R"doc()doc";

static const char *__doc_sgl_KeyCode_n = R"doc()doc";

static const char *__doc_sgl_KeyCode_num_lock = R"doc()doc";

static const char *__doc_sgl_KeyCode_o = R"doc()doc";

static const char *__doc_sgl_KeyCode_p = R"doc()doc";

static const char *__doc_sgl_KeyCode_page_down = R"doc()doc";

static const char *__doc_sgl_KeyCode_page_up = R"doc()doc";

static const char *__doc_sgl_KeyCode_pause = R"doc()doc";

static const char *__doc_sgl_KeyCode_period = R"doc()doc";

static const char *__doc_sgl_KeyCode_print_screen = R"doc()doc";

static const char *__doc_sgl_KeyCode_q = R"doc()doc";

static const char *__doc_sgl_KeyCode_r = R"doc()doc";

static const char *__doc_sgl_KeyCode_right = R"doc()doc";

static const char *__doc_sgl_KeyCode_right_alt = R"doc()doc";

static const char *__doc_sgl_KeyCode_right_bracket = R"doc()doc";

static const char *__doc_sgl_KeyCode_right_control = R"doc()doc";

static const char *__doc_sgl_KeyCode_right_shift = R"doc()doc";

static const char *__doc_sgl_KeyCode_right_super = R"doc()doc";

static const char *__doc_sgl_KeyCode_s = R"doc()doc";

static const char *__doc_sgl_KeyCode_scroll_lock = R"doc()doc";

static const char *__doc_sgl_KeyCode_semicolon = R"doc()doc";

static const char *__doc_sgl_KeyCode_slash = R"doc()doc";

static const char *__doc_sgl_KeyCode_space = R"doc()doc";

static const char *__doc_sgl_KeyCode_t = R"doc()doc";

static const char *__doc_sgl_KeyCode_tab = R"doc()doc";

static const char *__doc_sgl_KeyCode_u = R"doc()doc";

static const char *__doc_sgl_KeyCode_unknown = R"doc()doc";

static const char *__doc_sgl_KeyCode_up = R"doc()doc";

static const char *__doc_sgl_KeyCode_v = R"doc()doc";

static const char *__doc_sgl_KeyCode_w = R"doc()doc";

static const char *__doc_sgl_KeyCode_x = R"doc()doc";

static const char *__doc_sgl_KeyCode_y = R"doc()doc";

static const char *__doc_sgl_KeyCode_z = R"doc()doc";

static const char *__doc_sgl_KeyModifier = R"doc(Keyboard modifiers.)doc";

static const char *__doc_sgl_KeyModifierFlags = R"doc(Keyboard modifier flags.)doc";

static const char *__doc_sgl_KeyModifierFlags_alt = R"doc()doc";

static const char *__doc_sgl_KeyModifierFlags_ctrl = R"doc()doc";

static const char *__doc_sgl_KeyModifierFlags_info = R"doc()doc";

static const char *__doc_sgl_KeyModifierFlags_none = R"doc()doc";

static const char *__doc_sgl_KeyModifierFlags_shift = R"doc()doc";

static const char *__doc_sgl_KeyModifier_alt = R"doc()doc";

static const char *__doc_sgl_KeyModifier_ctrl = R"doc()doc";

static const char *__doc_sgl_KeyModifier_info = R"doc()doc";

static const char *__doc_sgl_KeyModifier_shift = R"doc()doc";

static const char *__doc_sgl_KeyboardEvent = R"doc()doc";

static const char *__doc_sgl_KeyboardEvent_2 = R"doc(Keyboard event.)doc";

static const char *__doc_sgl_KeyboardEventType = R"doc(Keyboard event types.)doc";

static const char *__doc_sgl_KeyboardEventType_info = R"doc()doc";

static const char *__doc_sgl_KeyboardEventType_input = R"doc(Character input.)doc";

static const char *__doc_sgl_KeyboardEventType_key_press = R"doc(Key was pressed.)doc";

static const char *__doc_sgl_KeyboardEventType_key_release = R"doc(Key was released.)doc";

static const char *__doc_sgl_KeyboardEventType_key_repeat = R"doc(Key is repeatedly down.)doc";

static const char *__doc_sgl_KeyboardEvent_codepoint = R"doc(UTF-32 codepoint for input events.)doc";

static const char *__doc_sgl_KeyboardEvent_has_modifier = R"doc(Returns true if the specified modifier is set.)doc";

static const char *__doc_sgl_KeyboardEvent_is_input = R"doc(Returns true if this event is an input event.)doc";

static const char *__doc_sgl_KeyboardEvent_is_key_press = R"doc(Returns true if this event is a key press event.)doc";

static const char *__doc_sgl_KeyboardEvent_is_key_release = R"doc(Returns true if this event is a key release event.)doc";

static const char *__doc_sgl_KeyboardEvent_is_key_repeat = R"doc(Returns true if this event is a key repeat event.)doc";

static const char *__doc_sgl_KeyboardEvent_key = R"doc(The key that was pressed/released/repeated.)doc";

static const char *__doc_sgl_KeyboardEvent_mods = R"doc(Keyboard modifier flags.)doc";

static const char *__doc_sgl_KeyboardEvent_to_string = R"doc()doc";

static const char *__doc_sgl_KeyboardEvent_type = R"doc(The event type.)doc";

static const char *__doc_sgl_LoadOp = R"doc()doc";

static const char *__doc_sgl_LoadOp_clear = R"doc()doc";

static const char *__doc_sgl_LoadOp_dont_care = R"doc()doc";

static const char *__doc_sgl_LoadOp_info = R"doc()doc";

static const char *__doc_sgl_LoadOp_load = R"doc()doc";

static const char *__doc_sgl_LogFrequency = R"doc(Log frequency.)doc";

static const char *__doc_sgl_LogFrequency_always = R"doc(Log the message every time.)doc";

static const char *__doc_sgl_LogFrequency_once = R"doc(Log the message only once.)doc";

static const char *__doc_sgl_LogLevel = R"doc(Log level.)doc";

static const char *__doc_sgl_LogLevel_debug = R"doc()doc";

static const char *__doc_sgl_LogLevel_error = R"doc()doc";

static const char *__doc_sgl_LogLevel_fatal = R"doc()doc";

static const char *__doc_sgl_LogLevel_info = R"doc()doc";

static const char *__doc_sgl_LogLevel_none = R"doc()doc";

static const char *__doc_sgl_LogLevel_warn = R"doc()doc";

static const char *__doc_sgl_Logger = R"doc()doc";

static const char *__doc_sgl_Logger_2 = R"doc()doc";

static const char *__doc_sgl_LoggerOutput = R"doc()doc";

static const char *__doc_sgl_LoggerOutput_2 = R"doc(Abstract base class for logger outputs.)doc";

static const char *__doc_sgl_LoggerOutput_class_name = R"doc()doc";

static const char *__doc_sgl_LoggerOutput_write =
R"doc(Write a log message.

Parameter ``level``:
    The log level.

Parameter ``module``:
    The module name.

Parameter ``msg``:
    The message.)doc";

static const char *__doc_sgl_Logger_Logger =
R"doc(Constructor.

Parameter ``level``:
    The log level to use (messages with level >= this will be logged).

Parameter ``name``:
    The name of the logger.

Parameter ``use_default_outputs``:
    Whether to use the default outputs (console + debug console on
    windows).)doc";

static const char *__doc_sgl_Logger_add_console_output =
R"doc(Add a console logger output.

Parameter ``colored``:
    Whether to use colored output.

Returns:
    The created logger output.)doc";

static const char *__doc_sgl_Logger_add_debug_console_output =
R"doc(Add a debug console logger output (Windows only).

Returns:
    The created logger output.)doc";

static const char *__doc_sgl_Logger_add_file_output =
R"doc(Add a file logger output.

Parameter ``path``:
    The path to the log file.

Returns:
    The created logger output.)doc";

static const char *__doc_sgl_Logger_add_output =
R"doc(Add a logger output.

Parameter ``output``:
    The logger output to add.)doc";

static const char *__doc_sgl_Logger_class_name = R"doc()doc";

static const char *__doc_sgl_Logger_create = R"doc()doc";

static const char *__doc_sgl_Logger_debug = R"doc()doc";

static const char *__doc_sgl_Logger_debug_2 = R"doc()doc";

static const char *__doc_sgl_Logger_debug_once = R"doc()doc";

static const char *__doc_sgl_Logger_debug_once_2 = R"doc()doc";

static const char *__doc_sgl_Logger_error = R"doc()doc";

static const char *__doc_sgl_Logger_error_2 = R"doc()doc";

static const char *__doc_sgl_Logger_error_once = R"doc()doc";

static const char *__doc_sgl_Logger_error_once_2 = R"doc()doc";

static const char *__doc_sgl_Logger_fatal = R"doc()doc";

static const char *__doc_sgl_Logger_fatal_2 = R"doc()doc";

static const char *__doc_sgl_Logger_fatal_once = R"doc()doc";

static const char *__doc_sgl_Logger_fatal_once_2 = R"doc()doc";

static const char *__doc_sgl_Logger_get = R"doc(Returns the global logger instance.)doc";

static const char *__doc_sgl_Logger_info = R"doc()doc";

static const char *__doc_sgl_Logger_info_2 = R"doc()doc";

static const char *__doc_sgl_Logger_info_once = R"doc()doc";

static const char *__doc_sgl_Logger_info_once_2 = R"doc()doc";

static const char *__doc_sgl_Logger_is_duplicate = R"doc(Checks if the given message has already been logged.)doc";

static const char *__doc_sgl_Logger_level = R"doc(The log level.)doc";

static const char *__doc_sgl_Logger_log =
R"doc(Log a message.

Parameter ``level``:
    The log level.

Parameter ``msg``:
    The message.

Parameter ``frequency``:
    The log frequency.)doc";

static const char *__doc_sgl_Logger_m_level = R"doc()doc";

static const char *__doc_sgl_Logger_m_messages = R"doc()doc";

static const char *__doc_sgl_Logger_m_mutex = R"doc()doc";

static const char *__doc_sgl_Logger_m_name = R"doc()doc";

static const char *__doc_sgl_Logger_m_outputs = R"doc()doc";

static const char *__doc_sgl_Logger_name = R"doc(The name of the logger.)doc";

static const char *__doc_sgl_Logger_remove_all_outputs = R"doc(Remove all logger outputs.)doc";

static const char *__doc_sgl_Logger_remove_output =
R"doc(Remove a logger output.

Parameter ``output``:
    The logger output to remove.)doc";

static const char *__doc_sgl_Logger_set_level = R"doc()doc";

static const char *__doc_sgl_Logger_set_name = R"doc()doc";

static const char *__doc_sgl_Logger_static_init = R"doc()doc";

static const char *__doc_sgl_Logger_static_shutdown = R"doc()doc";

static const char *__doc_sgl_Logger_use_same_outputs =
R"doc(Use the same outputs as the given logger.

Parameter ``other``:
    Logger to copy outputs from.)doc";

static const char *__doc_sgl_Logger_warn = R"doc()doc";

static const char *__doc_sgl_Logger_warn_2 = R"doc()doc";

static const char *__doc_sgl_Logger_warn_once = R"doc()doc";

static const char *__doc_sgl_Logger_warn_once_2 = R"doc()doc";

static const char *__doc_sgl_LogicOp = R"doc()doc";

static const char *__doc_sgl_LogicOp_info = R"doc()doc";

static const char *__doc_sgl_LogicOp_no_op = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFile = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFile_2 = R"doc(Utility class for reading memory-mapped files.)doc";

static const char *__doc_sgl_MemoryMappedFileStream = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFileStream_2 = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFileStream_MemoryMappedFileStream = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFileStream_class_name = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFileStream_m_file = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFileStream_m_path = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFileStream_path = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFileStream_to_string = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFile_AccessHint = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFile_AccessHint_normal = R"doc(Good overall performance.)doc";

static const char *__doc_sgl_MemoryMappedFile_AccessHint_random = R"doc(Good for random access.)doc";

static const char *__doc_sgl_MemoryMappedFile_AccessHint_sequential = R"doc(Read file once with few seeks.)doc";

static const char *__doc_sgl_MemoryMappedFile_MemoryMappedFile = R"doc(Default constructor. Use open() for opening a file.)doc";

static const char *__doc_sgl_MemoryMappedFile_MemoryMappedFile_2 =
R"doc(Constructor opening a file. Use is_open() to check if successful.

Parameter ``path``:
    Path to open.

Parameter ``mapped_size``:
    Number of bytes to map into memory (automatically clamped to the
    file size).

Parameter ``access_hint``:
    Hint on how memory is accessed.)doc";

static const char *__doc_sgl_MemoryMappedFile_MemoryMappedFile_3 = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFile_MemoryMappedFile_4 = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFile_close = R"doc(Close the file.)doc";

static const char *__doc_sgl_MemoryMappedFile_data = R"doc(Get the mapped data.)doc";

static const char *__doc_sgl_MemoryMappedFile_is_open = R"doc(True, if file successfully opened.)doc";

static const char *__doc_sgl_MemoryMappedFile_m_access_hint = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFile_m_file = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFile_m_mapped_data = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFile_m_mapped_file = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFile_m_mapped_size = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFile_m_path = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFile_m_size = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFile_mapped_size = R"doc(Get the mapped memory size in bytes.)doc";

static const char *__doc_sgl_MemoryMappedFile_open =
R"doc(Open a file.

Parameter ``path``:
    Path to open.

Parameter ``mapped_size``:
    Number of bytes to map into memory (automatically clamped to the
    file size).

Parameter ``access_hint``:
    Hint on how memory is accessed.

Returns:
    True if file was successfully opened.)doc";

static const char *__doc_sgl_MemoryMappedFile_operator_assign = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFile_operator_assign_2 = R"doc()doc";

static const char *__doc_sgl_MemoryMappedFile_page_size = R"doc(Get the OS page size (for remap).)doc";

static const char *__doc_sgl_MemoryMappedFile_remap =
R"doc(Replace mapping by a new one of the same file.

Parameter ``offset``:
    Offset from start of the file in bytes (must be multiple of page
    size).

Parameter ``mapped_size``:
    Size of mapping in bytes.

Returns:
    True if successful.)doc";

static const char *__doc_sgl_MemoryMappedFile_size = R"doc(Get the file size in bytes.)doc";

static const char *__doc_sgl_MemoryStream = R"doc()doc";

static const char *__doc_sgl_MemoryStream_2 = R"doc()doc";

static const char *__doc_sgl_MemoryStream_MemoryStream = R"doc(Create a read/write memory stream with the given initial capacity.)doc";

static const char *__doc_sgl_MemoryStream_MemoryStream_2 = R"doc(Create a read-only memory stream from an existing buffer.)doc";

static const char *__doc_sgl_MemoryStream_MemoryStream_3 = R"doc(Create a read/write memory stream from an existing buffer.)doc";

static const char *__doc_sgl_MemoryStream_capacity = R"doc(Return the current capacity of the underlying buffer.)doc";

static const char *__doc_sgl_MemoryStream_class_name = R"doc()doc";

static const char *__doc_sgl_MemoryStream_close = R"doc()doc";

static const char *__doc_sgl_MemoryStream_data = R"doc(Return the underlying raw buffer.)doc";

static const char *__doc_sgl_MemoryStream_flush = R"doc()doc";

static const char *__doc_sgl_MemoryStream_is_open = R"doc()doc";

static const char *__doc_sgl_MemoryStream_is_readable = R"doc()doc";

static const char *__doc_sgl_MemoryStream_is_writable = R"doc()doc";

static const char *__doc_sgl_MemoryStream_m_capacity = R"doc(Current capacity of the underlying buffer.)doc";

static const char *__doc_sgl_MemoryStream_m_data = R"doc(Pointer to the underlying buffer.)doc";

static const char *__doc_sgl_MemoryStream_m_is_open = R"doc(True if the stream is open.)doc";

static const char *__doc_sgl_MemoryStream_m_is_writable = R"doc(True if the stream is writable.)doc";

static const char *__doc_sgl_MemoryStream_m_owns_data = R"doc(True if the stream owns the underlying buffer.)doc";

static const char *__doc_sgl_MemoryStream_m_pos = R"doc(Current position in the stream.)doc";

static const char *__doc_sgl_MemoryStream_m_size = R"doc(Current size of the stream.)doc";

static const char *__doc_sgl_MemoryStream_owns_data = R"doc(Return true if stream owns the underlying buffer.)doc";

static const char *__doc_sgl_MemoryStream_read = R"doc()doc";

static const char *__doc_sgl_MemoryStream_resize = R"doc()doc";

static const char *__doc_sgl_MemoryStream_seek = R"doc()doc";

static const char *__doc_sgl_MemoryStream_size = R"doc()doc";

static const char *__doc_sgl_MemoryStream_tell = R"doc()doc";

static const char *__doc_sgl_MemoryStream_truncate = R"doc()doc";

static const char *__doc_sgl_MemoryStream_write = R"doc()doc";

static const char *__doc_sgl_MemoryType = R"doc()doc";

static const char *__doc_sgl_MemoryType_device_local = R"doc()doc";

static const char *__doc_sgl_MemoryType_info = R"doc()doc";

static const char *__doc_sgl_MemoryType_read_back = R"doc()doc";

static const char *__doc_sgl_MemoryType_upload = R"doc()doc";

static const char *__doc_sgl_ModifierID = R"doc()doc";

static const char *__doc_sgl_ModifierID_const = R"doc()doc";

static const char *__doc_sgl_ModifierID_differentiable = R"doc()doc";

static const char *__doc_sgl_ModifierID_export = R"doc()doc";

static const char *__doc_sgl_ModifierID_extern = R"doc()doc";

static const char *__doc_sgl_ModifierID_in = R"doc()doc";

static const char *__doc_sgl_ModifierID_info = R"doc()doc";

static const char *__doc_sgl_ModifierID_inout = R"doc()doc";

static const char *__doc_sgl_ModifierID_mutating = R"doc()doc";

static const char *__doc_sgl_ModifierID_nodiff = R"doc()doc";

static const char *__doc_sgl_ModifierID_out = R"doc()doc";

static const char *__doc_sgl_ModifierID_shared = R"doc()doc";

static const char *__doc_sgl_ModifierID_static = R"doc()doc";

static const char *__doc_sgl_MouseButton = R"doc(Mouse buttons.)doc";

static const char *__doc_sgl_MouseButton_info = R"doc()doc";

static const char *__doc_sgl_MouseButton_left = R"doc()doc";

static const char *__doc_sgl_MouseButton_middle = R"doc()doc";

static const char *__doc_sgl_MouseButton_right = R"doc()doc";

static const char *__doc_sgl_MouseButton_unknown = R"doc()doc";

static const char *__doc_sgl_MouseEvent = R"doc()doc";

static const char *__doc_sgl_MouseEvent_2 = R"doc(Mouse event.)doc";

static const char *__doc_sgl_MouseEventType = R"doc(Mouse event types.)doc";

static const char *__doc_sgl_MouseEventType_button_down = R"doc(Mouse button was pressed.)doc";

static const char *__doc_sgl_MouseEventType_button_up = R"doc(Mouse button was released.)doc";

static const char *__doc_sgl_MouseEventType_info = R"doc()doc";

static const char *__doc_sgl_MouseEventType_move = R"doc(Mouse cursor position changed.)doc";

static const char *__doc_sgl_MouseEventType_scroll = R"doc(Mouse wheel was scrolled.)doc";

static const char *__doc_sgl_MouseEvent_button = R"doc(The mouse button that was pressed/released.)doc";

static const char *__doc_sgl_MouseEvent_has_modifier = R"doc(Returns true if the specified modifier is set.)doc";

static const char *__doc_sgl_MouseEvent_is_button_down = R"doc(Returns true if this event is a mouse button down event.)doc";

static const char *__doc_sgl_MouseEvent_is_button_up = R"doc(Returns true if this event is a mouse button up event.)doc";

static const char *__doc_sgl_MouseEvent_is_move = R"doc(Returns true if this event is a mouse move event.)doc";

static const char *__doc_sgl_MouseEvent_is_scroll = R"doc(Returns true if this event is a mouse scroll event.)doc";

static const char *__doc_sgl_MouseEvent_mods = R"doc(Keyboard modifier flags.)doc";

static const char *__doc_sgl_MouseEvent_pos = R"doc(The mouse position.)doc";

static const char *__doc_sgl_MouseEvent_scroll = R"doc(The scroll offset.)doc";

static const char *__doc_sgl_MouseEvent_to_string = R"doc()doc";

static const char *__doc_sgl_MouseEvent_type = R"doc(The event type.)doc";

static const char *__doc_sgl_MultisampleDesc = R"doc()doc";

static const char *__doc_sgl_MultisampleDesc_alpha_to_coverage_enable = R"doc()doc";

static const char *__doc_sgl_MultisampleDesc_alpha_to_one_enable = R"doc()doc";

static const char *__doc_sgl_MultisampleDesc_sample_count = R"doc()doc";

static const char *__doc_sgl_MultisampleDesc_sample_mask = R"doc()doc";

static const char *__doc_sgl_NativeHandle =
R"doc(Represents a native graphics API handle (e.g. D3D12/Vulkan/Metal/CUDA
etc). Native handles are expected to fit into 64 bits. Type
information and conversion from/to native handles is done using type
traits from native_handle_traits.h which needs to be included when
creating and accessing NativeHandle. This separation is done so we
don't expose the heavy backend API headers everywhere.)doc";

static const char *__doc_sgl_NativeHandleTrait = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_2 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_3 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_4 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_5 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_6 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_7 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_8 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_9 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_10 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_11 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_12 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_13 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_14 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_15 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_16 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_17 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_18 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_19 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_20 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_21 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_22 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_2 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_3 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_4 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_5 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_6 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_7 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_8 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_9 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_10 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_11 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_12 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_13 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_14 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_15 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_16 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_17 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_18 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_19 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_pack_20 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_2 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_3 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_4 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_5 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_6 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_7 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_8 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_9 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_10 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_11 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_12 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_13 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_14 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_15 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_16 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_17 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_18 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_19 = R"doc()doc";

static const char *__doc_sgl_NativeHandleTrait_unpack_20 = R"doc()doc";

static const char *__doc_sgl_NativeHandleType = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_CUcontext = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_CUdevice = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_CUdeviceptr = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_CUstream = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_CUtexObject = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_D3D12CommandQueue = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_D3D12CpuDescriptorHandle = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_D3D12Device = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_D3D12DeviceAddress = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_D3D12Fence = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_D3D12GraphicsCommandList = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_D3D12PipelineState = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_D3D12Resource = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_D3D12StateObject = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_MTLAccelerationStructure = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_MTLBuffer = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_MTLCommandBuffer = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_MTLCommandQueue = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_MTLComputePipelineState = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_MTLDevice = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_MTLRenderPipelineState = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_MTLSamplerState = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_MTLSharedEvent = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_MTLTexture = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_OptixDeviceContext = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_OptixTraversableHandle = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_VkAccelerationStructureKHR = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_VkBuffer = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_VkCommandBuffer = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_VkDevice = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_VkImage = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_VkImageView = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_VkInstance = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_VkPhysicalDevice = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_VkPipeline = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_VkQueue = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_VkSampler = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_VkSemaphore = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_WGPUBuffer = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_WGPUCommandBuffer = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_WGPUCommandEncoder = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_WGPUComputePipeline = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_WGPUDevice = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_WGPUQueue = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_WGPURenderPipeline = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_WGPUSampler = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_WGPUTexture = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_WGPUTextureView = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_file_descriptor = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_info = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_undefined = R"doc()doc";

static const char *__doc_sgl_NativeHandleType_win32 = R"doc()doc";

static const char *__doc_sgl_NativeHandle_NativeHandle = R"doc()doc";

static const char *__doc_sgl_NativeHandle_NativeHandle_2 = R"doc()doc";

static const char *__doc_sgl_NativeHandle_NativeHandle_3 = R"doc()doc";

static const char *__doc_sgl_NativeHandle_NativeHandle_4 = R"doc()doc";

static const char *__doc_sgl_NativeHandle_as = R"doc()doc";

static const char *__doc_sgl_NativeHandle_is_valid = R"doc()doc";

static const char *__doc_sgl_NativeHandle_m_type = R"doc()doc";

static const char *__doc_sgl_NativeHandle_m_value = R"doc()doc";

static const char *__doc_sgl_NativeHandle_operator_bool = R"doc()doc";

static const char *__doc_sgl_NativeHandle_to_rhi = R"doc()doc";

static const char *__doc_sgl_NativeHandle_to_string = R"doc()doc";

static const char *__doc_sgl_NativeHandle_type = R"doc()doc";

static const char *__doc_sgl_NativeHandle_value = R"doc()doc";

static const char *__doc_sgl_Object = R"doc()doc";

static const char *__doc_sgl_Object_2 =
R"doc(Object base class with intrusive reference counting

The Object class provides a convenient foundation of a class hierarchy
that will ease lifetime and ownership-related issues whenever Python
bindings are involved.

Internally, its constructor sets the `m_state` field to `1`, which
indicates that the instance is owned by C++. Bits 2..63 of this field
are used to store the actual reference count value. The `inc_ref()`
and `dec_ref()` functions can be used to increment or decrement this
reference count. When `dec_ref()` removes the last reference, the
instance will be deallocated using a `delete` expression handled using
a polymorphic destructor.

When a subclass of `Object` is constructed to Python or returned from
C++ to Python, nanobind will invoke `Object::set_self_py()`, which
hands ownership over to Python/nanobind. Any remaining references will
be moved from the `m_state` field to the Python reference count. In
this mode, `inc_ref()` and `dec_ref()` wrap Python reference counting
primitives (`Py_INCREF()` / `Py_DECREF()`) which must be made
available by calling the function `object_init_py` once during module
initialization. Note that the `m_state` field is also used to store a
pointer to the `PyObject *`. Python instance pointers are always
aligned (i.e. bit 1 is zero), which disambiguates between the two
possible configurations.

Within C++, the RAII helper class `ref` (defined below) can be used to
keep instances alive. This removes the need to call the `inc_ref()` /
`dec_ref()` functions explicitly.

```
{
   ref<MyClass> inst = new MyClass();
   inst->my_function();
   ...
} // end of scope, 'inst' automatically deleted if no longer referenced
```

A separate optional file ``object_py.h`` provides a nanobind type
caster to bind functions taking/returning values of type `ref<T>`.)doc";

static const char *__doc_sgl_Object_Object = R"doc(Default constructor.)doc";

static const char *__doc_sgl_Object_Object_2 =
R"doc(Copy constructor. Note: We don't copy the reference counter, so that
the new object starts with a reference count of 0.)doc";

static const char *__doc_sgl_Object_Object_3 = R"doc(Make the object non-movable.)doc";

static const char *__doc_sgl_Object_class_name =
R"doc(Return the name of the class. Note: This reports the actual class name
if ``SGL_OBJECT``() is used.)doc";

static const char *__doc_sgl_Object_dec_ref = R"doc(Decrease the object's reference count and potentially deallocate it.)doc";

static const char *__doc_sgl_Object_inc_ref = R"doc(Increase the object's reference count.)doc";

static const char *__doc_sgl_Object_m_state = R"doc()doc";

static const char *__doc_sgl_Object_operator_assign =
R"doc(Copy assignment. Note: We don't copy the reference counter, but leave
the reference counts of the two objects unchanged. This results in the
same semantics that we would get if we used `std::shared_ptr` where
the reference counter is stored in a separate place from the object.)doc";

static const char *__doc_sgl_Object_operator_assign_2 = R"doc()doc";

static const char *__doc_sgl_Object_ref_count = R"doc(Return current reference count.)doc";

static const char *__doc_sgl_Object_self_py = R"doc(Return the Python object associated with this instance (or NULL))doc";

static const char *__doc_sgl_Object_set_self_py = R"doc(Set the Python object associated with this instance)doc";

static const char *__doc_sgl_Object_to_string =
R"doc(Return a string representation of this object. This is used for
debugging purposes.)doc";

static const char *__doc_sgl_OwnedSubresourceData = R"doc()doc";

static const char *__doc_sgl_OwnedSubresourceData_owned_data = R"doc()doc";

static const char *__doc_sgl_PassEncoder = R"doc()doc";

static const char *__doc_sgl_PassEncoder_class_name = R"doc()doc";

static const char *__doc_sgl_PassEncoder_end = R"doc()doc";

static const char *__doc_sgl_PassEncoder_insert_debug_marker =
R"doc(Insert a debug marker.

Parameter ``name``:
    Name of the marker.

Parameter ``color``:
    Color of the marker.)doc";

static const char *__doc_sgl_PassEncoder_m_command_encoder = R"doc()doc";

static const char *__doc_sgl_PassEncoder_m_rhi_pass_encoder = R"doc()doc";

static const char *__doc_sgl_PassEncoder_pop_debug_group = R"doc(Pop a debug group.)doc";

static const char *__doc_sgl_PassEncoder_push_debug_group = R"doc(Push a debug group.)doc";

static const char *__doc_sgl_Pipeline = R"doc()doc";

static const char *__doc_sgl_Pipeline_2 = R"doc(Pipeline base class.)doc";

static const char *__doc_sgl_Pipeline_Pipeline = R"doc()doc";

static const char *__doc_sgl_Pipeline_class_name = R"doc()doc";

static const char *__doc_sgl_Pipeline_m_program =
R"doc(Pipelines store program (and thus maintain the ref count) in their
descriptor - this is just so we can register/unregister the with
program. However due to order of destruction this still needs to hold
a strong reference.)doc";

static const char *__doc_sgl_Pipeline_notify_program_reloaded = R"doc()doc";

static const char *__doc_sgl_Pipeline_recreate = R"doc()doc";

static const char *__doc_sgl_PluginManager =
R"doc(Plugin manager for loading plugin libraries and creating plugin
instances.

The plugin system is based around the following principles: - Plugins
are compiled into shared libraries known as _plugin libraries_. - A
_plugin library_ registers one or more _plugin classes_ when being
loaded. - A _plugin class_ needs to inherit from a known _plugin base
class_, implementing it's interface. - A _plugin base class_ defines a
`PluginInfo` struct, describing meta information about each _plugin
class_ as well as a `PluginCreate` typedef used for instantiating
plugin classes.

A _plugin base class_ can be any class that is exported from the main
sgl library. It needs to define a `PluginInfo` struct as well as a
`PluginCreate` typedef. The SGL_PLUGIN_BASE_CLASS macro extends the
class with the required members for the plugin system. For example:

```
class SGL_API PluginBase
{
public:
    struct PluginInfo { const char* desc };
    using PluginCreate = (PluginBase *)(*)();
    SGL_PLUGIN_BASE_CLASS(PluginBase);

    virtual void do_struff() = 0;
};
```

To implement a _plugin class_ type, we simply inherit from the base
class. The SGL_PLUGIN_CLASS macro extends the class with the required
members for the plugin system. The class also needs to implement a
static create function, matching the `PluginCreate` type of the base
class. For example:

```
class PluginA : public PluginBase
{
public:
    SGL_PLUGIN_CLASS(PluginA, "PluginA", PluginInfo({"plugin description string"}));

    static PluginBase* create() { return new PluginA(); }

    void do_struff() override { ... }
};
```

The _plugin library_ must export a `register_plugin` function for
registering all the plugin class types when called.

```
extern "C" SGL_API_EXPORT void register_plugin(sgl::PluginRegistry& registry)
{
    registry.register_class<PluginBase, PluginA>();
}
```

At runtime, the plugin manager can load plugin libraries using
`load_plugin_by_name` and `load_plugin`. New plugin instances can be
created using `create_class`, for example:

```
PluginBase* p = PluginManager::instance().create_class<PluginBase>("PluginA");
```

The `get_infos` function returns a list of plugin infos for all loaded
plugin types of a given plugin base class.)doc";

static const char *__doc_sgl_PluginManager_ClassDesc = R"doc()doc";

static const char *__doc_sgl_PluginManager_ClassDescBase = R"doc()doc";

static const char *__doc_sgl_PluginManager_ClassDescBase_ClassDescBase = R"doc()doc";

static const char *__doc_sgl_PluginManager_ClassDescBase_library = R"doc()doc";

static const char *__doc_sgl_PluginManager_ClassDescBase_type = R"doc()doc";

static const char *__doc_sgl_PluginManager_ClassDesc_ClassDesc = R"doc()doc";

static const char *__doc_sgl_PluginManager_ClassDesc_create = R"doc()doc";

static const char *__doc_sgl_PluginManager_ClassDesc_info = R"doc()doc";

static const char *__doc_sgl_PluginManager_create_class =
R"doc(Create an instance of a plugin class.

Template parameter ``BaseT``:
    The plugin base class.

Template parameter ``Args``:
    The argument type pack used for construction.

Parameter ``type``:
    The plugin type name.

Parameter ``args``:
    Additional arguments passed on construction.

Returns:
    Returns a new instance of the requested plugin type or nullptr if
    not registered.)doc";

static const char *__doc_sgl_PluginManager_find_class_desc = R"doc()doc";

static const char *__doc_sgl_PluginManager_get_infos =
R"doc(Get infos for all registered plugin types for a given plugin base
class.

Template parameter ``BaseT``:
    The plugin base class.

Returns:
    A list of infos.)doc";

static const char *__doc_sgl_PluginManager_has_class =
R"doc(Check if a given type of a plugin is available.

Template parameter ``BaseT``:
    The plugin base class.

Parameter ``type``:
    The plugin type name.

Returns:
    True if plugin type is available.)doc";

static const char *__doc_sgl_PluginManager_instance = R"doc(Singleton accessor.)doc";

static const char *__doc_sgl_PluginManager_load_plugin =
R"doc(Load a plugin library.

Parameter ``path``:
    File path of the plugin library.

Returns:
    True if successful.)doc";

static const char *__doc_sgl_PluginManager_load_plugin_by_name =
R"doc(Load a plugin library by name. This will automatically determine the
plugin path and file extension.

Parameter ``plugin_dir``:
    Path to plugin directory.

Parameter ``name``:
    Name of the plugin library.

Returns:
    True if successful.)doc";

static const char *__doc_sgl_PluginManager_load_plugins_by_name =
R"doc(Load a list of plugin libraries.

Parameter ``plugin_dir``:
    Path to plugin directory.

Parameter ``names``:
    Names of the plugin libraries.)doc";

static const char *__doc_sgl_PluginManager_m_class_descs = R"doc()doc";

static const char *__doc_sgl_PluginManager_m_class_descs_mutex = R"doc()doc";

static const char *__doc_sgl_PluginManager_m_libraries = R"doc()doc";

static const char *__doc_sgl_PluginManager_m_libraries_mutex = R"doc()doc";

static const char *__doc_sgl_PluginManager_register_class = R"doc()doc";

static const char *__doc_sgl_PluginManager_release_all_plugins = R"doc(Release all loaded plugin libraries.)doc";

static const char *__doc_sgl_PluginManager_release_plugin =
R"doc(Release a previously loaded plugin library.

Parameter ``path``:
    File path of the plugin library.

Returns:
    True if successful.)doc";

static const char *__doc_sgl_PluginRegistry = R"doc(Helper class passed to plugin libraries to register plugin classes.)doc";

static const char *__doc_sgl_PluginRegistry_PluginRegistry = R"doc()doc";

static const char *__doc_sgl_PluginRegistry_PluginRegistry_2 = R"doc()doc";

static const char *__doc_sgl_PluginRegistry_m_library = R"doc()doc";

static const char *__doc_sgl_PluginRegistry_m_plugin_manager = R"doc()doc";

static const char *__doc_sgl_PluginRegistry_operator_assign = R"doc()doc";

static const char *__doc_sgl_PluginRegistry_register_class =
R"doc(Register a plugin class. Throws if a class with the same type name
(and same base class) has already been registered.

Template parameter ``BaseT``:
    The plugin base class.

Parameter ``type``:
    The plugin type name.

Parameter ``info``:
    The plugin info (type of BaseT::PluginInfo).

Parameter ``create``:
    The plugin create function (type of BaseT::PluginCreate).)doc";

static const char *__doc_sgl_PluginRegistry_register_class_2 =
R"doc(Register a plugin class. This helper assumes that the plugin class has
a `create` function matching the BaseT::PluginCreate type.

Template parameter ``BaseT``:
    The plugin base class.

Template parameter ``T``:
    The plugin class.)doc";

static const char *__doc_sgl_PrimitiveTopology = R"doc()doc";

static const char *__doc_sgl_PrimitiveTopology_info = R"doc()doc";

static const char *__doc_sgl_PrimitiveTopology_line_list = R"doc()doc";

static const char *__doc_sgl_PrimitiveTopology_line_strip = R"doc()doc";

static const char *__doc_sgl_PrimitiveTopology_patch_list = R"doc()doc";

static const char *__doc_sgl_PrimitiveTopology_point_list = R"doc()doc";

static const char *__doc_sgl_PrimitiveTopology_triangle_list = R"doc()doc";

static const char *__doc_sgl_PrimitiveTopology_triangle_strip = R"doc()doc";

static const char *__doc_sgl_ProgramLayout = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_2 = R"doc()doc";

static const char *__doc_sgl_ProgramLayoutEntryPointList = R"doc()doc";

static const char *__doc_sgl_ProgramLayoutEntryPointList_2 = R"doc(ProgramLayout lazy entry point list evaluation.)doc";

static const char *__doc_sgl_ProgramLayoutEntryPointList_ProgramLayoutEntryPointList = R"doc()doc";

static const char *__doc_sgl_ProgramLayoutEntryPointList_evaluate = R"doc(Get a specific child.)doc";

static const char *__doc_sgl_ProgramLayoutEntryPointList_size = R"doc(Number of entries in list.)doc";

static const char *__doc_sgl_ProgramLayoutParameterList = R"doc()doc";

static const char *__doc_sgl_ProgramLayoutParameterList_2 = R"doc(ProgramLayout lazy parameter list evaluation.)doc";

static const char *__doc_sgl_ProgramLayoutParameterList_ProgramLayoutParameterList = R"doc()doc";

static const char *__doc_sgl_ProgramLayoutParameterList_evaluate = R"doc(Get a specific child.)doc";

static const char *__doc_sgl_ProgramLayoutParameterList_size = R"doc(Number of entries in list.)doc";

static const char *__doc_sgl_ProgramLayout_HashedString = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_HashedString_hash = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_HashedString_string = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_ProgramLayout = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_entry_point_count = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_entry_points = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_find_entry_point_by_name = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_find_entry_point_by_name_2 = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_find_function_by_name =
R"doc(Find a given function by name. Handles generic specilization if
generic variable values are provided.)doc";

static const char *__doc_sgl_ProgramLayout_find_function_by_name_in_type =
R"doc(Find a given function in a type by name. Handles generic specilization
if generic variable values are provided.)doc";

static const char *__doc_sgl_ProgramLayout_find_type_by_name =
R"doc(Find a given type by name. Handles generic specilization if generic
variable values are provided.)doc";

static const char *__doc_sgl_ProgramLayout_from_slang = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_get_entry_point_by_index = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_get_entry_point_by_name = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_get_entry_point_by_name_2 = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_get_hashed_string = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_get_parameter_by_index = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_get_type_layout = R"doc(Get corresponding type layout from a given type.)doc";

static const char *__doc_sgl_ProgramLayout_globals_type_layout = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_globals_variable_layout = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_hashed_string_count = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_hashed_strings = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_hashed_strings_map = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_is_sub_type =
R"doc(Test whether a type is a sub type of another type. Handles both struct
inheritance and interface implementation.)doc";

static const char *__doc_sgl_ProgramLayout_parameter_count = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_parameters = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_to_string = R"doc()doc";

static const char *__doc_sgl_ProgramLayout_type_parameter_count = R"doc()doc";

static const char *__doc_sgl_QueryPool = R"doc()doc";

static const char *__doc_sgl_QueryPool_2 = R"doc()doc";

static const char *__doc_sgl_QueryPoolDesc = R"doc()doc";

static const char *__doc_sgl_QueryPoolDesc_2 = R"doc()doc";

static const char *__doc_sgl_QueryPoolDesc_count = R"doc(Number of queries in the pool.)doc";

static const char *__doc_sgl_QueryPoolDesc_type = R"doc(Query type.)doc";

static const char *__doc_sgl_QueryPool_QueryPool = R"doc()doc";

static const char *__doc_sgl_QueryPool_class_name = R"doc()doc";

static const char *__doc_sgl_QueryPool_desc = R"doc()doc";

static const char *__doc_sgl_QueryPool_get_result = R"doc()doc";

static const char *__doc_sgl_QueryPool_get_results = R"doc()doc";

static const char *__doc_sgl_QueryPool_get_results_2 = R"doc()doc";

static const char *__doc_sgl_QueryPool_get_timestamp_result = R"doc()doc";

static const char *__doc_sgl_QueryPool_get_timestamp_results = R"doc()doc";

static const char *__doc_sgl_QueryPool_get_timestamp_results_2 = R"doc()doc";

static const char *__doc_sgl_QueryPool_m_desc = R"doc()doc";

static const char *__doc_sgl_QueryPool_m_rhi_query_pool = R"doc()doc";

static const char *__doc_sgl_QueryPool_reset = R"doc()doc";

static const char *__doc_sgl_QueryPool_rhi_query_pool = R"doc()doc";

static const char *__doc_sgl_QueryPool_to_string = R"doc()doc";

static const char *__doc_sgl_QueryType = R"doc()doc";

static const char *__doc_sgl_QueryType_acceleration_structure_compacted_size = R"doc()doc";

static const char *__doc_sgl_QueryType_acceleration_structure_current_size = R"doc()doc";

static const char *__doc_sgl_QueryType_acceleration_structure_serialized_size = R"doc()doc";

static const char *__doc_sgl_QueryType_info = R"doc()doc";

static const char *__doc_sgl_QueryType_timestamp = R"doc()doc";

static const char *__doc_sgl_RasterizerDesc = R"doc()doc";

static const char *__doc_sgl_RasterizerDesc_antialiased_line_enable = R"doc()doc";

static const char *__doc_sgl_RasterizerDesc_cull_mode = R"doc()doc";

static const char *__doc_sgl_RasterizerDesc_depth_bias = R"doc()doc";

static const char *__doc_sgl_RasterizerDesc_depth_bias_clamp = R"doc()doc";

static const char *__doc_sgl_RasterizerDesc_depth_clip_enable = R"doc()doc";

static const char *__doc_sgl_RasterizerDesc_enable_conservative_rasterization = R"doc()doc";

static const char *__doc_sgl_RasterizerDesc_fill_mode = R"doc()doc";

static const char *__doc_sgl_RasterizerDesc_forced_sample_count = R"doc()doc";

static const char *__doc_sgl_RasterizerDesc_front_face = R"doc()doc";

static const char *__doc_sgl_RasterizerDesc_multisample_enable = R"doc()doc";

static const char *__doc_sgl_RasterizerDesc_scissor_enable = R"doc()doc";

static const char *__doc_sgl_RasterizerDesc_slope_scaled_depth_bias = R"doc()doc";

static const char *__doc_sgl_Ray = R"doc(Ray type. This should match the layout of DXR RayDesc.)doc";

static const char *__doc_sgl_RayTracingKernel = R"doc()doc";

static const char *__doc_sgl_RayTracingKernel_RayTracingKernel = R"doc()doc";

static const char *__doc_sgl_RayTracingKernel_class_name = R"doc()doc";

static const char *__doc_sgl_RayTracingKernel_dispatch = R"doc()doc";

static const char *__doc_sgl_RayTracingKernel_m_pipeline = R"doc()doc";

static const char *__doc_sgl_RayTracingKernel_pipeline = R"doc()doc";

static const char *__doc_sgl_RayTracingPassEncoder = R"doc()doc";

static const char *__doc_sgl_RayTracingPassEncoder_2 = R"doc()doc";

static const char *__doc_sgl_RayTracingPassEncoder_bind_pipeline = R"doc()doc";

static const char *__doc_sgl_RayTracingPassEncoder_bind_pipeline_2 = R"doc()doc";

static const char *__doc_sgl_RayTracingPassEncoder_class_name = R"doc()doc";

static const char *__doc_sgl_RayTracingPassEncoder_dispatch_rays = R"doc()doc";

static const char *__doc_sgl_RayTracingPassEncoder_end = R"doc()doc";

static const char *__doc_sgl_RayTracingPassEncoder_m_rhi_ray_tracing_pass_encoder = R"doc()doc";

static const char *__doc_sgl_RayTracingPipeline = R"doc()doc";

static const char *__doc_sgl_RayTracingPipeline_2 = R"doc(Ray tracing pipeline.)doc";

static const char *__doc_sgl_RayTracingPipelineDesc = R"doc()doc";

static const char *__doc_sgl_RayTracingPipelineDesc_2 = R"doc()doc";

static const char *__doc_sgl_RayTracingPipelineDesc_flags = R"doc()doc";

static const char *__doc_sgl_RayTracingPipelineDesc_hit_groups = R"doc()doc";

static const char *__doc_sgl_RayTracingPipelineDesc_label = R"doc()doc";

static const char *__doc_sgl_RayTracingPipelineDesc_max_attribute_size = R"doc()doc";

static const char *__doc_sgl_RayTracingPipelineDesc_max_ray_payload_size = R"doc()doc";

static const char *__doc_sgl_RayTracingPipelineDesc_max_recursion = R"doc()doc";

static const char *__doc_sgl_RayTracingPipelineDesc_program = R"doc()doc";

static const char *__doc_sgl_RayTracingPipelineFlags = R"doc()doc";

static const char *__doc_sgl_RayTracingPipelineFlags_info = R"doc()doc";

static const char *__doc_sgl_RayTracingPipelineFlags_none = R"doc()doc";

static const char *__doc_sgl_RayTracingPipelineFlags_skip_procedurals = R"doc()doc";

static const char *__doc_sgl_RayTracingPipelineFlags_skip_triangles = R"doc()doc";

static const char *__doc_sgl_RayTracingPipeline_RayTracingPipeline = R"doc()doc";

static const char *__doc_sgl_RayTracingPipeline_desc = R"doc()doc";

static const char *__doc_sgl_RayTracingPipeline_m_desc = R"doc()doc";

static const char *__doc_sgl_RayTracingPipeline_m_rhi_pipeline = R"doc()doc";

static const char *__doc_sgl_RayTracingPipeline_native_handle = R"doc(Get the native pipeline handle.)doc";

static const char *__doc_sgl_RayTracingPipeline_recreate = R"doc()doc";

static const char *__doc_sgl_RayTracingPipeline_rhi_pipeline = R"doc()doc";

static const char *__doc_sgl_RayTracingPipeline_to_string = R"doc()doc";

static const char *__doc_sgl_Ray_Ray = R"doc(Default constructor (uninitialized).)doc";

static const char *__doc_sgl_Ray_Ray_2 = R"doc(Constructor.)doc";

static const char *__doc_sgl_Ray_dir = R"doc()doc";

static const char *__doc_sgl_Ray_eval = R"doc(Evaluate position on the ray at t.)doc";

static const char *__doc_sgl_Ray_operator_call = R"doc(Evaluate position on the ray at t.)doc";

static const char *__doc_sgl_Ray_origin = R"doc()doc";

static const char *__doc_sgl_Ray_t_max = R"doc()doc";

static const char *__doc_sgl_Ray_t_min = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_ReflectionCursor = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_ReflectionCursor_2 = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_ReflectionCursor_3 = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_ReflectionCursor_4 = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_find_element = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_find_field = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_has_element = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_has_field = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_is_valid = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_m_entry_point_layout = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_m_shader_program = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_m_type_layout = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_m_valid = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_operator_array = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_operator_array_2 = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_operator_ref = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_to_string = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_type = R"doc()doc";

static const char *__doc_sgl_ReflectionCursor_type_layout = R"doc()doc";

static const char *__doc_sgl_RenderPassColorAttachment = R"doc()doc";

static const char *__doc_sgl_RenderPassColorAttachment_clear_value = R"doc()doc";

static const char *__doc_sgl_RenderPassColorAttachment_load_op = R"doc()doc";

static const char *__doc_sgl_RenderPassColorAttachment_resolve_target = R"doc()doc";

static const char *__doc_sgl_RenderPassColorAttachment_store_op = R"doc()doc";

static const char *__doc_sgl_RenderPassColorAttachment_view = R"doc()doc";

static const char *__doc_sgl_RenderPassDepthStencilAttachment = R"doc()doc";

static const char *__doc_sgl_RenderPassDepthStencilAttachment_depth_clear_value = R"doc()doc";

static const char *__doc_sgl_RenderPassDepthStencilAttachment_depth_load_op = R"doc()doc";

static const char *__doc_sgl_RenderPassDepthStencilAttachment_depth_read_only = R"doc()doc";

static const char *__doc_sgl_RenderPassDepthStencilAttachment_depth_store_op = R"doc()doc";

static const char *__doc_sgl_RenderPassDepthStencilAttachment_stencil_clear_value = R"doc()doc";

static const char *__doc_sgl_RenderPassDepthStencilAttachment_stencil_load_op = R"doc()doc";

static const char *__doc_sgl_RenderPassDepthStencilAttachment_stencil_read_only = R"doc()doc";

static const char *__doc_sgl_RenderPassDepthStencilAttachment_stencil_store_op = R"doc()doc";

static const char *__doc_sgl_RenderPassDepthStencilAttachment_view = R"doc()doc";

static const char *__doc_sgl_RenderPassDesc = R"doc()doc";

static const char *__doc_sgl_RenderPassDesc_color_attachments = R"doc()doc";

static const char *__doc_sgl_RenderPassDesc_depth_stencil_attachment = R"doc()doc";

static const char *__doc_sgl_RenderPassEncoder = R"doc()doc";

static const char *__doc_sgl_RenderPassEncoder_2 = R"doc()doc";

static const char *__doc_sgl_RenderPassEncoder_bind_pipeline = R"doc()doc";

static const char *__doc_sgl_RenderPassEncoder_bind_pipeline_2 = R"doc()doc";

static const char *__doc_sgl_RenderPassEncoder_class_name = R"doc()doc";

static const char *__doc_sgl_RenderPassEncoder_draw = R"doc()doc";

static const char *__doc_sgl_RenderPassEncoder_draw_indexed = R"doc()doc";

static const char *__doc_sgl_RenderPassEncoder_draw_indexed_indirect = R"doc()doc";

static const char *__doc_sgl_RenderPassEncoder_draw_indirect = R"doc()doc";

static const char *__doc_sgl_RenderPassEncoder_draw_mesh_tasks = R"doc()doc";

static const char *__doc_sgl_RenderPassEncoder_end = R"doc()doc";

static const char *__doc_sgl_RenderPassEncoder_m_rhi_render_pass_encoder = R"doc()doc";

static const char *__doc_sgl_RenderPassEncoder_set_render_state = R"doc()doc";

static const char *__doc_sgl_RenderPipeline = R"doc()doc";

static const char *__doc_sgl_RenderPipeline_2 = R"doc(Render pipeline.)doc";

static const char *__doc_sgl_RenderPipelineDesc = R"doc()doc";

static const char *__doc_sgl_RenderPipelineDesc_2 = R"doc()doc";

static const char *__doc_sgl_RenderPipelineDesc_depth_stencil = R"doc()doc";

static const char *__doc_sgl_RenderPipelineDesc_input_layout = R"doc()doc";

static const char *__doc_sgl_RenderPipelineDesc_label = R"doc()doc";

static const char *__doc_sgl_RenderPipelineDesc_multisample = R"doc()doc";

static const char *__doc_sgl_RenderPipelineDesc_primitive_topology = R"doc()doc";

static const char *__doc_sgl_RenderPipelineDesc_program = R"doc()doc";

static const char *__doc_sgl_RenderPipelineDesc_rasterizer = R"doc()doc";

static const char *__doc_sgl_RenderPipelineDesc_targets = R"doc()doc";

static const char *__doc_sgl_RenderPipeline_RenderPipeline = R"doc()doc";

static const char *__doc_sgl_RenderPipeline_desc = R"doc()doc";

static const char *__doc_sgl_RenderPipeline_m_desc = R"doc()doc";

static const char *__doc_sgl_RenderPipeline_m_rhi_pipeline = R"doc()doc";

static const char *__doc_sgl_RenderPipeline_m_stored_input_layout = R"doc()doc";

static const char *__doc_sgl_RenderPipeline_native_handle = R"doc(Get the native pipeline handle.)doc";

static const char *__doc_sgl_RenderPipeline_recreate = R"doc()doc";

static const char *__doc_sgl_RenderPipeline_rhi_pipeline = R"doc()doc";

static const char *__doc_sgl_RenderPipeline_to_string = R"doc()doc";

static const char *__doc_sgl_RenderState = R"doc()doc";

static const char *__doc_sgl_RenderState_index_buffer = R"doc()doc";

static const char *__doc_sgl_RenderState_index_format = R"doc()doc";

static const char *__doc_sgl_RenderState_scissor_rects = R"doc()doc";

static const char *__doc_sgl_RenderState_stencil_ref = R"doc()doc";

static const char *__doc_sgl_RenderState_vertex_buffers = R"doc()doc";

static const char *__doc_sgl_RenderState_viewports = R"doc()doc";

static const char *__doc_sgl_RenderTargetWriteMask = R"doc()doc";

static const char *__doc_sgl_RenderTargetWriteMask_all = R"doc()doc";

static const char *__doc_sgl_RenderTargetWriteMask_alpha = R"doc()doc";

static const char *__doc_sgl_RenderTargetWriteMask_blue = R"doc()doc";

static const char *__doc_sgl_RenderTargetWriteMask_green = R"doc()doc";

static const char *__doc_sgl_RenderTargetWriteMask_info = R"doc()doc";

static const char *__doc_sgl_RenderTargetWriteMask_none = R"doc()doc";

static const char *__doc_sgl_RenderTargetWriteMask_red = R"doc()doc";

static const char *__doc_sgl_Resolver = R"doc()doc";

static const char *__doc_sgl_Resolver_resolve = R"doc()doc";

static const char *__doc_sgl_Resource = R"doc()doc";

static const char *__doc_sgl_Resource_2 = R"doc()doc";

static const char *__doc_sgl_ResourceState = R"doc()doc";

static const char *__doc_sgl_ResourceState_acceleration_structure = R"doc()doc";

static const char *__doc_sgl_ResourceState_acceleration_structure_build_output = R"doc()doc";

static const char *__doc_sgl_ResourceState_constant_buffer = R"doc()doc";

static const char *__doc_sgl_ResourceState_copy_destination = R"doc()doc";

static const char *__doc_sgl_ResourceState_copy_source = R"doc()doc";

static const char *__doc_sgl_ResourceState_depth_read = R"doc()doc";

static const char *__doc_sgl_ResourceState_depth_write = R"doc()doc";

static const char *__doc_sgl_ResourceState_general = R"doc()doc";

static const char *__doc_sgl_ResourceState_index_buffer = R"doc()doc";

static const char *__doc_sgl_ResourceState_indirect_argument = R"doc()doc";

static const char *__doc_sgl_ResourceState_info = R"doc()doc";

static const char *__doc_sgl_ResourceState_present = R"doc()doc";

static const char *__doc_sgl_ResourceState_render_target = R"doc()doc";

static const char *__doc_sgl_ResourceState_resolve_destination = R"doc()doc";

static const char *__doc_sgl_ResourceState_resolve_source = R"doc()doc";

static const char *__doc_sgl_ResourceState_shader_resource = R"doc()doc";

static const char *__doc_sgl_ResourceState_stream_output = R"doc()doc";

static const char *__doc_sgl_ResourceState_undefined = R"doc()doc";

static const char *__doc_sgl_ResourceState_unordered_access = R"doc()doc";

static const char *__doc_sgl_ResourceState_vertex_buffer = R"doc()doc";

static const char *__doc_sgl_Resource_Resource = R"doc()doc";

static const char *__doc_sgl_Resource_Resource_2 = R"doc()doc";

static const char *__doc_sgl_Resource_class_name = R"doc()doc";

static const char *__doc_sgl_Resource_native_handle = R"doc(Get the native resource handle.)doc";

static const char *__doc_sgl_Resource_rhi_resource = R"doc()doc";

static const char *__doc_sgl_SHA1 = R"doc(Helper to compute SHA-1 hash.)doc";

static const char *__doc_sgl_SHA1_SHA1 = R"doc()doc";

static const char *__doc_sgl_SHA1_SHA1_2 = R"doc()doc";

static const char *__doc_sgl_SHA1_SHA1_3 = R"doc()doc";

static const char *__doc_sgl_SHA1_SHA1_4 = R"doc()doc";

static const char *__doc_sgl_SHA1_SHA1_5 = R"doc()doc";

static const char *__doc_sgl_SHA1_add_byte = R"doc()doc";

static const char *__doc_sgl_SHA1_digest = R"doc(Return the message digest.)doc";

static const char *__doc_sgl_SHA1_finalize = R"doc()doc";

static const char *__doc_sgl_SHA1_hex_digest = R"doc(Return the message digest as a hex string.)doc";

static const char *__doc_sgl_SHA1_m_bits = R"doc()doc";

static const char *__doc_sgl_SHA1_m_buf = R"doc()doc";

static const char *__doc_sgl_SHA1_m_index = R"doc()doc";

static const char *__doc_sgl_SHA1_m_state = R"doc()doc";

static const char *__doc_sgl_SHA1_operator_assign = R"doc()doc";

static const char *__doc_sgl_SHA1_operator_assign_2 = R"doc()doc";

static const char *__doc_sgl_SHA1_process_block = R"doc()doc";

static const char *__doc_sgl_SHA1_update =
R"doc(Update hash by adding one byte.

Parameter ``byte``:
    Byte to hash.)doc";

static const char *__doc_sgl_SHA1_update_2 =
R"doc(Update hash by adding the given data.

Parameter ``data``:
    Data to hash.

Parameter ``len``:
    Length of data in bytes.)doc";

static const char *__doc_sgl_SHA1_update_3 =
R"doc(Update hash by adding the given string.

Parameter ``str``:
    String to hash.)doc";

static const char *__doc_sgl_SHA1_update_4 =
R"doc(Update hash by adding the given basic value.

Parameter ``value``:
    to hash.)doc";

static const char *__doc_sgl_Sampler = R"doc()doc";

static const char *__doc_sgl_Sampler_2 = R"doc()doc";

static const char *__doc_sgl_SamplerDesc = R"doc()doc";

static const char *__doc_sgl_SamplerDesc_2 = R"doc()doc";

static const char *__doc_sgl_SamplerDesc_address_u = R"doc()doc";

static const char *__doc_sgl_SamplerDesc_address_v = R"doc()doc";

static const char *__doc_sgl_SamplerDesc_address_w = R"doc()doc";

static const char *__doc_sgl_SamplerDesc_border_color = R"doc()doc";

static const char *__doc_sgl_SamplerDesc_comparison_func = R"doc()doc";

static const char *__doc_sgl_SamplerDesc_label = R"doc()doc";

static const char *__doc_sgl_SamplerDesc_mag_filter = R"doc()doc";

static const char *__doc_sgl_SamplerDesc_max_anisotropy = R"doc()doc";

static const char *__doc_sgl_SamplerDesc_max_lod = R"doc()doc";

static const char *__doc_sgl_SamplerDesc_min_filter = R"doc()doc";

static const char *__doc_sgl_SamplerDesc_min_lod = R"doc()doc";

static const char *__doc_sgl_SamplerDesc_mip_filter = R"doc()doc";

static const char *__doc_sgl_SamplerDesc_mip_lod_bias = R"doc()doc";

static const char *__doc_sgl_SamplerDesc_reduction_op = R"doc()doc";

static const char *__doc_sgl_Sampler_Sampler = R"doc()doc";

static const char *__doc_sgl_Sampler_class_name = R"doc()doc";

static const char *__doc_sgl_Sampler_desc = R"doc()doc";

static const char *__doc_sgl_Sampler_descriptor_handle = R"doc()doc";

static const char *__doc_sgl_Sampler_m_desc = R"doc()doc";

static const char *__doc_sgl_Sampler_m_rhi_sampler = R"doc()doc";

static const char *__doc_sgl_Sampler_native_handle = R"doc(Get the native sampler handle.)doc";

static const char *__doc_sgl_Sampler_rhi_sampler = R"doc()doc";

static const char *__doc_sgl_Sampler_to_string = R"doc()doc";

static const char *__doc_sgl_ScissorRect = R"doc()doc";

static const char *__doc_sgl_ScissorRect_from_size = R"doc()doc";

static const char *__doc_sgl_ScissorRect_max_x = R"doc()doc";

static const char *__doc_sgl_ScissorRect_max_y = R"doc()doc";

static const char *__doc_sgl_ScissorRect_min_x = R"doc()doc";

static const char *__doc_sgl_ScissorRect_min_y = R"doc()doc";

static const char *__doc_sgl_ScissorRect_to_string = R"doc()doc";

static const char *__doc_sgl_SearchPathsResolver = R"doc()doc";

static const char *__doc_sgl_SearchPathsResolver_SearchPathsResolver = R"doc()doc";

static const char *__doc_sgl_SearchPathsResolver_m_search_paths = R"doc()doc";

static const char *__doc_sgl_SearchPathsResolver_resolve = R"doc()doc";

static const char *__doc_sgl_ShaderCacheStats = R"doc()doc";

static const char *__doc_sgl_ShaderCacheStats_entry_count = R"doc(Number of entries in the cache.)doc";

static const char *__doc_sgl_ShaderCacheStats_hit_count = R"doc(Number of hits in the cache.)doc";

static const char *__doc_sgl_ShaderCacheStats_miss_count = R"doc(Number of misses in the cache.)doc";

static const char *__doc_sgl_ShaderCursor = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_2 =
R"doc(Cursor used for parsing and setting shader object fields. This class
does *NOT* use the SGL reflection wrappers for accessing due to the
performance implications of allocating/freeing them repeatedly. This
is far faster, however does introduce a risk of mem access problems if
the shader cursor is kept alive longer than the shader object it was
created from.)doc";

static const char *__doc_sgl_ShaderCursor_ShaderCursor = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_ShaderCursor_2 = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_ShaderCursor_3 = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_dereference = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_find_element = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_find_entry_point = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_find_field = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_has_element = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_has_field = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_is_reference = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_is_valid = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_m_offset = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_m_shader_object = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_m_type_layout = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_offset = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_operator_array = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_operator_array_2 = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_operator_assign = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_acceleration_structure = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_array = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_array_unsafe = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_buffer = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_buffer_view = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_cuda_tensor_view = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_data = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_descriptor_handle = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_matrix = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_object = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_pointer = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_sampler = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_scalar = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_texture = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_texture_view = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_set_vector = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_slang_type_layout = R"doc()doc";

static const char *__doc_sgl_ShaderCursor_to_string = R"doc()doc";

static const char *__doc_sgl_ShaderHotReloadEvent = R"doc(Event data for hot reload hook.)doc";

static const char *__doc_sgl_ShaderModel = R"doc()doc";

static const char *__doc_sgl_ShaderModel_info = R"doc()doc";

static const char *__doc_sgl_ShaderModel_sm_6_0 = R"doc()doc";

static const char *__doc_sgl_ShaderModel_sm_6_1 = R"doc()doc";

static const char *__doc_sgl_ShaderModel_sm_6_2 = R"doc()doc";

static const char *__doc_sgl_ShaderModel_sm_6_3 = R"doc()doc";

static const char *__doc_sgl_ShaderModel_sm_6_4 = R"doc()doc";

static const char *__doc_sgl_ShaderModel_sm_6_5 = R"doc()doc";

static const char *__doc_sgl_ShaderModel_sm_6_6 = R"doc()doc";

static const char *__doc_sgl_ShaderModel_sm_6_7 = R"doc()doc";

static const char *__doc_sgl_ShaderModel_unknown = R"doc()doc";

static const char *__doc_sgl_ShaderObject = R"doc()doc";

static const char *__doc_sgl_ShaderObject_2 = R"doc()doc";

static const char *__doc_sgl_ShaderObject_ShaderObject = R"doc()doc";

static const char *__doc_sgl_ShaderObject_class_name = R"doc()doc";

static const char *__doc_sgl_ShaderObject_device = R"doc()doc";

static const char *__doc_sgl_ShaderObject_element_type_layout = R"doc()doc";

static const char *__doc_sgl_ShaderObject_get_cuda_interop_buffers = R"doc()doc";

static const char *__doc_sgl_ShaderObject_get_entry_point = R"doc()doc";

static const char *__doc_sgl_ShaderObject_get_entry_point_count = R"doc()doc";

static const char *__doc_sgl_ShaderObject_get_object = R"doc()doc";

static const char *__doc_sgl_ShaderObject_m_cuda_interop_buffers = R"doc()doc";

static const char *__doc_sgl_ShaderObject_m_device = R"doc()doc";

static const char *__doc_sgl_ShaderObject_m_objects = R"doc()doc";

static const char *__doc_sgl_ShaderObject_m_retain = R"doc()doc";

static const char *__doc_sgl_ShaderObject_m_shader_object = R"doc()doc";

static const char *__doc_sgl_ShaderObject_rhi_shader_object = R"doc()doc";

static const char *__doc_sgl_ShaderObject_set_acceleration_structure = R"doc()doc";

static const char *__doc_sgl_ShaderObject_set_buffer = R"doc()doc";

static const char *__doc_sgl_ShaderObject_set_buffer_view = R"doc()doc";

static const char *__doc_sgl_ShaderObject_set_cuda_tensor_view = R"doc()doc";

static const char *__doc_sgl_ShaderObject_set_data = R"doc()doc";

static const char *__doc_sgl_ShaderObject_set_descriptor_handle = R"doc()doc";

static const char *__doc_sgl_ShaderObject_set_object = R"doc()doc";

static const char *__doc_sgl_ShaderObject_set_sampler = R"doc()doc";

static const char *__doc_sgl_ShaderObject_set_texture = R"doc()doc";

static const char *__doc_sgl_ShaderObject_set_texture_view = R"doc()doc";

static const char *__doc_sgl_ShaderObject_slang_element_type_layout = R"doc()doc";

static const char *__doc_sgl_ShaderOffset =
R"doc(Represents the offset of a shader variable relative to its enclosing
type/buffer/block.

A `ShaderOffset` can be used to store the offset of a shader variable
that might use ordinary/uniform data, resources like
textures/buffers/samplers, or some combination.

A `ShaderOffset` can also encode an invalid offset, to indicate that a
particular shader variable is not present.)doc";

static const char *__doc_sgl_ShaderOffset_ShaderOffset = R"doc(Default constructor. Creates an invalid offset.)doc";

static const char *__doc_sgl_ShaderOffset_ShaderOffset_2 = R"doc(Constructor.)doc";

static const char *__doc_sgl_ShaderOffset_binding_array_index = R"doc()doc";

static const char *__doc_sgl_ShaderOffset_binding_range_index = R"doc()doc";

static const char *__doc_sgl_ShaderOffset_invalid = R"doc(Create an invalid offset.)doc";

static const char *__doc_sgl_ShaderOffset_is_valid = R"doc(Check whether this offset is valid.)doc";

static const char *__doc_sgl_ShaderOffset_operator_add =
R"doc(Adds another offset to this offset. Returns an invalid offset if
either offset is invalid.)doc";

static const char *__doc_sgl_ShaderOffset_operator_le = R"doc(Comparison.)doc";

static const char *__doc_sgl_ShaderOffset_uniform_offset = R"doc()doc";

static const char *__doc_sgl_ShaderOffset_zero = R"doc(Create a zero offset.)doc";

static const char *__doc_sgl_ShaderProgram = R"doc()doc";

static const char *__doc_sgl_ShaderProgram_2 = R"doc()doc";

static const char *__doc_sgl_ShaderProgram_3 = R"doc()doc";

static const char *__doc_sgl_ShaderProgramData = R"doc()doc";

static const char *__doc_sgl_ShaderProgramData_2 = R"doc()doc";

static const char *__doc_sgl_ShaderProgramData_linked_program = R"doc()doc";

static const char *__doc_sgl_ShaderProgramData_rhi_shader_program = R"doc()doc";

static const char *__doc_sgl_ShaderProgramDesc = R"doc()doc";

static const char *__doc_sgl_ShaderProgramDesc_entry_points = R"doc()doc";

static const char *__doc_sgl_ShaderProgramDesc_label = R"doc()doc";

static const char *__doc_sgl_ShaderProgramDesc_link_options = R"doc()doc";

static const char *__doc_sgl_ShaderProgramDesc_modules = R"doc()doc";

static const char *__doc_sgl_ShaderProgram_ShaderProgram = R"doc()doc";

static const char *__doc_sgl_ShaderProgram_class_name = R"doc()doc";

static const char *__doc_sgl_ShaderProgram_desc = R"doc()doc";

static const char *__doc_sgl_ShaderProgram_layout = R"doc()doc";

static const char *__doc_sgl_ShaderProgram_link =
R"doc(Links program and outputs the resulting ShaderProgramData to current
build info.)doc";

static const char *__doc_sgl_ShaderProgram_m_data = R"doc()doc";

static const char *__doc_sgl_ShaderProgram_m_desc = R"doc()doc";

static const char *__doc_sgl_ShaderProgram_m_registered_pipelines = R"doc()doc";

static const char *__doc_sgl_ShaderProgram_m_session = R"doc()doc";

static const char *__doc_sgl_ShaderProgram_reflection = R"doc()doc";

static const char *__doc_sgl_ShaderProgram_register_pipeline = R"doc()doc";

static const char *__doc_sgl_ShaderProgram_rhi_shader_program = R"doc()doc";

static const char *__doc_sgl_ShaderProgram_store_built_data =
R"doc(Finds this program in current build and updates internal m_data to
point at it.)doc";

static const char *__doc_sgl_ShaderProgram_to_string = R"doc()doc";

static const char *__doc_sgl_ShaderProgram_unregister_pipeline = R"doc()doc";

static const char *__doc_sgl_ShaderStage = R"doc()doc";

static const char *__doc_sgl_ShaderStage_amplification = R"doc()doc";

static const char *__doc_sgl_ShaderStage_any_hit = R"doc()doc";

static const char *__doc_sgl_ShaderStage_callable = R"doc()doc";

static const char *__doc_sgl_ShaderStage_closest_hit = R"doc()doc";

static const char *__doc_sgl_ShaderStage_compute = R"doc()doc";

static const char *__doc_sgl_ShaderStage_domain = R"doc()doc";

static const char *__doc_sgl_ShaderStage_fragment = R"doc()doc";

static const char *__doc_sgl_ShaderStage_geometry = R"doc()doc";

static const char *__doc_sgl_ShaderStage_hull = R"doc()doc";

static const char *__doc_sgl_ShaderStage_info = R"doc()doc";

static const char *__doc_sgl_ShaderStage_intersection = R"doc()doc";

static const char *__doc_sgl_ShaderStage_mesh = R"doc()doc";

static const char *__doc_sgl_ShaderStage_miss = R"doc()doc";

static const char *__doc_sgl_ShaderStage_none = R"doc()doc";

static const char *__doc_sgl_ShaderStage_ray_generation = R"doc()doc";

static const char *__doc_sgl_ShaderStage_vertex = R"doc()doc";

static const char *__doc_sgl_ShaderTable = R"doc()doc";

static const char *__doc_sgl_ShaderTable_2 = R"doc()doc";

static const char *__doc_sgl_ShaderTableDesc = R"doc()doc";

static const char *__doc_sgl_ShaderTableDesc_2 = R"doc()doc";

static const char *__doc_sgl_ShaderTableDesc_callable_entry_points = R"doc()doc";

static const char *__doc_sgl_ShaderTableDesc_hit_group_names = R"doc()doc";

static const char *__doc_sgl_ShaderTableDesc_miss_entry_points = R"doc()doc";

static const char *__doc_sgl_ShaderTableDesc_program = R"doc()doc";

static const char *__doc_sgl_ShaderTableDesc_ray_gen_entry_points = R"doc()doc";

static const char *__doc_sgl_ShaderTable_ShaderTable = R"doc()doc";

static const char *__doc_sgl_ShaderTable_class_name = R"doc()doc";

static const char *__doc_sgl_ShaderTable_m_rhi_shader_table = R"doc()doc";

static const char *__doc_sgl_ShaderTable_rhi_shader_table = R"doc()doc";

static const char *__doc_sgl_ShaderTable_to_string = R"doc()doc";

static const char *__doc_sgl_SlangCompileError = R"doc(Exception thrown on compilation errors.)doc";

static const char *__doc_sgl_SlangCompileError_SlangCompileError = R"doc()doc";

static const char *__doc_sgl_SlangCompilerOptions = R"doc(Slang compiler options. Can be set when creating a Slang session.)doc";

static const char *__doc_sgl_SlangCompilerOptions_debug_info =
R"doc(Specifies the level of debug information to include in the generated
code.)doc";

static const char *__doc_sgl_SlangCompilerOptions_defines = R"doc(Specifies a list of preprocessor defines.)doc";

static const char *__doc_sgl_SlangCompilerOptions_disable_warnings = R"doc(Specifies a list of warnings to disable (warning codes or names).)doc";

static const char *__doc_sgl_SlangCompilerOptions_downstream_args =
R"doc(Specifies a list of additional arguments to be passed to the
downstream compiler.)doc";

static const char *__doc_sgl_SlangCompilerOptions_dump_intermediates = R"doc(When set will dump the intermediate source output.)doc";

static const char *__doc_sgl_SlangCompilerOptions_dump_intermediates_prefix = R"doc(The file name prefix for the intermediate source output.)doc";

static const char *__doc_sgl_SlangCompilerOptions_enable_warnings = R"doc(Specifies a list of warnings to enable (warning codes or names).)doc";

static const char *__doc_sgl_SlangCompilerOptions_floating_point_mode = R"doc(Specifies the floating point mode.)doc";

static const char *__doc_sgl_SlangCompilerOptions_include_paths =
R"doc(Specifies a list of include paths to be used when resolving
module/include paths.)doc";

static const char *__doc_sgl_SlangCompilerOptions_matrix_layout = R"doc(Specifies the matrix layout. Defaults to row-major.)doc";

static const char *__doc_sgl_SlangCompilerOptions_optimization = R"doc(Specifies the optimization level.)doc";

static const char *__doc_sgl_SlangCompilerOptions_report_downstream_time = R"doc(Turn on/off downstream compilation time report.)doc";

static const char *__doc_sgl_SlangCompilerOptions_report_perf_benchmark =
R"doc(Turn on/off reporting of time spend in different parts of the
compiler.)doc";

static const char *__doc_sgl_SlangCompilerOptions_shader_model =
R"doc(Specifies the shader model to use. Defaults to latest available on the
device.)doc";

static const char *__doc_sgl_SlangCompilerOptions_skip_spirv_validation =
R"doc(Specifies whether or not to skip the validation step after emitting
SPIRV.)doc";

static const char *__doc_sgl_SlangCompilerOptions_warnings_as_errors =
R"doc(Specifies a list of warnings to be treated as errors (warning codes or
names, or "all" to indicate all warnings).)doc";

static const char *__doc_sgl_SlangComponentType = R"doc()doc";

static const char *__doc_sgl_SlangDebugInfoLevel = R"doc(Slang debug info levels.)doc";

static const char *__doc_sgl_SlangDebugInfoLevel_info = R"doc()doc";

static const char *__doc_sgl_SlangDebugInfoLevel_maximal = R"doc(Emit as much debug infromation as possible for each target.)doc";

static const char *__doc_sgl_SlangDebugInfoLevel_minimal =
R"doc(Emit as little debug information as possible, while still supporting
stack trackes.)doc";

static const char *__doc_sgl_SlangDebugInfoLevel_none = R"doc(No debug information.)doc";

static const char *__doc_sgl_SlangDebugInfoLevel_standard =
R"doc(Emit whatever is the standard level of debug information for each
target.)doc";

static const char *__doc_sgl_SlangEntryPoint = R"doc()doc";

static const char *__doc_sgl_SlangEntryPoint_2 = R"doc()doc";

static const char *__doc_sgl_SlangEntryPointData = R"doc()doc";

static const char *__doc_sgl_SlangEntryPointData_2 = R"doc()doc";

static const char *__doc_sgl_SlangEntryPointData_name = R"doc()doc";

static const char *__doc_sgl_SlangEntryPointData_slang_entry_point = R"doc()doc";

static const char *__doc_sgl_SlangEntryPointData_stage = R"doc()doc";

static const char *__doc_sgl_SlangEntryPointDesc = R"doc()doc";

static const char *__doc_sgl_SlangEntryPointDesc_name = R"doc()doc";

static const char *__doc_sgl_SlangEntryPointDesc_type_conformances = R"doc()doc";

static const char *__doc_sgl_SlangEntryPoint_SlangEntryPoint = R"doc()doc";

static const char *__doc_sgl_SlangEntryPoint_class_name = R"doc()doc";

static const char *__doc_sgl_SlangEntryPoint_desc = R"doc()doc";

static const char *__doc_sgl_SlangEntryPoint_init =
R"doc(Inits slang entry point and outputs the resulting SlangEntryPointData
in current build info.)doc";

static const char *__doc_sgl_SlangEntryPoint_layout = R"doc()doc";

static const char *__doc_sgl_SlangEntryPoint_m_data = R"doc()doc";

static const char *__doc_sgl_SlangEntryPoint_m_desc = R"doc()doc";

static const char *__doc_sgl_SlangEntryPoint_m_module = R"doc()doc";

static const char *__doc_sgl_SlangEntryPoint_module = R"doc()doc";

static const char *__doc_sgl_SlangEntryPoint_name = R"doc()doc";

static const char *__doc_sgl_SlangEntryPoint_populate_build_data = R"doc(Repopulates a build structure with reference to internal m_data ptr.)doc";

static const char *__doc_sgl_SlangEntryPoint_rename = R"doc()doc";

static const char *__doc_sgl_SlangEntryPoint_slang_entry_point = R"doc()doc";

static const char *__doc_sgl_SlangEntryPoint_stage = R"doc()doc";

static const char *__doc_sgl_SlangEntryPoint_store_built_data =
R"doc(Finds this entry point in current build and updates internal m_data to
point at it.)doc";

static const char *__doc_sgl_SlangEntryPoint_to_string = R"doc()doc";

static const char *__doc_sgl_SlangEntryPoint_with_name = R"doc(Returns a copy of the entry point with a new name.)doc";

static const char *__doc_sgl_SlangFloatingPointMode = R"doc(Slang floating point modes.)doc";

static const char *__doc_sgl_SlangFloatingPointMode_default = R"doc()doc";

static const char *__doc_sgl_SlangFloatingPointMode_fast = R"doc()doc";

static const char *__doc_sgl_SlangFloatingPointMode_info = R"doc()doc";

static const char *__doc_sgl_SlangFloatingPointMode_precise = R"doc()doc";

static const char *__doc_sgl_SlangGlobalScope = R"doc()doc";

static const char *__doc_sgl_SlangLinkOptions =
R"doc(Slang link options. These can optionally be set when linking a shader
program.)doc";

static const char *__doc_sgl_SlangLinkOptions_debug_info =
R"doc(Specifies the level of debug information to include in the generated
code.)doc";

static const char *__doc_sgl_SlangLinkOptions_downstream_args =
R"doc(Specifies a list of additional arguments to be passed to the
downstream compiler.)doc";

static const char *__doc_sgl_SlangLinkOptions_dump_intermediates = R"doc(When set will dump the intermediate source output.)doc";

static const char *__doc_sgl_SlangLinkOptions_dump_intermediates_prefix = R"doc(The file name prefix for the intermediate source output.)doc";

static const char *__doc_sgl_SlangLinkOptions_floating_point_mode = R"doc(Specifies the floating point mode.)doc";

static const char *__doc_sgl_SlangLinkOptions_optimization = R"doc(Specifies the optimization level.)doc";

static const char *__doc_sgl_SlangMatrixLayout = R"doc(Slang matrix layout modes.)doc";

static const char *__doc_sgl_SlangMatrixLayout_column_major = R"doc()doc";

static const char *__doc_sgl_SlangMatrixLayout_info = R"doc()doc";

static const char *__doc_sgl_SlangMatrixLayout_row_major = R"doc()doc";

static const char *__doc_sgl_SlangModule = R"doc()doc";

static const char *__doc_sgl_SlangModule_2 = R"doc()doc";

static const char *__doc_sgl_SlangModuleData = R"doc()doc";

static const char *__doc_sgl_SlangModuleData_2 = R"doc()doc";

static const char *__doc_sgl_SlangModuleData_name = R"doc()doc";

static const char *__doc_sgl_SlangModuleData_path = R"doc()doc";

static const char *__doc_sgl_SlangModuleData_slang_module = R"doc()doc";

static const char *__doc_sgl_SlangModuleDesc = R"doc()doc";

static const char *__doc_sgl_SlangModuleDesc_module_name = R"doc(Required module name)doc";

static const char *__doc_sgl_SlangModuleDesc_path = R"doc(If source specified, additional path for compilation.)doc";

static const char *__doc_sgl_SlangModuleDesc_source =
R"doc(Optional module source. If not specified slang module resolution is
used.)doc";

static const char *__doc_sgl_SlangModule_SlangModule = R"doc()doc";

static const char *__doc_sgl_SlangModule_break_strong_reference_to_session =
R"doc(Unlinks the session reference for modules that are referred to by the
session to avoid ref loops.)doc";

static const char *__doc_sgl_SlangModule_class_name = R"doc()doc";

static const char *__doc_sgl_SlangModule_data = R"doc()doc";

static const char *__doc_sgl_SlangModule_desc = R"doc(Descriptor that holds all data required to create this module.)doc";

static const char *__doc_sgl_SlangModule_entry_point = R"doc(Get an entry point, optionally applying type conformances to it.)doc";

static const char *__doc_sgl_SlangModule_entry_points = R"doc(Build and return vector of all current entry points in the module.)doc";

static const char *__doc_sgl_SlangModule_has_entry_point = R"doc()doc";

static const char *__doc_sgl_SlangModule_layout = R"doc()doc";

static const char *__doc_sgl_SlangModule_load =
R"doc(Loads slang module and outputs the resulting SlangModuleData in
current build info.)doc";

static const char *__doc_sgl_SlangModule_m_data = R"doc()doc";

static const char *__doc_sgl_SlangModule_m_desc = R"doc()doc";

static const char *__doc_sgl_SlangModule_m_registered_entry_points = R"doc()doc";

static const char *__doc_sgl_SlangModule_m_session = R"doc()doc";

static const char *__doc_sgl_SlangModule_module_decl = R"doc(Get root decl ref for this module)doc";

static const char *__doc_sgl_SlangModule_name = R"doc(Module name.)doc";

static const char *__doc_sgl_SlangModule_path =
R"doc(Module source path. This can be empty if the module was generated from
a string.)doc";

static const char *__doc_sgl_SlangModule_populate_build_data = R"doc(Repopulates a build structure with reference to internal m_data ptr.)doc";

static const char *__doc_sgl_SlangModule_register_entry_point = R"doc()doc";

static const char *__doc_sgl_SlangModule_session = R"doc(The session from which this module was built.)doc";

static const char *__doc_sgl_SlangModule_slang_module = R"doc(Internal slang module.)doc";

static const char *__doc_sgl_SlangModule_store_built_data =
R"doc(Finds this module in current build and updates internal m_data to
point at it.)doc";

static const char *__doc_sgl_SlangModule_to_string = R"doc(Text summary of the module.)doc";

static const char *__doc_sgl_SlangModule_unregister_entry_point = R"doc()doc";

static const char *__doc_sgl_SlangOptimizationLevel = R"doc(Slang optimization levels.)doc";

static const char *__doc_sgl_SlangOptimizationLevel_default = R"doc(Default optimization level: balance code quality and compilation time.)doc";

static const char *__doc_sgl_SlangOptimizationLevel_high = R"doc(Optimize aggressively.)doc";

static const char *__doc_sgl_SlangOptimizationLevel_info = R"doc()doc";

static const char *__doc_sgl_SlangOptimizationLevel_maximal =
R"doc(Include optimizations that may take a very long time, or may involve
severe space-vs-speed tradeoffs.)doc";

static const char *__doc_sgl_SlangOptimizationLevel_none = R"doc(No optimizations.)doc";

static const char *__doc_sgl_SlangSession = R"doc()doc";

static const char *__doc_sgl_SlangSession_2 = R"doc(A slang session, used to load modules and link programs.)doc";

static const char *__doc_sgl_SlangSessionBuild =
R"doc(Intermediate structure used during a build that stores new session,
module, program and entry point information. This is populated during
a build, then applied to all built modules/entrypoints/programs at
once on success.)doc";

static const char *__doc_sgl_SlangSessionBuild_entry_points = R"doc()doc";

static const char *__doc_sgl_SlangSessionBuild_modules = R"doc()doc";

static const char *__doc_sgl_SlangSessionBuild_programs = R"doc()doc";

static const char *__doc_sgl_SlangSessionBuild_session = R"doc()doc";

static const char *__doc_sgl_SlangSessionData = R"doc(Internal data stored once the slang session has been created.)doc";

static const char *__doc_sgl_SlangSessionData_2 = R"doc(Internal data stored once the slang session has been created.)doc";

static const char *__doc_sgl_SlangSessionData_cache_enabled = R"doc(True if session cache is enabled.)doc";

static const char *__doc_sgl_SlangSessionData_cache_include_paths = R"doc(One cache path for each include path under the root cache path.)doc";

static const char *__doc_sgl_SlangSessionData_cache_path = R"doc(Cache root path.)doc";

static const char *__doc_sgl_SlangSessionData_include_paths = R"doc(List of include paths used for resolving module/include paths.)doc";

static const char *__doc_sgl_SlangSessionData_loaded_modules = R"doc(Set of all currently loaded slang modules.)doc";

static const char *__doc_sgl_SlangSessionData_resolve_module_name =
R"doc(Finds fully qualified module name by scanning the cache and include
paths.)doc";

static const char *__doc_sgl_SlangSessionData_slang_session = R"doc(Pointer to internal slang session.)doc";

static const char *__doc_sgl_SlangSessionData_uid = R"doc(Unique session hash.)doc";

static const char *__doc_sgl_SlangSessionDesc = R"doc()doc";

static const char *__doc_sgl_SlangSessionDesc_2 = R"doc(Descriptor for slang session initialization.)doc";

static const char *__doc_sgl_SlangSessionDesc_add_default_include_paths = R"doc()doc";

static const char *__doc_sgl_SlangSessionDesc_cache_path = R"doc()doc";

static const char *__doc_sgl_SlangSessionDesc_compiler_options = R"doc()doc";

static const char *__doc_sgl_SlangSession_SlangSession = R"doc()doc";

static const char *__doc_sgl_SlangSession_class_name = R"doc()doc";

static const char *__doc_sgl_SlangSession_create_session = R"doc()doc";

static const char *__doc_sgl_SlangSession_data = R"doc()doc";

static const char *__doc_sgl_SlangSession_desc = R"doc()doc";

static const char *__doc_sgl_SlangSession_device = R"doc()doc";

static const char *__doc_sgl_SlangSession_get_slang_session = R"doc()doc";

static const char *__doc_sgl_SlangSession_link_program = R"doc(Link a program with a set of modules and entry points.)doc";

static const char *__doc_sgl_SlangSession_load_module = R"doc(Load a module by name.)doc";

static const char *__doc_sgl_SlangSession_load_module_from_source = R"doc(Load a module from string source code.)doc";

static const char *__doc_sgl_SlangSession_load_program =
R"doc(Load a program from a given module with a set of entry points.
Internally this simply wraps link_program without requiring the user
to explicitly load modules.)doc";

static const char *__doc_sgl_SlangSession_load_source = R"doc(Load the source code for a given module.)doc";

static const char *__doc_sgl_SlangSession_m_data = R"doc(Data pointer contains all slang pointers + data resulting from build.)doc";

static const char *__doc_sgl_SlangSession_m_desc = R"doc(Descriptor containing all info required to build the session.)doc";

static const char *__doc_sgl_SlangSession_m_device = R"doc()doc";

static const char *__doc_sgl_SlangSession_m_nvapi_module = R"doc(Global NVAPI module linked to all programs.)doc";

static const char *__doc_sgl_SlangSession_m_registered_modules =
R"doc(All loaded sgl modules (wrappers around IModule returned from
load_module). Note: this is a vector, as order of creation matters.)doc";

static const char *__doc_sgl_SlangSession_m_registered_programs = R"doc(All created sgl programs (via link_program))doc";

static const char *__doc_sgl_SlangSession_recreate_session =
R"doc(Fully recreates this session and any loaded modules or linked
programs.)doc";

static const char *__doc_sgl_SlangSession_register_module = R"doc()doc";

static const char *__doc_sgl_SlangSession_register_program = R"doc()doc";

static const char *__doc_sgl_SlangSession_to_string = R"doc()doc";

static const char *__doc_sgl_SlangSession_unregister_module = R"doc()doc";

static const char *__doc_sgl_SlangSession_unregister_program = R"doc()doc";

static const char *__doc_sgl_SlangSession_update_module_cache_and_dependencies = R"doc()doc";

static const char *__doc_sgl_SlangSession_write_module_to_cache = R"doc()doc";

static const char *__doc_sgl_StencilOp = R"doc()doc";

static const char *__doc_sgl_StencilOp_decrement_saturate = R"doc()doc";

static const char *__doc_sgl_StencilOp_decrement_wrap = R"doc()doc";

static const char *__doc_sgl_StencilOp_increment_saturate = R"doc()doc";

static const char *__doc_sgl_StencilOp_increment_wrap = R"doc()doc";

static const char *__doc_sgl_StencilOp_info = R"doc()doc";

static const char *__doc_sgl_StencilOp_invert = R"doc()doc";

static const char *__doc_sgl_StencilOp_keep = R"doc()doc";

static const char *__doc_sgl_StencilOp_replace = R"doc()doc";

static const char *__doc_sgl_StencilOp_zero = R"doc()doc";

static const char *__doc_sgl_StoreOp = R"doc()doc";

static const char *__doc_sgl_StoreOp_dont_care = R"doc()doc";

static const char *__doc_sgl_StoreOp_info = R"doc()doc";

static const char *__doc_sgl_StoreOp_store = R"doc()doc";

static const char *__doc_sgl_Stream = R"doc()doc";

static const char *__doc_sgl_Stream_2 = R"doc(Base class for all stream objects.)doc";

static const char *__doc_sgl_Stream_class_name = R"doc()doc";

static const char *__doc_sgl_Stream_close =
R"doc(Close the stream. No further read/writes are allowed. \note This is a
no-op if the stream is already closed.)doc";

static const char *__doc_sgl_Stream_flush = R"doc(Flush the internal buffers.)doc";

static const char *__doc_sgl_Stream_is_open = R"doc(Returns true if the stream is open.)doc";

static const char *__doc_sgl_Stream_is_readable = R"doc(Returns true if the stream is readable.)doc";

static const char *__doc_sgl_Stream_is_writable = R"doc(Returns true if the stream is writable.)doc";

static const char *__doc_sgl_Stream_read =
R"doc(Read data from the stream. Throws an exception if not all data could
be read.)doc";

static const char *__doc_sgl_Stream_seek = R"doc(Seeks a position within the stream.)doc";

static const char *__doc_sgl_Stream_size = R"doc(Get the size of the stream.)doc";

static const char *__doc_sgl_Stream_tell = R"doc(Get the current position in the stream.)doc";

static const char *__doc_sgl_Stream_truncate =
R"doc(Truncate the stream to the given size. Throws an exception if stream
is in read-only mode or does not support truncation.)doc";

static const char *__doc_sgl_Stream_write =
R"doc(Write data to the stream. Throws an exception if not all data could be
written.)doc";

static const char *__doc_sgl_SubresourceData = R"doc()doc";

static const char *__doc_sgl_SubresourceData_data = R"doc()doc";

static const char *__doc_sgl_SubresourceData_row_pitch = R"doc()doc";

static const char *__doc_sgl_SubresourceData_size = R"doc()doc";

static const char *__doc_sgl_SubresourceData_slice_pitch = R"doc()doc";

static const char *__doc_sgl_SubresourceLayout = R"doc()doc";

static const char *__doc_sgl_SubresourceLayout_block_height = R"doc(Block height in texels (1 for uncompressed formats).)doc";

static const char *__doc_sgl_SubresourceLayout_block_width = R"doc(Block width in texels (1 for uncompressed formats).)doc";

static const char *__doc_sgl_SubresourceLayout_col_pitch =
R"doc(Stride in bytes between columns (i.e. blocks) of the subresource
tensor.)doc";

static const char *__doc_sgl_SubresourceLayout_row_count =
R"doc(Number of rows. For uncompressed formats this matches size.y. For
compressed formats this matches align_up(size.y, block_height) /
block_height.)doc";

static const char *__doc_sgl_SubresourceLayout_row_pitch = R"doc(Stride in bytes between rows of the subresource tensor.)doc";

static const char *__doc_sgl_SubresourceLayout_size = R"doc(Dimensions of the subresource (in texels).)doc";

static const char *__doc_sgl_SubresourceLayout_size_in_bytes =
R"doc(Overall size required to fit the subresource data (typically size.z *
slice_pitch).)doc";

static const char *__doc_sgl_SubresourceLayout_slice_pitch = R"doc(Stride in bytes between slices of the subresource tensor.)doc";

static const char *__doc_sgl_SubresourceRange = R"doc()doc";

static const char *__doc_sgl_SubresourceRange_layer = R"doc(First array layer.)doc";

static const char *__doc_sgl_SubresourceRange_layer_count = R"doc(Number of array layers.)doc";

static const char *__doc_sgl_SubresourceRange_mip = R"doc(Most detailed mip level.)doc";

static const char *__doc_sgl_SubresourceRange_mip_count = R"doc(Number of mip levels.)doc";

static const char *__doc_sgl_SubresourceRange_operator_le = R"doc()doc";

static const char *__doc_sgl_SubresourceRange_to_string = R"doc()doc";

static const char *__doc_sgl_Surface = R"doc()doc";

static const char *__doc_sgl_Surface_2 = R"doc()doc";

static const char *__doc_sgl_SurfaceConfig = R"doc()doc";

static const char *__doc_sgl_SurfaceConfig_2 = R"doc()doc";

static const char *__doc_sgl_SurfaceConfig_desired_image_count = R"doc(Desired number of images.)doc";

static const char *__doc_sgl_SurfaceConfig_format = R"doc(Surface texture format.)doc";

static const char *__doc_sgl_SurfaceConfig_height = R"doc(Surface texture height.)doc";

static const char *__doc_sgl_SurfaceConfig_usage = R"doc(Surface texture usage.)doc";

static const char *__doc_sgl_SurfaceConfig_vsync = R"doc(Enable/disable vertical synchronization.)doc";

static const char *__doc_sgl_SurfaceConfig_width = R"doc(Surface texture width.)doc";

static const char *__doc_sgl_SurfaceInfo = R"doc()doc";

static const char *__doc_sgl_SurfaceInfo_2 = R"doc()doc";

static const char *__doc_sgl_SurfaceInfo_formats = R"doc(Supported texture formats.)doc";

static const char *__doc_sgl_SurfaceInfo_preferred_format = R"doc(Preferred format for the surface.)doc";

static const char *__doc_sgl_SurfaceInfo_supported_usage = R"doc(Supported texture usages.)doc";

static const char *__doc_sgl_Surface_Surface = R"doc()doc";

static const char *__doc_sgl_Surface_Surface_2 = R"doc()doc";

static const char *__doc_sgl_Surface_acquire_next_image = R"doc(Acquries the next surface image.)doc";

static const char *__doc_sgl_Surface_class_name = R"doc()doc";

static const char *__doc_sgl_Surface_config = R"doc(Returns the surface config.)doc";

static const char *__doc_sgl_Surface_configure = R"doc(Configure the surface.)doc";

static const char *__doc_sgl_Surface_get_images = R"doc()doc";

static const char *__doc_sgl_Surface_info = R"doc(Returns the surface info.)doc";

static const char *__doc_sgl_Surface_m_config = R"doc()doc";

static const char *__doc_sgl_Surface_m_device = R"doc()doc";

static const char *__doc_sgl_Surface_m_info = R"doc()doc";

static const char *__doc_sgl_Surface_m_rhi_surface = R"doc()doc";

static const char *__doc_sgl_Surface_present = R"doc(Present the previously acquire image.)doc";

static const char *__doc_sgl_Surface_unconfigure = R"doc(Unconfigure the surface.)doc";

static const char *__doc_sgl_Texture = R"doc()doc";

static const char *__doc_sgl_Texture_2 = R"doc()doc";

static const char *__doc_sgl_TextureAddressingMode = R"doc()doc";

static const char *__doc_sgl_TextureAddressingMode_clamp_to_border = R"doc()doc";

static const char *__doc_sgl_TextureAddressingMode_clamp_to_edge = R"doc()doc";

static const char *__doc_sgl_TextureAddressingMode_info = R"doc()doc";

static const char *__doc_sgl_TextureAddressingMode_mirror_once = R"doc()doc";

static const char *__doc_sgl_TextureAddressingMode_mirror_repeat = R"doc()doc";

static const char *__doc_sgl_TextureAddressingMode_wrap = R"doc()doc";

static const char *__doc_sgl_TextureAspect = R"doc()doc";

static const char *__doc_sgl_TextureAspect_all = R"doc()doc";

static const char *__doc_sgl_TextureAspect_depth_only = R"doc()doc";

static const char *__doc_sgl_TextureAspect_info = R"doc()doc";

static const char *__doc_sgl_TextureAspect_stencil_only = R"doc()doc";

static const char *__doc_sgl_TextureDesc = R"doc()doc";

static const char *__doc_sgl_TextureDesc_2 = R"doc()doc";

static const char *__doc_sgl_TextureDesc_array_length = R"doc(Array length.)doc";

static const char *__doc_sgl_TextureDesc_data = R"doc()doc";

static const char *__doc_sgl_TextureDesc_default_state = R"doc()doc";

static const char *__doc_sgl_TextureDesc_depth = R"doc(Depth in pixels.)doc";

static const char *__doc_sgl_TextureDesc_format = R"doc(Texture format.)doc";

static const char *__doc_sgl_TextureDesc_height = R"doc(Height in pixels.)doc";

static const char *__doc_sgl_TextureDesc_label = R"doc(Debug label.)doc";

static const char *__doc_sgl_TextureDesc_memory_type = R"doc()doc";

static const char *__doc_sgl_TextureDesc_mip_count = R"doc(Number of mip levels (ALL_MIPS for all mip levels).)doc";

static const char *__doc_sgl_TextureDesc_sample_count = R"doc(Number of samples per pixel.)doc";

static const char *__doc_sgl_TextureDesc_sample_quality = R"doc(Quality level for multisampled textures.)doc";

static const char *__doc_sgl_TextureDesc_type = R"doc(Texture type.)doc";

static const char *__doc_sgl_TextureDesc_usage = R"doc()doc";

static const char *__doc_sgl_TextureDesc_width = R"doc(Width in pixels.)doc";

static const char *__doc_sgl_TextureFilteringMode = R"doc()doc";

static const char *__doc_sgl_TextureFilteringMode_info = R"doc()doc";

static const char *__doc_sgl_TextureFilteringMode_linear = R"doc()doc";

static const char *__doc_sgl_TextureFilteringMode_point = R"doc()doc";

static const char *__doc_sgl_TextureLoader = R"doc()doc";

static const char *__doc_sgl_TextureLoader_2 = R"doc(Utility class for loading textures from bitmaps and image files.)doc";

static const char *__doc_sgl_TextureLoader_Options = R"doc()doc";

static const char *__doc_sgl_TextureLoader_Options_Options = R"doc()doc";

static const char *__doc_sgl_TextureLoader_Options_allocate_mips = R"doc(Allocate mip levels for the texture.)doc";

static const char *__doc_sgl_TextureLoader_Options_extend_alpha = R"doc(Extend RGB to RGBA if RGB texture format is not available.)doc";

static const char *__doc_sgl_TextureLoader_Options_generate_mips = R"doc(Generate mip levels for the texture.)doc";

static const char *__doc_sgl_TextureLoader_Options_load_as_normalized = R"doc(Load 8/16-bit integer data as normalized resource format.)doc";

static const char *__doc_sgl_TextureLoader_Options_load_as_srgb =
R"doc(Use ``Format::rgba8_unorm_srgb`` format if bitmap is 8-bit RGBA with
sRGB gamma.)doc";

static const char *__doc_sgl_TextureLoader_Options_usage =
R"doc(Resource usage flags for the texture. ``TextureUsage::render_target``
will be added automatically if ``generate_mips`` is true.)doc";

static const char *__doc_sgl_TextureLoader_TextureLoader = R"doc()doc";

static const char *__doc_sgl_TextureLoader_class_name = R"doc()doc";

static const char *__doc_sgl_TextureLoader_load_texture =
R"doc(Load a texture from a bitmap.

Parameter ``bitmap``:
    Bitmap to load.

Parameter ``options``:
    Texture loading options.

Returns:
    New texture object.)doc";

static const char *__doc_sgl_TextureLoader_load_texture_2 =
R"doc(Load a texture from an image file.

Parameter ``path``:
    Image file path.

Parameter ``options``:
    Texture loading options.

Returns:
    New texture object.)doc";

static const char *__doc_sgl_TextureLoader_load_texture_array =
R"doc(Load a texture array from a list of bitmaps.

All bitmaps need to have the same format and dimensions.

Parameter ``bitmaps``:
    Bitmaps to load.

Parameter ``options``:
    Texture loading options.

Returns:
    New texture array object.)doc";

static const char *__doc_sgl_TextureLoader_load_texture_array_2 =
R"doc(Load a texture array from a list of image files.

All images need to have the same format and dimensions.

Parameter ``paths``:
    Image file paths.

Parameter ``options``:
    Texture loading options.

Returns:
    New texture array object.)doc";

static const char *__doc_sgl_TextureLoader_load_textures =
R"doc(Load textures from a list of bitmaps.

Parameter ``bitmaps``:
    Bitmaps to load.

Parameter ``options``:
    Texture loading options.

Returns:
    List of new of texture objects.)doc";

static const char *__doc_sgl_TextureLoader_load_textures_2 =
R"doc(Load textures from a list of image files.

Parameter ``paths``:
    Image file paths.

Parameter ``options``:
    Texture loading options.

Returns:
    List of new texture objects.)doc";

static const char *__doc_sgl_TextureLoader_m_blitter = R"doc()doc";

static const char *__doc_sgl_TextureLoader_m_device = R"doc()doc";

static const char *__doc_sgl_TextureReductionOp = R"doc()doc";

static const char *__doc_sgl_TextureReductionOp_average = R"doc()doc";

static const char *__doc_sgl_TextureReductionOp_comparison = R"doc()doc";

static const char *__doc_sgl_TextureReductionOp_info = R"doc()doc";

static const char *__doc_sgl_TextureReductionOp_maximum = R"doc()doc";

static const char *__doc_sgl_TextureReductionOp_minimum = R"doc()doc";

static const char *__doc_sgl_TextureType = R"doc()doc";

static const char *__doc_sgl_TextureType_info = R"doc()doc";

static const char *__doc_sgl_TextureType_texture_1d = R"doc()doc";

static const char *__doc_sgl_TextureType_texture_1d_array = R"doc()doc";

static const char *__doc_sgl_TextureType_texture_2d = R"doc()doc";

static const char *__doc_sgl_TextureType_texture_2d_array = R"doc()doc";

static const char *__doc_sgl_TextureType_texture_2d_ms = R"doc()doc";

static const char *__doc_sgl_TextureType_texture_2d_ms_array = R"doc()doc";

static const char *__doc_sgl_TextureType_texture_3d = R"doc()doc";

static const char *__doc_sgl_TextureType_texture_cube = R"doc()doc";

static const char *__doc_sgl_TextureType_texture_cube_array = R"doc()doc";

static const char *__doc_sgl_TextureUsage = R"doc()doc";

static const char *__doc_sgl_TextureUsage_copy_destination = R"doc()doc";

static const char *__doc_sgl_TextureUsage_copy_source = R"doc()doc";

static const char *__doc_sgl_TextureUsage_depth_stencil = R"doc()doc";

static const char *__doc_sgl_TextureUsage_info = R"doc()doc";

static const char *__doc_sgl_TextureUsage_none = R"doc()doc";

static const char *__doc_sgl_TextureUsage_present = R"doc()doc";

static const char *__doc_sgl_TextureUsage_render_target = R"doc()doc";

static const char *__doc_sgl_TextureUsage_resolve_destination = R"doc()doc";

static const char *__doc_sgl_TextureUsage_resolve_source = R"doc()doc";

static const char *__doc_sgl_TextureUsage_shader_resource = R"doc()doc";

static const char *__doc_sgl_TextureUsage_shared = R"doc()doc";

static const char *__doc_sgl_TextureUsage_typeless = R"doc()doc";

static const char *__doc_sgl_TextureUsage_unordered_access = R"doc()doc";

static const char *__doc_sgl_TextureView = R"doc()doc";

static const char *__doc_sgl_TextureView_2 = R"doc()doc";

static const char *__doc_sgl_TextureViewDesc = R"doc()doc";

static const char *__doc_sgl_TextureViewDesc_2 = R"doc()doc";

static const char *__doc_sgl_TextureViewDesc_aspect = R"doc()doc";

static const char *__doc_sgl_TextureViewDesc_format = R"doc()doc";

static const char *__doc_sgl_TextureViewDesc_label = R"doc()doc";

static const char *__doc_sgl_TextureViewDesc_subresource_range = R"doc()doc";

static const char *__doc_sgl_TextureView_TextureView = R"doc()doc";

static const char *__doc_sgl_TextureView_aspect = R"doc()doc";

static const char *__doc_sgl_TextureView_class_name = R"doc()doc";

static const char *__doc_sgl_TextureView_desc = R"doc()doc";

static const char *__doc_sgl_TextureView_descriptor_handle_ro = R"doc()doc";

static const char *__doc_sgl_TextureView_descriptor_handle_rw = R"doc()doc";

static const char *__doc_sgl_TextureView_format = R"doc()doc";

static const char *__doc_sgl_TextureView_label = R"doc()doc";

static const char *__doc_sgl_TextureView_m_desc = R"doc()doc";

static const char *__doc_sgl_TextureView_m_rhi_texture_view = R"doc()doc";

static const char *__doc_sgl_TextureView_m_texture = R"doc()doc";

static const char *__doc_sgl_TextureView_native_handle = R"doc(Get the native texture view handle.)doc";

static const char *__doc_sgl_TextureView_rhi_texture_view = R"doc()doc";

static const char *__doc_sgl_TextureView_subresource_range = R"doc()doc";

static const char *__doc_sgl_TextureView_texture = R"doc()doc";

static const char *__doc_sgl_TextureView_to_string = R"doc()doc";

static const char *__doc_sgl_Texture_Texture = R"doc()doc";

static const char *__doc_sgl_Texture_Texture_2 = R"doc()doc";

static const char *__doc_sgl_Texture_array_length = R"doc()doc";

static const char *__doc_sgl_Texture_class_name = R"doc()doc";

static const char *__doc_sgl_Texture_create_view = R"doc()doc";

static const char *__doc_sgl_Texture_depth = R"doc()doc";

static const char *__doc_sgl_Texture_desc = R"doc()doc";

static const char *__doc_sgl_Texture_format = R"doc()doc";

static const char *__doc_sgl_Texture_get_mip_depth = R"doc()doc";

static const char *__doc_sgl_Texture_get_mip_height = R"doc()doc";

static const char *__doc_sgl_Texture_get_mip_size = R"doc()doc";

static const char *__doc_sgl_Texture_get_mip_width = R"doc()doc";

static const char *__doc_sgl_Texture_get_subresource_data =
R"doc(Get subresource data to host memory. \note This will wait until the
data is copied back to host memory.

Parameter ``layer``:
    Layer index.

Parameter ``mip``:
    Mip level.

Returns:
    Subresource data.)doc";

static const char *__doc_sgl_Texture_get_subresource_layout =
R"doc(Get layout of a texture subresource. By default, the row alignment
used is that required by the target for direct buffer upload/download.
Pass in 1 for a completely packed layout.)doc";

static const char *__doc_sgl_Texture_height = R"doc()doc";

static const char *__doc_sgl_Texture_layer_count = R"doc()doc";

static const char *__doc_sgl_Texture_m_desc = R"doc()doc";

static const char *__doc_sgl_Texture_m_rhi_texture = R"doc()doc";

static const char *__doc_sgl_Texture_memory_usage = R"doc()doc";

static const char *__doc_sgl_Texture_mip_count = R"doc()doc";

static const char *__doc_sgl_Texture_rhi_resource = R"doc()doc";

static const char *__doc_sgl_Texture_rhi_texture = R"doc()doc";

static const char *__doc_sgl_Texture_set_subresource_data =
R"doc(Set subresource data from host memory.

Parameter ``layer``:
    Layer index.

Parameter ``mip``:
    Mip level.

Parameter ``subresource_data``:
    Subresource data.)doc";

static const char *__doc_sgl_Texture_shared_handle =
R"doc(Get the shared resource handle. Note: Texture must be created with the
``TextureUsage::shared`` usage flag.)doc";

static const char *__doc_sgl_Texture_subresource_count = R"doc()doc";

static const char *__doc_sgl_Texture_to_bitmap = R"doc()doc";

static const char *__doc_sgl_Texture_to_string = R"doc()doc";

static const char *__doc_sgl_Texture_type = R"doc()doc";

static const char *__doc_sgl_Texture_width = R"doc()doc";

static const char *__doc_sgl_Timer = R"doc()doc";

static const char *__doc_sgl_Timer_2 = R"doc(High resolution CPU timer.)doc";

static const char *__doc_sgl_Timer_Timer = R"doc()doc";

static const char *__doc_sgl_Timer_delta_ms = R"doc(Compute elapsed milliseconds between two time points.)doc";

static const char *__doc_sgl_Timer_delta_ns = R"doc(Compute elapsed nanoseconds between two time points.)doc";

static const char *__doc_sgl_Timer_delta_s = R"doc(Compute elapsed seconds between two time points.)doc";

static const char *__doc_sgl_Timer_delta_us = R"doc(Compute elapsed microseconds between two time points.)doc";

static const char *__doc_sgl_Timer_elapsed_ms = R"doc(Elapsed milliseconds since last reset.)doc";

static const char *__doc_sgl_Timer_elapsed_ns = R"doc(Elapsed nanoseconds since last reset.)doc";

static const char *__doc_sgl_Timer_elapsed_s = R"doc(Elapsed seconds since last reset.)doc";

static const char *__doc_sgl_Timer_elapsed_us = R"doc(Elapsed microseconds since last reset.)doc";

static const char *__doc_sgl_Timer_m_start = R"doc()doc";

static const char *__doc_sgl_Timer_now = R"doc(Current time point in nanoseconds since epoch.)doc";

static const char *__doc_sgl_Timer_reset = R"doc(Reset the timer.)doc";

static const char *__doc_sgl_TypeConformance =
R"doc(Type conformance entry. Type conformances are used to narrow the set
of types supported by a slang interface. They can be specified on an
entry point to omit generating code for types that do not conform.)doc";

static const char *__doc_sgl_TypeConformance_TypeConformance = R"doc()doc";

static const char *__doc_sgl_TypeConformance_TypeConformance_2 = R"doc()doc";

static const char *__doc_sgl_TypeConformance_id = R"doc(Unique id per type for an interface (optional).)doc";

static const char *__doc_sgl_TypeConformance_interface_name = R"doc(Name of the interface.)doc";

static const char *__doc_sgl_TypeConformance_to_string = R"doc()doc";

static const char *__doc_sgl_TypeConformance_type_name = R"doc(Name of the concrete type.)doc";

static const char *__doc_sgl_TypeLayoutReflection = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_2 = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflectionFieldList = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflectionFieldList_2 = R"doc(TypeLayoutReflection lazy field list evaluation.)doc";

static const char *__doc_sgl_TypeLayoutReflectionFieldList_TypeLayoutReflectionFieldList = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflectionFieldList_evaluate = R"doc(Get a specific child.)doc";

static const char *__doc_sgl_TypeLayoutReflectionFieldList_size = R"doc(Number of entries in list.)doc";

static const char *__doc_sgl_TypeLayoutReflection_TypeLayoutReflection = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_alignment = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_container_var_layout = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_element_count = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_element_stride = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_element_type_layout = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_element_var_layout = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_field_count = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_fields = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_find_field_by_name = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_find_field_index_by_name = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_from_slang = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_get_field_binding_range_offset = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_get_field_by_index = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_get_slang_type_layout = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_is_array = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_kind = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_name = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_parameter_category = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_size = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_stride = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_to_string = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_type = R"doc()doc";

static const char *__doc_sgl_TypeLayoutReflection_unwrap_array = R"doc()doc";

static const char *__doc_sgl_TypeReflection = R"doc()doc";

static const char *__doc_sgl_TypeReflection_2 = R"doc()doc";

static const char *__doc_sgl_TypeReflectionFieldList = R"doc()doc";

static const char *__doc_sgl_TypeReflectionFieldList_2 = R"doc(TypeReflection lazy field list evaluation.)doc";

static const char *__doc_sgl_TypeReflectionFieldList_TypeReflectionFieldList = R"doc()doc";

static const char *__doc_sgl_TypeReflectionFieldList_evaluate = R"doc(Get a specific child.)doc";

static const char *__doc_sgl_TypeReflectionFieldList_size = R"doc(Number of entries in list.)doc";

static const char *__doc_sgl_TypeReflection_Kind = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_array = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_constant_buffer = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_feedback = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_generic_type_parameter = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_info = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_interface = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_matrix = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_none = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_output_stream = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_parameter_block = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_pointer = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_resource = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_sampler_state = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_scalar = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_shader_storage_buffer = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_specialized = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_struct = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_texture_buffer = R"doc()doc";

static const char *__doc_sgl_TypeReflection_Kind_vector = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_callable_payload = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_constant_buffer = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_descriptor_table_slot = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_existential_object_param = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_existential_type_param = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_generic = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_hit_attributes = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_info = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_mixed = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_none = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_push_constant_buffer = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_ray_payload = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_register_space = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_sampler_state = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_shader_record = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_shader_resource = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_specialization_constant = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_uniform = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_unordered_access = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_varying_input = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ParameterCategory_varying_output = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceAccess = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceAccess_access_append = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceAccess_access_consume = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceAccess_access_write = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceAccess_info = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceAccess_none = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceAccess_raster_ordered = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceAccess_read = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceAccess_read_write = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_acceleration_structure = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_base_shape_mask = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_byte_address_buffer = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_ext_shape_mask = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_info = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_none = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_structured_buffer = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_texture_1d = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_texture_1d_array = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_texture_2d = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_texture_2d_array = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_texture_2d_multisample = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_texture_2d_multisample_array = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_texture_3d = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_texture_array_flag = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_texture_buffer = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_texture_cube = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_texture_cube_array = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_texture_feedback_flag = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_texture_multisample_flag = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ResourceShape_unknown = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_COUNT = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_bool = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_float16 = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_float32 = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_float64 = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_info = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_int16 = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_int32 = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_int64 = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_int8 = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_intptr = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_none = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_uint16 = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_uint32 = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_uint64 = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_uint8 = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_uintptr = R"doc()doc";

static const char *__doc_sgl_TypeReflection_ScalarType_void = R"doc()doc";

static const char *__doc_sgl_TypeReflection_TypeReflection = R"doc()doc";

static const char *__doc_sgl_TypeReflection_col_count = R"doc()doc";

static const char *__doc_sgl_TypeReflection_element_count = R"doc()doc";

static const char *__doc_sgl_TypeReflection_element_type = R"doc()doc";

static const char *__doc_sgl_TypeReflection_field_count = R"doc()doc";

static const char *__doc_sgl_TypeReflection_fields = R"doc()doc";

static const char *__doc_sgl_TypeReflection_full_name = R"doc()doc";

static const char *__doc_sgl_TypeReflection_get_field_by_index = R"doc()doc";

static const char *__doc_sgl_TypeReflection_is_array = R"doc()doc";

static const char *__doc_sgl_TypeReflection_is_struct = R"doc()doc";

static const char *__doc_sgl_TypeReflection_kind = R"doc()doc";

static const char *__doc_sgl_TypeReflection_name = R"doc()doc";

static const char *__doc_sgl_TypeReflection_resource_access = R"doc()doc";

static const char *__doc_sgl_TypeReflection_resource_result_type = R"doc()doc";

static const char *__doc_sgl_TypeReflection_resource_shape = R"doc()doc";

static const char *__doc_sgl_TypeReflection_row_count = R"doc()doc";

static const char *__doc_sgl_TypeReflection_scalar_type = R"doc()doc";

static const char *__doc_sgl_TypeReflection_to_string = R"doc()doc";

static const char *__doc_sgl_TypeReflection_total_element_count = R"doc()doc";

static const char *__doc_sgl_TypeReflection_unwrap_array = R"doc()doc";

static const char *__doc_sgl_UnownedSlangBlob =
R"doc(Implementation of slang's ISlangBlob interface to access an unowned
blob of data.)doc";

static const char *__doc_sgl_UnownedSlangBlob_UnownedSlangBlob = R"doc()doc";

static const char *__doc_sgl_UnownedSlangBlob_addRef = R"doc()doc";

static const char *__doc_sgl_UnownedSlangBlob_getBufferPointer = R"doc()doc";

static const char *__doc_sgl_UnownedSlangBlob_getBufferSize = R"doc()doc";

static const char *__doc_sgl_UnownedSlangBlob_m_data = R"doc()doc";

static const char *__doc_sgl_UnownedSlangBlob_m_size = R"doc()doc";

static const char *__doc_sgl_UnownedSlangBlob_queryInterface = R"doc()doc";

static const char *__doc_sgl_UnownedSlangBlob_release = R"doc()doc";

static const char *__doc_sgl_VariableLayoutReflection = R"doc()doc";

static const char *__doc_sgl_VariableLayoutReflection_2 = R"doc()doc";

static const char *__doc_sgl_VariableLayoutReflection_VariableLayoutReflection = R"doc()doc";

static const char *__doc_sgl_VariableLayoutReflection_name = R"doc()doc";

static const char *__doc_sgl_VariableLayoutReflection_offset = R"doc()doc";

static const char *__doc_sgl_VariableLayoutReflection_to_string = R"doc()doc";

static const char *__doc_sgl_VariableLayoutReflection_type_layout = R"doc()doc";

static const char *__doc_sgl_VariableLayoutReflection_variable = R"doc()doc";

static const char *__doc_sgl_VariableReflection = R"doc()doc";

static const char *__doc_sgl_VariableReflection_2 = R"doc()doc";

static const char *__doc_sgl_VariableReflection_VariableReflection = R"doc()doc";

static const char *__doc_sgl_VariableReflection_from_slang = R"doc()doc";

static const char *__doc_sgl_VariableReflection_has_modifier = R"doc(Check if variable has a given modifier (e.g. 'inout').)doc";

static const char *__doc_sgl_VariableReflection_name = R"doc(Variable name.)doc";

static const char *__doc_sgl_VariableReflection_type = R"doc(Variable type reflection.)doc";

static const char *__doc_sgl_VertexStreamDesc = R"doc()doc";

static const char *__doc_sgl_VertexStreamDesc_instance_data_step_rate = R"doc(How many instances to draw per chunk of data.)doc";

static const char *__doc_sgl_VertexStreamDesc_slot_class = R"doc(Whether the stream contains per-vertex or per-instance data.)doc";

static const char *__doc_sgl_VertexStreamDesc_stride = R"doc(The stride in bytes for this vertex stream.)doc";

static const char *__doc_sgl_Viewport = R"doc()doc";

static const char *__doc_sgl_Viewport_from_size = R"doc()doc";

static const char *__doc_sgl_Viewport_height = R"doc()doc";

static const char *__doc_sgl_Viewport_max_depth = R"doc()doc";

static const char *__doc_sgl_Viewport_min_depth = R"doc()doc";

static const char *__doc_sgl_Viewport_to_string = R"doc()doc";

static const char *__doc_sgl_Viewport_width = R"doc()doc";

static const char *__doc_sgl_Viewport_x = R"doc()doc";

static const char *__doc_sgl_Viewport_y = R"doc()doc";

static const char *__doc_sgl_Window = R"doc()doc";

static const char *__doc_sgl_Window_2 =
R"doc(Window class.

Platform independent class for managing a window and handle input
events.)doc";

static const char *__doc_sgl_Window_3 = R"doc()doc";

static const char *__doc_sgl_WindowDesc = R"doc(Window description.)doc";

static const char *__doc_sgl_WindowDesc_height = R"doc(Height of the window in pixels.)doc";

static const char *__doc_sgl_WindowDesc_mode = R"doc(Window mode.)doc";

static const char *__doc_sgl_WindowDesc_resizable = R"doc(Whether the window is resizable.)doc";

static const char *__doc_sgl_WindowDesc_title = R"doc(Title of the window.)doc";

static const char *__doc_sgl_WindowDesc_width = R"doc(Width of the window in pixels.)doc";

static const char *__doc_sgl_WindowHandle = R"doc(Native window handle.)doc";

static const char *__doc_sgl_WindowHandle_hwnd = R"doc()doc";

static const char *__doc_sgl_WindowMode = R"doc(Window modes.)doc";

static const char *__doc_sgl_WindowMode_fullscreen = R"doc()doc";

static const char *__doc_sgl_WindowMode_minimized = R"doc()doc";

static const char *__doc_sgl_WindowMode_normal = R"doc()doc";

static const char *__doc_sgl_Window_Window =
R"doc(Constructor.

Parameter ``width``:
    Width of the window in pixels.

Parameter ``height``:
    Height of the window in pixels.

Parameter ``title``:
    Title of the window.

Parameter ``mode``:
    Window mode.

Parameter ``resizable``:
    Whether the window is resizable.)doc";

static const char *__doc_sgl_Window_class_name = R"doc()doc";

static const char *__doc_sgl_Window_close = R"doc(Close the window.)doc";

static const char *__doc_sgl_Window_create = R"doc()doc";

static const char *__doc_sgl_Window_cursor_mode = R"doc(The mouse cursor mode.)doc";

static const char *__doc_sgl_Window_get_clipboard = R"doc(Get the clipboard content.)doc";

static const char *__doc_sgl_Window_handle_drop_files = R"doc()doc";

static const char *__doc_sgl_Window_handle_gamepad_event = R"doc()doc";

static const char *__doc_sgl_Window_handle_keyboard_event = R"doc()doc";

static const char *__doc_sgl_Window_handle_mouse_event = R"doc()doc";

static const char *__doc_sgl_Window_handle_window_size = R"doc()doc";

static const char *__doc_sgl_Window_height = R"doc(The height of the window in pixels.)doc";

static const char *__doc_sgl_Window_m_cursor_mode = R"doc()doc";

static const char *__doc_sgl_Window_m_gamepad_id = R"doc()doc";

static const char *__doc_sgl_Window_m_gamepad_prev_state = R"doc()doc";

static const char *__doc_sgl_Window_m_height = R"doc()doc";

static const char *__doc_sgl_Window_m_mods = R"doc()doc";

static const char *__doc_sgl_Window_m_mouse_pos = R"doc()doc";

static const char *__doc_sgl_Window_m_on_drop_files = R"doc()doc";

static const char *__doc_sgl_Window_m_on_gamepad_event = R"doc()doc";

static const char *__doc_sgl_Window_m_on_gamepad_state = R"doc()doc";

static const char *__doc_sgl_Window_m_on_keyboard_event = R"doc()doc";

static const char *__doc_sgl_Window_m_on_mouse_event = R"doc()doc";

static const char *__doc_sgl_Window_m_on_resize = R"doc()doc";

static const char *__doc_sgl_Window_m_should_close = R"doc()doc";

static const char *__doc_sgl_Window_m_title = R"doc()doc";

static const char *__doc_sgl_Window_m_width = R"doc()doc";

static const char *__doc_sgl_Window_m_window = R"doc()doc";

static const char *__doc_sgl_Window_on_drop_files = R"doc(Event handler to be called when files are dropped onto the window.)doc";

static const char *__doc_sgl_Window_on_gamepad_event = R"doc(Event handler to be called when a gamepad event occurs.)doc";

static const char *__doc_sgl_Window_on_gamepad_state = R"doc(Event handler to be called when the gamepad state changes.)doc";

static const char *__doc_sgl_Window_on_keyboard_event = R"doc(Event handler to be called when a keyboard event occurs.)doc";

static const char *__doc_sgl_Window_on_mouse_event = R"doc(Event handler to be called when a mouse event occurs.)doc";

static const char *__doc_sgl_Window_on_resize = R"doc(Event handler to be called when the window is resized.)doc";

static const char *__doc_sgl_Window_poll_gamepad_input = R"doc()doc";

static const char *__doc_sgl_Window_process_events = R"doc(Process any pending events.)doc";

static const char *__doc_sgl_Window_resize =
R"doc(Resize the window.

Parameter ``width``:
    The new width of the window in pixels.

Parameter ``height``:
    The new height of the window in pixels.)doc";

static const char *__doc_sgl_Window_set_clipboard = R"doc(Set the clipboard content.)doc";

static const char *__doc_sgl_Window_set_cursor_mode = R"doc()doc";

static const char *__doc_sgl_Window_set_icon = R"doc()doc";

static const char *__doc_sgl_Window_set_on_drop_files = R"doc()doc";

static const char *__doc_sgl_Window_set_on_gamepad_event = R"doc()doc";

static const char *__doc_sgl_Window_set_on_gamepad_state = R"doc()doc";

static const char *__doc_sgl_Window_set_on_keyboard_event = R"doc()doc";

static const char *__doc_sgl_Window_set_on_mouse_event = R"doc()doc";

static const char *__doc_sgl_Window_set_on_resize = R"doc()doc";

static const char *__doc_sgl_Window_set_title = R"doc()doc";

static const char *__doc_sgl_Window_should_close = R"doc(True if the window should be closed.)doc";

static const char *__doc_sgl_Window_title = R"doc(The title of the window.)doc";

static const char *__doc_sgl_Window_to_string = R"doc()doc";

static const char *__doc_sgl_Window_width = R"doc(The width of the window in pixels.)doc";

static const char *__doc_sgl_Window_window_handle = R"doc(The native window handle.)doc";

static const char *__doc_sgl_align_to = R"doc(Align an integer value to the given alignment.)doc";

static const char *__doc_sgl_breakable_ref = R"doc()doc";

static const char *__doc_sgl_breakable_ref_2 =
R"doc(Breakable reference counting helper for avoding reference cycles.

This helper represents a strong reference (ref<T>) that can be broken.
This is accomplished by storing both a strong reference and a raw
pointer. When the strong reference is broken, we access the referenced
object through the raw pointer.

This helper can be used in scenarios where some object holds nested
objects that themselves hold a reference to the parent object. In such
cases, the nested objects should hold a breakable reference to the
parent object. When the nested objects are created, we can immediately
break the strong reference to the parent object. This allows the
parent object to be destroyed when all of the external references to
it are released.

This helper can be used in place of a *ref*, but it cannot be
reassigned.)doc";

static const char *__doc_sgl_breakable_ref_break_strong_reference = R"doc()doc";

static const char *__doc_sgl_breakable_ref_breakable_ref = R"doc()doc";

static const char *__doc_sgl_breakable_ref_breakable_ref_2 = R"doc()doc";

static const char *__doc_sgl_breakable_ref_breakable_ref_3 = R"doc()doc";

static const char *__doc_sgl_breakable_ref_get = R"doc()doc";

static const char *__doc_sgl_breakable_ref_m_strong_ref = R"doc()doc";

static const char *__doc_sgl_breakable_ref_m_weak_ref = R"doc()doc";

static const char *__doc_sgl_breakable_ref_operator_T0 = R"doc()doc";

static const char *__doc_sgl_breakable_ref_operator_assign = R"doc()doc";

static const char *__doc_sgl_breakable_ref_operator_assign_2 = R"doc()doc";

static const char *__doc_sgl_breakable_ref_operator_bool = R"doc()doc";

static const char *__doc_sgl_breakable_ref_operator_mul = R"doc()doc";

static const char *__doc_sgl_breakable_ref_operator_ref = R"doc()doc";

static const char *__doc_sgl_breakable_ref_operator_sub = R"doc()doc";

static const char *__doc_sgl_clamp = R"doc(Clamp x to [lo..hi] range.)doc";

static const char *__doc_sgl_comparator = R"doc()doc";

static const char *__doc_sgl_comparator_operator_call = R"doc()doc";

static const char *__doc_sgl_cuda_ContextScope = R"doc()doc";

static const char *__doc_sgl_cuda_ContextScope_ContextScope = R"doc()doc";

static const char *__doc_sgl_cuda_ContextScope_ContextScope_2 = R"doc()doc";

static const char *__doc_sgl_cuda_Device = R"doc(Wraps a CUDA device, context and stream.)doc";

static const char *__doc_sgl_cuda_Device_2 = R"doc()doc";

static const char *__doc_sgl_cuda_Device_Device =
R"doc(Constructor. Creates a CUDA device on the same adapter as the sgl
device.)doc";

static const char *__doc_sgl_cuda_Device_class_name = R"doc()doc";

static const char *__doc_sgl_cuda_Device_context = R"doc()doc";

static const char *__doc_sgl_cuda_Device_device = R"doc()doc";

static const char *__doc_sgl_cuda_Device_m_context = R"doc()doc";

static const char *__doc_sgl_cuda_Device_m_device = R"doc()doc";

static const char *__doc_sgl_cuda_Device_m_stream = R"doc()doc";

static const char *__doc_sgl_cuda_Device_stream = R"doc()doc";

static const char *__doc_sgl_cuda_ExternalMemory = R"doc(Wraps an external memory resource.)doc";

static const char *__doc_sgl_cuda_ExternalMemory_2 = R"doc()doc";

static const char *__doc_sgl_cuda_ExternalMemory_ExternalMemory = R"doc()doc";

static const char *__doc_sgl_cuda_ExternalMemory_class_name = R"doc()doc";

static const char *__doc_sgl_cuda_ExternalMemory_m_external_memory = R"doc()doc";

static const char *__doc_sgl_cuda_ExternalMemory_m_mapped_data = R"doc()doc";

static const char *__doc_sgl_cuda_ExternalMemory_m_resource = R"doc(Non-owning pointer to the resource.)doc";

static const char *__doc_sgl_cuda_ExternalMemory_m_size = R"doc()doc";

static const char *__doc_sgl_cuda_ExternalMemory_mapped_data = R"doc()doc";

static const char *__doc_sgl_cuda_ExternalMemory_size = R"doc()doc";

static const char *__doc_sgl_cuda_ExternalSemaphore = R"doc(Wraps an external semaphore.)doc";

static const char *__doc_sgl_cuda_ExternalSemaphore_2 = R"doc()doc";

static const char *__doc_sgl_cuda_ExternalSemaphore_ExternalSemaphore = R"doc()doc";

static const char *__doc_sgl_cuda_ExternalSemaphore_class_name = R"doc()doc";

static const char *__doc_sgl_cuda_ExternalSemaphore_m_external_semaphore = R"doc()doc";

static const char *__doc_sgl_cuda_ExternalSemaphore_m_fence = R"doc(Non-owning pointer to the fence.)doc";

static const char *__doc_sgl_cuda_ExternalSemaphore_signal = R"doc()doc";

static const char *__doc_sgl_cuda_ExternalSemaphore_wait = R"doc()doc";

static const char *__doc_sgl_cuda_InteropBuffer = R"doc()doc";

static const char *__doc_sgl_cuda_InteropBuffer_2 = R"doc()doc";

static const char *__doc_sgl_cuda_InteropBuffer_InteropBuffer = R"doc()doc";

static const char *__doc_sgl_cuda_InteropBuffer_buffer = R"doc()doc";

static const char *__doc_sgl_cuda_InteropBuffer_copy_from_cuda = R"doc()doc";

static const char *__doc_sgl_cuda_InteropBuffer_copy_to_cuda = R"doc()doc";

static const char *__doc_sgl_cuda_InteropBuffer_is_uav = R"doc()doc";

static const char *__doc_sgl_cuda_InteropBuffer_m_buffer = R"doc()doc";

static const char *__doc_sgl_cuda_InteropBuffer_m_device = R"doc()doc";

static const char *__doc_sgl_cuda_InteropBuffer_m_external_memory = R"doc()doc";

static const char *__doc_sgl_cuda_InteropBuffer_m_is_uav = R"doc()doc";

static const char *__doc_sgl_cuda_InteropBuffer_m_tensor_view = R"doc()doc";

static const char *__doc_sgl_cuda_TensorView = R"doc()doc";

static const char *__doc_sgl_cuda_TensorView_2 = R"doc()doc";

static const char *__doc_sgl_cuda_TensorView_data = R"doc()doc";

static const char *__doc_sgl_cuda_TensorView_size = R"doc()doc";

static const char *__doc_sgl_cuda_TensorView_stride = R"doc()doc";

static const char *__doc_sgl_cuda_destroy_external_memory = R"doc()doc";

static const char *__doc_sgl_cuda_destroy_external_semaphore = R"doc()doc";

static const char *__doc_sgl_cuda_external_memory_get_mapped_buffer = R"doc()doc";

static const char *__doc_sgl_cuda_free_device = R"doc()doc";

static const char *__doc_sgl_cuda_import_external_memory = R"doc()doc";

static const char *__doc_sgl_cuda_import_external_semaphore = R"doc()doc";

static const char *__doc_sgl_cuda_malloc_device = R"doc()doc";

static const char *__doc_sgl_cuda_memcpy_device_to_device = R"doc()doc";

static const char *__doc_sgl_cuda_memcpy_device_to_host = R"doc()doc";

static const char *__doc_sgl_cuda_memcpy_host_to_device = R"doc()doc";

static const char *__doc_sgl_cuda_memset_device = R"doc()doc";

static const char *__doc_sgl_cuda_signal_external_semaphore = R"doc()doc";

static const char *__doc_sgl_cuda_wait_external_semaphore = R"doc()doc";

static const char *__doc_sgl_cursor_utils_check_array = R"doc()doc";

static const char *__doc_sgl_cursor_utils_check_matrix = R"doc()doc";

static const char *__doc_sgl_cursor_utils_check_scalar = R"doc()doc";

static const char *__doc_sgl_cursor_utils_check_vector = R"doc()doc";

static const char *__doc_sgl_cursor_utils_get_scalar_type_size = R"doc()doc";

static const char *__doc_sgl_cursor_utils_unwrap_array = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_2 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_3 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_4 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_5 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_6 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_7 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_8 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_9 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_10 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_11 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_12 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_13 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_14 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_15 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_16 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_17 = R"doc()doc";

static const char *__doc_sgl_detail_HostTypeToFormat_18 = R"doc()doc";

static const char *__doc_sgl_detail_build_slang_error_message = R"doc()doc";

static const char *__doc_sgl_detail_build_slang_rhi_error_message = R"doc()doc";

static const char *__doc_sgl_detail_format_enum =
R"doc(Format an enum value. First, we check for a single value and return
"name::value" if it succeeds. Otherwise, we check for flags and return
"name::(value1 | value2 | ...)". Any bits that are not found in the
enum information are formatted as hex.)doc";

static const char *__doc_sgl_detail_from_slang = R"doc()doc";

static const char *__doc_sgl_detail_from_slang_2 = R"doc()doc";

static const char *__doc_sgl_detail_from_slang_3 = R"doc()doc";

static const char *__doc_sgl_detail_from_slang_4 = R"doc()doc";

static const char *__doc_sgl_detail_from_slang_5 = R"doc()doc";

static const char *__doc_sgl_detail_from_slang_6 = R"doc()doc";

static const char *__doc_sgl_detail_from_slang_7 = R"doc()doc";

static const char *__doc_sgl_detail_from_slang_8 = R"doc()doc";

static const char *__doc_sgl_detail_get_slang_rhi_message_count = R"doc()doc";

static const char *__doc_sgl_detail_invalidate_all_reflection_data = R"doc()doc";

static const char *__doc_sgl_detail_on_slang_wrapper_destroyed = R"doc()doc";

static const char *__doc_sgl_detail_throw_exception = R"doc()doc";

static const char *__doc_sgl_detail_throw_exception_2 = R"doc()doc";

static const char *__doc_sgl_detail_unused = R"doc()doc";

static const char *__doc_sgl_div_round_up = R"doc(Divide a by b and round up to the next integer.)doc";

static const char *__doc_sgl_dynamic_ref_cast = R"doc()doc";

static const char *__doc_sgl_enum_has_value = R"doc(Check if an enum has a value with the given name.)doc";

static const char *__doc_sgl_enum_to_string = R"doc(Convert an enum value to a string.)doc";

static const char *__doc_sgl_find_enum_info_adl = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_2 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_3 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_4 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_5 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_6 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_7 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_8 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_9 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_10 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_11 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_12 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_13 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_14 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_15 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_16 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_17 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_18 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_19 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_20 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_21 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_22 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_23 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_24 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_25 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_26 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_27 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_28 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_29 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_30 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_31 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_32 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_33 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_34 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_35 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_36 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_37 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_38 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_39 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_40 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_41 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_42 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_43 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_44 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_45 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_46 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_47 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_48 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_49 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_50 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_51 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_52 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_53 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_54 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_55 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_56 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_57 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_58 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_59 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_60 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_61 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_62 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_63 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_64 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_65 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_66 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_67 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_68 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_69 = R"doc()doc";

static const char *__doc_sgl_find_enum_info_adl_70 = R"doc()doc";

static const char *__doc_sgl_flags_to_string_list = R"doc(Convert an flags enum value to a list of strings.)doc";

static const char *__doc_sgl_flip_bit = R"doc()doc";

static const char *__doc_sgl_flip_bit_2 = R"doc()doc";

static const char *__doc_sgl_flip_bit_3 = R"doc()doc";

static const char *__doc_sgl_flip_bit_4 = R"doc()doc";

static const char *__doc_sgl_flip_bit_5 = R"doc()doc";

static const char *__doc_sgl_flip_bit_6 = R"doc()doc";

static const char *__doc_sgl_flip_bit_7 = R"doc()doc";

static const char *__doc_sgl_flip_bit_8 = R"doc()doc";

static const char *__doc_sgl_flip_bit_9 = R"doc()doc";

static const char *__doc_sgl_flip_bit_10 = R"doc()doc";

static const char *__doc_sgl_flip_bit_11 = R"doc()doc";

static const char *__doc_sgl_flip_bit_12 = R"doc()doc";

static const char *__doc_sgl_flip_bit_13 = R"doc()doc";

static const char *__doc_sgl_get_cuda_current_context_native_handles =
R"doc(Gets the device and context handles for the current CUDA context. Use
to retrieve an existing context (eg from PyTorch) to pass as the
existing_device_handles from which to create a device in the
DeviceDesc.)doc";

static const char *__doc_sgl_get_dxgi_format = R"doc(Convert from sgl to DXGI format.)doc";

static const char *__doc_sgl_get_format = R"doc(Convert from DXGI to sgl format.)doc";

static const char *__doc_sgl_get_format_2 = R"doc(Convert from Vulkan to sgl format.)doc";

static const char *__doc_sgl_get_format_info = R"doc()doc";

static const char *__doc_sgl_get_shader_model_major_version = R"doc()doc";

static const char *__doc_sgl_get_shader_model_minor_version = R"doc()doc";

static const char *__doc_sgl_get_vulkan_format = R"doc(Convert from sgl to Vulkan format.)doc";

static const char *__doc_sgl_hash = R"doc()doc";

static const char *__doc_sgl_hash_2 = R"doc()doc";

static const char *__doc_sgl_hash_3 = R"doc()doc";

static const char *__doc_sgl_hash_combine = R"doc()doc";

static const char *__doc_sgl_hasher = R"doc()doc";

static const char *__doc_sgl_hasher_operator_call = R"doc()doc";

static const char *__doc_sgl_host_type_to_format = R"doc()doc";

static const char *__doc_sgl_is_boolean = R"doc()doc";

static const char *__doc_sgl_is_power_of_two = R"doc(Returns true if integer value is a power of two.)doc";

static const char *__doc_sgl_is_ref = R"doc()doc";

static const char *__doc_sgl_is_set = R"doc()doc";

static const char *__doc_sgl_is_set_2 = R"doc()doc";

static const char *__doc_sgl_is_set_3 = R"doc()doc";

static const char *__doc_sgl_is_set_4 = R"doc()doc";

static const char *__doc_sgl_is_set_5 = R"doc()doc";

static const char *__doc_sgl_is_set_6 = R"doc()doc";

static const char *__doc_sgl_is_set_7 = R"doc()doc";

static const char *__doc_sgl_is_set_8 = R"doc()doc";

static const char *__doc_sgl_is_set_9 = R"doc()doc";

static const char *__doc_sgl_is_set_10 = R"doc()doc";

static const char *__doc_sgl_is_set_11 = R"doc()doc";

static const char *__doc_sgl_is_set_12 = R"doc()doc";

static const char *__doc_sgl_is_set_13 = R"doc()doc";

static const char *__doc_sgl_layout_from_rhilayout = R"doc()doc";

static const char *__doc_sgl_lerp = R"doc(Linearly interpolate between a and b.)doc";

static const char *__doc_sgl_log_debug = R"doc()doc";

static const char *__doc_sgl_log_debug_2 = R"doc()doc";

static const char *__doc_sgl_log_debug_once = R"doc()doc";

static const char *__doc_sgl_log_debug_once_2 = R"doc()doc";

static const char *__doc_sgl_log_error = R"doc()doc";

static const char *__doc_sgl_log_error_2 = R"doc()doc";

static const char *__doc_sgl_log_error_once = R"doc()doc";

static const char *__doc_sgl_log_error_once_2 = R"doc()doc";

static const char *__doc_sgl_log_fatal = R"doc()doc";

static const char *__doc_sgl_log_fatal_2 = R"doc()doc";

static const char *__doc_sgl_log_fatal_once = R"doc()doc";

static const char *__doc_sgl_log_fatal_once_2 = R"doc()doc";

static const char *__doc_sgl_log_info = R"doc()doc";

static const char *__doc_sgl_log_info_2 = R"doc()doc";

static const char *__doc_sgl_log_info_once = R"doc()doc";

static const char *__doc_sgl_log_info_once_2 = R"doc()doc";

static const char *__doc_sgl_log_warn = R"doc()doc";

static const char *__doc_sgl_log_warn_2 = R"doc()doc";

static const char *__doc_sgl_log_warn_once = R"doc()doc";

static const char *__doc_sgl_log_warn_once_2 = R"doc()doc";

static const char *__doc_sgl_make_ref = R"doc()doc";

static const char *__doc_sgl_math_Handedness = R"doc()doc";

static const char *__doc_sgl_math_Handedness_left_handed = R"doc()doc";

static const char *__doc_sgl_math_Handedness_right_handed = R"doc()doc";

static const char *__doc_sgl_math_ScalarTraits = R"doc()doc";

static const char *__doc_sgl_math_ScalarTraits_2 = R"doc()doc";

static const char *__doc_sgl_math_ScalarTraits_3 = R"doc()doc";

static const char *__doc_sgl_math_ScalarTraits_4 = R"doc()doc";

static const char *__doc_sgl_math_ScalarTraits_5 = R"doc()doc";

static const char *__doc_sgl_math_ScalarTraits_6 = R"doc()doc";

static const char *__doc_sgl_math_ScalarTraits_7 = R"doc()doc";

static const char *__doc_sgl_math_abs = R"doc()doc";

static const char *__doc_sgl_math_abs_2 = R"doc(abs)doc";

static const char *__doc_sgl_math_acos = R"doc()doc";

static const char *__doc_sgl_math_acos_2 = R"doc()doc";

static const char *__doc_sgl_math_acos_3 = R"doc()doc";

static const char *__doc_sgl_math_acos_4 = R"doc(acos)doc";

static const char *__doc_sgl_math_all = R"doc()doc";

static const char *__doc_sgl_math_all_2 = R"doc()doc";

static const char *__doc_sgl_math_all_3 = R"doc()doc";

static const char *__doc_sgl_math_all_4 = R"doc()doc";

static const char *__doc_sgl_math_all_5 = R"doc()doc";

static const char *__doc_sgl_math_any = R"doc()doc";

static const char *__doc_sgl_math_any_2 = R"doc()doc";

static const char *__doc_sgl_math_any_3 = R"doc()doc";

static const char *__doc_sgl_math_any_4 = R"doc()doc";

static const char *__doc_sgl_math_any_5 = R"doc()doc";

static const char *__doc_sgl_math_asfloat = R"doc()doc";

static const char *__doc_sgl_math_asfloat_2 = R"doc()doc";

static const char *__doc_sgl_math_asfloat16 = R"doc()doc";

static const char *__doc_sgl_math_asin = R"doc()doc";

static const char *__doc_sgl_math_asin_2 = R"doc()doc";

static const char *__doc_sgl_math_asin_3 = R"doc()doc";

static const char *__doc_sgl_math_asin_4 = R"doc(asin)doc";

static const char *__doc_sgl_math_asint = R"doc()doc";

static const char *__doc_sgl_math_asuint = R"doc()doc";

static const char *__doc_sgl_math_asuint_2 = R"doc()doc";

static const char *__doc_sgl_math_asuint_3 = R"doc()doc";

static const char *__doc_sgl_math_asuint_4 = R"doc()doc";

static const char *__doc_sgl_math_asuint16 = R"doc()doc";

static const char *__doc_sgl_math_atan = R"doc()doc";

static const char *__doc_sgl_math_atan_2 = R"doc()doc";

static const char *__doc_sgl_math_atan_3 = R"doc()doc";

static const char *__doc_sgl_math_atan_4 = R"doc(atan)doc";

static const char *__doc_sgl_math_atan2 = R"doc()doc";

static const char *__doc_sgl_math_atan2_2 = R"doc()doc";

static const char *__doc_sgl_math_atan2_3 = R"doc()doc";

static const char *__doc_sgl_math_atan2_4 = R"doc(atan2)doc";

static const char *__doc_sgl_math_ceil = R"doc()doc";

static const char *__doc_sgl_math_ceil_2 = R"doc()doc";

static const char *__doc_sgl_math_ceil_3 = R"doc()doc";

static const char *__doc_sgl_math_ceil_4 = R"doc(ceil)doc";

static const char *__doc_sgl_math_clamp = R"doc()doc";

static const char *__doc_sgl_math_clamp_2 = R"doc(clamp)doc";

static const char *__doc_sgl_math_conjugate = R"doc(conjugate)doc";

static const char *__doc_sgl_math_cos = R"doc()doc";

static const char *__doc_sgl_math_cos_2 = R"doc()doc";

static const char *__doc_sgl_math_cos_3 = R"doc()doc";

static const char *__doc_sgl_math_cos_4 = R"doc(cos)doc";

static const char *__doc_sgl_math_cosh = R"doc()doc";

static const char *__doc_sgl_math_cosh_2 = R"doc()doc";

static const char *__doc_sgl_math_cosh_3 = R"doc()doc";

static const char *__doc_sgl_math_cosh_4 = R"doc(cosh)doc";

static const char *__doc_sgl_math_cross = R"doc(cross)doc";

static const char *__doc_sgl_math_cross_2 = R"doc(cross)doc";

static const char *__doc_sgl_math_decompose =
R"doc(Decomposes a model matrix into translation, rotation and scale
components.)doc";

static const char *__doc_sgl_math_degrees = R"doc()doc";

static const char *__doc_sgl_math_degrees_2 = R"doc()doc";

static const char *__doc_sgl_math_degrees_3 = R"doc()doc";

static const char *__doc_sgl_math_degrees_4 = R"doc(degrees)doc";

static const char *__doc_sgl_math_determinant = R"doc(Compute determinant of a 2x2 matrix.)doc";

static const char *__doc_sgl_math_determinant_2 = R"doc(Compute determinant of a 3x3 matrix.)doc";

static const char *__doc_sgl_math_determinant_3 = R"doc(Compute determinant of a 4x4 matrix.)doc";

static const char *__doc_sgl_math_dot = R"doc(dot)doc";

static const char *__doc_sgl_math_dot_2 = R"doc(dot)doc";

static const char *__doc_sgl_math_euler_angles =
R"doc(Extract the euler angles in radians from a quaternion (pitch as x, yaw
as y, roll as z).)doc";

static const char *__doc_sgl_math_exp = R"doc()doc";

static const char *__doc_sgl_math_exp_2 = R"doc()doc";

static const char *__doc_sgl_math_exp_3 = R"doc()doc";

static const char *__doc_sgl_math_exp_4 = R"doc()doc";

static const char *__doc_sgl_math_exp_5 = R"doc(exp)doc";

static const char *__doc_sgl_math_exp2 = R"doc()doc";

static const char *__doc_sgl_math_exp2_2 = R"doc()doc";

static const char *__doc_sgl_math_exp2_3 = R"doc()doc";

static const char *__doc_sgl_math_exp2_4 = R"doc()doc";

static const char *__doc_sgl_math_exp2_5 = R"doc(exp2)doc";

static const char *__doc_sgl_math_extract_euler_angle_xyz = R"doc(Compute the (X * Y * Z) euler angles of a 4x4 matrix.)doc";

static const char *__doc_sgl_math_f16tof32 = R"doc()doc";

static const char *__doc_sgl_math_f16tof32_2 = R"doc()doc";

static const char *__doc_sgl_math_f16tof32_3 = R"doc()doc";

static const char *__doc_sgl_math_f16tof32_4 = R"doc()doc";

static const char *__doc_sgl_math_f32tof16 = R"doc()doc";

static const char *__doc_sgl_math_f32tof16_2 = R"doc()doc";

static const char *__doc_sgl_math_f32tof16_3 = R"doc()doc";

static const char *__doc_sgl_math_f32tof16_4 = R"doc()doc";

static const char *__doc_sgl_math_float16_t = R"doc()doc";

static const char *__doc_sgl_math_float16_t_Tag = R"doc()doc";

static const char *__doc_sgl_math_float16_t_Tag_FromBits = R"doc()doc";

static const char *__doc_sgl_math_float16_t_exponent = R"doc()doc";

static const char *__doc_sgl_math_float16_t_float16_t = R"doc()doc";

static const char *__doc_sgl_math_float16_t_float16_t_2 = R"doc()doc";

static const char *__doc_sgl_math_float16_t_float16_t_3 = R"doc()doc";

static const char *__doc_sgl_math_float16_t_float16_t_4 = R"doc()doc";

static const char *__doc_sgl_math_float16_t_float16_t_5 = R"doc()doc";

static const char *__doc_sgl_math_float16_t_fromBits = R"doc()doc";

static const char *__doc_sgl_math_float16_t_is_denormalized = R"doc()doc";

static const char *__doc_sgl_math_float16_t_is_normalized = R"doc()doc";

static const char *__doc_sgl_math_float16_t_isfinite = R"doc()doc";

static const char *__doc_sgl_math_float16_t_isinf = R"doc()doc";

static const char *__doc_sgl_math_float16_t_isnan = R"doc()doc";

static const char *__doc_sgl_math_float16_t_m_bits = R"doc()doc";

static const char *__doc_sgl_math_float16_t_mantissa = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_add = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_add_2 = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_div = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_eq = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_float = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_ge = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_gt = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_iadd = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_idiv = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_imul = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_isub = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_le = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_lt = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_mul = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_ne = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_sub = R"doc()doc";

static const char *__doc_sgl_math_float16_t_operator_sub_2 = R"doc()doc";

static const char *__doc_sgl_math_float16_t_toBits = R"doc()doc";

static const char *__doc_sgl_math_float16_to_float32 = R"doc()doc";

static const char *__doc_sgl_math_float32_to_float16 = R"doc()doc";

static const char *__doc_sgl_math_floor = R"doc()doc";

static const char *__doc_sgl_math_floor_2 = R"doc()doc";

static const char *__doc_sgl_math_floor_3 = R"doc()doc";

static const char *__doc_sgl_math_floor_4 = R"doc(floor)doc";

static const char *__doc_sgl_math_fmod = R"doc()doc";

static const char *__doc_sgl_math_fmod_2 = R"doc()doc";

static const char *__doc_sgl_math_fmod_3 = R"doc()doc";

static const char *__doc_sgl_math_fmod_4 = R"doc(fmod)doc";

static const char *__doc_sgl_math_frac = R"doc()doc";

static const char *__doc_sgl_math_frac_2 = R"doc()doc";

static const char *__doc_sgl_math_frac_3 = R"doc()doc";

static const char *__doc_sgl_math_frac_4 = R"doc(frac)doc";

static const char *__doc_sgl_math_inverse = R"doc(Compute inverse of a 2x2 matrix.)doc";

static const char *__doc_sgl_math_inverse_2 = R"doc(Compute inverse of a 3x3 matrix.)doc";

static const char *__doc_sgl_math_inverse_3 = R"doc(Compute inverse of a 4x4 matrix.)doc";

static const char *__doc_sgl_math_inverse_4 = R"doc(inverse)doc";

static const char *__doc_sgl_math_isfinite = R"doc(isfinite)doc";

static const char *__doc_sgl_math_isfinite_2 = R"doc()doc";

static const char *__doc_sgl_math_isfinite_3 = R"doc()doc";

static const char *__doc_sgl_math_isfinite_4 = R"doc()doc";

static const char *__doc_sgl_math_isfinite_5 = R"doc()doc";

static const char *__doc_sgl_math_isfinite_6 = R"doc(isfinite)doc";

static const char *__doc_sgl_math_isinf = R"doc(isinf)doc";

static const char *__doc_sgl_math_isinf_2 = R"doc()doc";

static const char *__doc_sgl_math_isinf_3 = R"doc()doc";

static const char *__doc_sgl_math_isinf_4 = R"doc()doc";

static const char *__doc_sgl_math_isinf_5 = R"doc()doc";

static const char *__doc_sgl_math_isinf_6 = R"doc(isinf)doc";

static const char *__doc_sgl_math_isnan = R"doc(isnan)doc";

static const char *__doc_sgl_math_isnan_2 = R"doc()doc";

static const char *__doc_sgl_math_isnan_3 = R"doc()doc";

static const char *__doc_sgl_math_isnan_4 = R"doc()doc";

static const char *__doc_sgl_math_isnan_5 = R"doc()doc";

static const char *__doc_sgl_math_isnan_6 = R"doc(isnan)doc";

static const char *__doc_sgl_math_length = R"doc(length)doc";

static const char *__doc_sgl_math_length_2 = R"doc(length)doc";

static const char *__doc_sgl_math_lerp = R"doc(Linear interpolation.)doc";

static const char *__doc_sgl_math_lerp_2 = R"doc()doc";

static const char *__doc_sgl_math_lerp_3 = R"doc()doc";

static const char *__doc_sgl_math_lerp_4 = R"doc()doc";

static const char *__doc_sgl_math_lerp_5 = R"doc(lerp)doc";

static const char *__doc_sgl_math_lerp_6 = R"doc()doc";

static const char *__doc_sgl_math_lex_lt = R"doc()doc";

static const char *__doc_sgl_math_linear_to_srgb = R"doc()doc";

static const char *__doc_sgl_math_linear_to_srgb_2 = R"doc()doc";

static const char *__doc_sgl_math_log = R"doc()doc";

static const char *__doc_sgl_math_log_2 = R"doc()doc";

static const char *__doc_sgl_math_log_3 = R"doc()doc";

static const char *__doc_sgl_math_log_4 = R"doc()doc";

static const char *__doc_sgl_math_log_5 = R"doc(log)doc";

static const char *__doc_sgl_math_log10 = R"doc()doc";

static const char *__doc_sgl_math_log10_2 = R"doc()doc";

static const char *__doc_sgl_math_log10_3 = R"doc()doc";

static const char *__doc_sgl_math_log10_4 = R"doc(log10)doc";

static const char *__doc_sgl_math_log2 = R"doc()doc";

static const char *__doc_sgl_math_log2_2 = R"doc()doc";

static const char *__doc_sgl_math_log2_3 = R"doc()doc";

static const char *__doc_sgl_math_log2_4 = R"doc(log2)doc";

static const char *__doc_sgl_math_matrix =
R"doc(Matrix type with row-major storage.

The semantics are aligned with Slang: - Row major storage - Math
operators are element-wise (e.g. +, -, *, /) - Free standing functions
for matrix operations (e.g. mul(), transpose(), etc.)

Template parameter ``T``:
    Scalar type

Template parameter ``RowCount``:
    Number of rows (1-4)

Template parameter ``ColCount``:
    Number of columns (1-4))doc";

static const char *__doc_sgl_math_matrix_Form = R"doc()doc";

static const char *__doc_sgl_math_matrix_Form_Identity = R"doc()doc";

static const char *__doc_sgl_math_matrix_Form_Undefined = R"doc()doc";

static const char *__doc_sgl_math_matrix_Form_Zeros = R"doc()doc";

static const char *__doc_sgl_math_matrix_data = R"doc()doc";

static const char *__doc_sgl_math_matrix_data_2 = R"doc()doc";

static const char *__doc_sgl_math_matrix_from_coefficients = R"doc(Creates a matrix from coefficients in row-major order.)doc";

static const char *__doc_sgl_math_matrix_from_columns = R"doc(Creates a matrix from column vectors.)doc";

static const char *__doc_sgl_math_matrix_from_columns_2 = R"doc(Creates a matrix from column vectors.)doc";

static const char *__doc_sgl_math_matrix_from_columns_3 = R"doc(Creates a matrix from column vectors.)doc";

static const char *__doc_sgl_math_matrix_from_columns_4 = R"doc(Creates a matrix from column vectors.)doc";

static const char *__doc_sgl_math_matrix_from_diagonal = R"doc(Creates a square matrix from a diagonal vector.)doc";

static const char *__doc_sgl_math_matrix_from_look_at =
R"doc(Build a look-at matrix. If right handed, forward direction is mapped
onto -Z axis. If left handed, forward direction is mapped onto +Z
axis.

Parameter ``eye``:
    Eye position

Parameter ``center``:
    Center position

Parameter ``up``:
    Up vector

Parameter ``handedness``:
    Coordinate system handedness.)doc";

static const char *__doc_sgl_math_matrix_from_quat = R"doc()doc";

static const char *__doc_sgl_math_matrix_from_rotation = R"doc(Creates a rotation matrix from an angle and an axis.)doc";

static const char *__doc_sgl_math_matrix_from_rotation_x = R"doc(Creates a rotation matrix around the X-axis.)doc";

static const char *__doc_sgl_math_matrix_from_rotation_xyz = R"doc(Creates a rotation matrix (X * Y * Z).)doc";

static const char *__doc_sgl_math_matrix_from_rotation_xyz_2 = R"doc()doc";

static const char *__doc_sgl_math_matrix_from_rotation_y = R"doc(Creates a rotation matrix around the Y-axis.)doc";

static const char *__doc_sgl_math_matrix_from_rotation_z = R"doc(Creates a rotation matrix around the Z-axis.)doc";

static const char *__doc_sgl_math_matrix_from_scaling = R"doc(Creates a scaling matrix.)doc";

static const char *__doc_sgl_math_matrix_from_translation = R"doc(Creates a translation matrix.)doc";

static const char *__doc_sgl_math_matrix_get_col = R"doc()doc";

static const char *__doc_sgl_math_matrix_get_row = R"doc()doc";

static const char *__doc_sgl_math_matrix_get_row_2 = R"doc()doc";

static const char *__doc_sgl_math_matrix_identity = R"doc(Identity matrix.)doc";

static const char *__doc_sgl_math_matrix_m_rows = R"doc()doc";

static const char *__doc_sgl_math_matrix_matrix = R"doc()doc";

static const char *__doc_sgl_math_matrix_matrix_2 = R"doc()doc";

static const char *__doc_sgl_math_matrix_matrix_3 = R"doc()doc";

static const char *__doc_sgl_math_matrix_matrix_4 = R"doc()doc";

static const char *__doc_sgl_math_matrix_matrix_5 = R"doc()doc";

static const char *__doc_sgl_math_matrix_matrix_6 =
R"doc(Construct matrix from another matrix with different dimensions. In
HLSL/Slang, destination matrix must be equal or smaller than source
matrix. In sgl, destination matrix can be larger than source matrix
(initialized with identity).)doc";

static const char *__doc_sgl_math_matrix_matrix_7 = R"doc()doc";

static const char *__doc_sgl_math_matrix_operator_array = R"doc()doc";

static const char *__doc_sgl_math_matrix_operator_array_2 = R"doc()doc";

static const char *__doc_sgl_math_matrix_operator_assign = R"doc()doc";

static const char *__doc_sgl_math_matrix_operator_assign_2 = R"doc()doc";

static const char *__doc_sgl_math_matrix_operator_assign_3 = R"doc()doc";

static const char *__doc_sgl_math_matrix_operator_eq = R"doc()doc";

static const char *__doc_sgl_math_matrix_operator_ne = R"doc()doc";

static const char *__doc_sgl_math_matrix_set_col = R"doc()doc";

static const char *__doc_sgl_math_matrix_set_row = R"doc()doc";

static const char *__doc_sgl_math_matrix_zeros = R"doc(Zero matrix.)doc";

static const char *__doc_sgl_math_max = R"doc()doc";

static const char *__doc_sgl_math_max_2 = R"doc(max)doc";

static const char *__doc_sgl_math_min = R"doc()doc";

static const char *__doc_sgl_math_min_2 = R"doc(min)doc";

static const char *__doc_sgl_math_mul = R"doc(Multiply matrix and matrix.)doc";

static const char *__doc_sgl_math_mul_2 = R"doc(Multiply matrix and vector. Vector is treated as a column vector.)doc";

static const char *__doc_sgl_math_mul_3 = R"doc(Multiply vector and matrix. Vector is treated as a row vector.)doc";

static const char *__doc_sgl_math_mul_4 = R"doc(Multiply quaternion with another quaternion.)doc";

static const char *__doc_sgl_math_mul_5 = R"doc(Multiply quaternion and 3 component vector.)doc";

static const char *__doc_sgl_math_none = R"doc()doc";

static const char *__doc_sgl_math_none_2 = R"doc()doc";

static const char *__doc_sgl_math_none_3 = R"doc()doc";

static const char *__doc_sgl_math_none_4 = R"doc()doc";

static const char *__doc_sgl_math_normalize = R"doc(normalize)doc";

static const char *__doc_sgl_math_normalize_2 = R"doc(normalize)doc";

static const char *__doc_sgl_math_operator = R"doc(Binary ^ operator)doc";

static const char *__doc_sgl_math_operator_2 = R"doc(Binary ^ operator)doc";

static const char *__doc_sgl_math_operator_3 = R"doc(Binary ^ operator)doc";

static const char *__doc_sgl_math_operator_add = R"doc(Unary plus operator)doc";

static const char *__doc_sgl_math_operator_add_2 = R"doc(Binary + operator)doc";

static const char *__doc_sgl_math_operator_add_3 = R"doc(Binary + operator)doc";

static const char *__doc_sgl_math_operator_add_4 = R"doc(Binary + operator)doc";

static const char *__doc_sgl_math_operator_add_5 = R"doc(Unary plus operator)doc";

static const char *__doc_sgl_math_operator_add_6 = R"doc(Binary + operator)doc";

static const char *__doc_sgl_math_operator_add_7 = R"doc(Binary + operator)doc";

static const char *__doc_sgl_math_operator_add_8 = R"doc(Binary + operator)doc";

static const char *__doc_sgl_math_operator_band = R"doc(Binary & operator)doc";

static const char *__doc_sgl_math_operator_band_2 = R"doc(Binary & operator)doc";

static const char *__doc_sgl_math_operator_band_3 = R"doc(Binary & operator)doc";

static const char *__doc_sgl_math_operator_bnot = R"doc(Unary not operator)doc";

static const char *__doc_sgl_math_operator_bor = R"doc(Binary | operator)doc";

static const char *__doc_sgl_math_operator_bor_2 = R"doc(Binary | operator)doc";

static const char *__doc_sgl_math_operator_bor_3 = R"doc(Binary | operator)doc";

static const char *__doc_sgl_math_operator_div = R"doc(Binary / operator)doc";

static const char *__doc_sgl_math_operator_div_2 = R"doc(Binary / operator)doc";

static const char *__doc_sgl_math_operator_div_3 = R"doc(Binary / operator)doc";

static const char *__doc_sgl_math_operator_div_4 = R"doc(Binary / operator)doc";

static const char *__doc_sgl_math_operator_eq = R"doc(Binary == operator)doc";

static const char *__doc_sgl_math_operator_eq_2 = R"doc(Binary == operator)doc";

static const char *__doc_sgl_math_operator_eq_3 = R"doc(Binary == operator)doc";

static const char *__doc_sgl_math_operator_eq_4 = R"doc(Binary == operator)doc";

static const char *__doc_sgl_math_operator_ge = R"doc(Binary >= operator)doc";

static const char *__doc_sgl_math_operator_ge_2 = R"doc(Binary >= operator)doc";

static const char *__doc_sgl_math_operator_ge_3 = R"doc(Binary >= operator)doc";

static const char *__doc_sgl_math_operator_gt = R"doc(Binary > operator)doc";

static const char *__doc_sgl_math_operator_gt_2 = R"doc(Binary > operator)doc";

static const char *__doc_sgl_math_operator_gt_3 = R"doc(Binary > operator)doc";

static const char *__doc_sgl_math_operator_h = R"doc(h suffix for "half float" literals.)doc";

static const char *__doc_sgl_math_operator_iadd = R"doc(+= assignment operator)doc";

static const char *__doc_sgl_math_operator_iadd_2 = R"doc(+= assignment operator)doc";

static const char *__doc_sgl_math_operator_iand = R"doc(&= assignment operator)doc";

static const char *__doc_sgl_math_operator_iand_2 = R"doc(&= assignment operator)doc";

static const char *__doc_sgl_math_operator_idiv = R"doc(/= assignment operator)doc";

static const char *__doc_sgl_math_operator_idiv_2 = R"doc(/= assignment operator)doc";

static const char *__doc_sgl_math_operator_ilshift = R"doc(<<= assignment operator)doc";

static const char *__doc_sgl_math_operator_ilshift_2 = R"doc(<<= assignment operator)doc";

static const char *__doc_sgl_math_operator_imod = R"doc(%= assignment operator)doc";

static const char *__doc_sgl_math_operator_imod_2 = R"doc(%= assignment operator)doc";

static const char *__doc_sgl_math_operator_imul = R"doc(*= assignment operator)doc";

static const char *__doc_sgl_math_operator_imul_2 = R"doc(*= assignment operator)doc";

static const char *__doc_sgl_math_operator_ior = R"doc(|= assignment operator)doc";

static const char *__doc_sgl_math_operator_ior_2 = R"doc(|= assignment operator)doc";

static const char *__doc_sgl_math_operator_irshift = R"doc(>>= assignment operator)doc";

static const char *__doc_sgl_math_operator_irshift_2 = R"doc(>>= assignment operator)doc";

static const char *__doc_sgl_math_operator_isub = R"doc(-= assignment operator)doc";

static const char *__doc_sgl_math_operator_isub_2 = R"doc(-= assignment operator)doc";

static const char *__doc_sgl_math_operator_ixor = R"doc(^= assignment operator)doc";

static const char *__doc_sgl_math_operator_ixor_2 = R"doc(^= assignment operator)doc";

static const char *__doc_sgl_math_operator_land = R"doc(Binary && operator)doc";

static const char *__doc_sgl_math_operator_land_2 = R"doc(Binary && operator)doc";

static const char *__doc_sgl_math_operator_land_3 = R"doc(Binary && operator)doc";

static const char *__doc_sgl_math_operator_le = R"doc(Binary <= operator)doc";

static const char *__doc_sgl_math_operator_le_2 = R"doc(Binary <= operator)doc";

static const char *__doc_sgl_math_operator_le_3 = R"doc(Binary <= operator)doc";

static const char *__doc_sgl_math_operator_lnot = R"doc(Unary not operator)doc";

static const char *__doc_sgl_math_operator_lor = R"doc(Binary || operator)doc";

static const char *__doc_sgl_math_operator_lor_2 = R"doc(Binary || operator)doc";

static const char *__doc_sgl_math_operator_lor_3 = R"doc(Binary || operator)doc";

static const char *__doc_sgl_math_operator_lshift = R"doc(Binary << operator)doc";

static const char *__doc_sgl_math_operator_lshift_2 = R"doc(Binary << operator)doc";

static const char *__doc_sgl_math_operator_lshift_3 = R"doc(Binary << operator)doc";

static const char *__doc_sgl_math_operator_lt = R"doc(Binary < operator)doc";

static const char *__doc_sgl_math_operator_lt_2 = R"doc(Binary < operator)doc";

static const char *__doc_sgl_math_operator_lt_3 = R"doc(Binary < operator)doc";

static const char *__doc_sgl_math_operator_mod = R"doc(Binary % operator)doc";

static const char *__doc_sgl_math_operator_mod_2 = R"doc(Binary % operator)doc";

static const char *__doc_sgl_math_operator_mod_3 = R"doc(Binary % operator)doc";

static const char *__doc_sgl_math_operator_mul = R"doc(Binary * operator)doc";

static const char *__doc_sgl_math_operator_mul_2 = R"doc(Binary * operator)doc";

static const char *__doc_sgl_math_operator_mul_3 = R"doc(Binary * operator)doc";

static const char *__doc_sgl_math_operator_mul_4 = R"doc(Binary * operator)doc";

static const char *__doc_sgl_math_operator_mul_5 = R"doc(Binary * operator)doc";

static const char *__doc_sgl_math_operator_mul_6 = R"doc(Binary * operator)doc";

static const char *__doc_sgl_math_operator_ne = R"doc(Binary != operator)doc";

static const char *__doc_sgl_math_operator_ne_2 = R"doc(Binary != operator)doc";

static const char *__doc_sgl_math_operator_ne_3 = R"doc(Binary != operator)doc";

static const char *__doc_sgl_math_operator_ne_4 = R"doc(Binary != operator)doc";

static const char *__doc_sgl_math_operator_rshift = R"doc(Binary >> operator)doc";

static const char *__doc_sgl_math_operator_rshift_2 = R"doc(Binary >> operator)doc";

static const char *__doc_sgl_math_operator_rshift_3 = R"doc(Binary >> operator)doc";

static const char *__doc_sgl_math_operator_sub = R"doc(Unary minus operator)doc";

static const char *__doc_sgl_math_operator_sub_2 = R"doc(Binary - operator)doc";

static const char *__doc_sgl_math_operator_sub_3 = R"doc(Binary - operator)doc";

static const char *__doc_sgl_math_operator_sub_4 = R"doc(Binary - operator)doc";

static const char *__doc_sgl_math_operator_sub_5 = R"doc(Unary minus operator)doc";

static const char *__doc_sgl_math_operator_sub_6 = R"doc(Binary - operator)doc";

static const char *__doc_sgl_math_operator_sub_7 = R"doc(Binary - operator)doc";

static const char *__doc_sgl_math_operator_sub_8 = R"doc(Binary - operator)doc";

static const char *__doc_sgl_math_ortho =
R"doc(Creates a right-handed orthographic projection matrix. Depth is mapped
to [0, 1].)doc";

static const char *__doc_sgl_math_perspective =
R"doc(Creates a right-handed perspective projection matrix. Depth is mapped
to [0, 1].)doc";

static const char *__doc_sgl_math_pitch = R"doc(Returns pitch value of euler angles expressed in radians.)doc";

static const char *__doc_sgl_math_pow = R"doc()doc";

static const char *__doc_sgl_math_pow_2 = R"doc()doc";

static const char *__doc_sgl_math_pow_3 = R"doc()doc";

static const char *__doc_sgl_math_pow_4 = R"doc(pow)doc";

static const char *__doc_sgl_math_quat =
R"doc(Quaternion type.

A quaternion is an expression of the form:

q = w + xi + yj + zk

where w, x, y, and z are real numbers and i, j, and k are the
imaginary units.

The quaternion is normalized if: w^2 + x^2 + y^2 + z^2 = 1

Quaternions are stored as (x, y, z, w) to make them better for interop
with the GPU.)doc";

static const char *__doc_sgl_math_quat_from_angle_axis =
R"doc(Build a quaternion from an angle and a normalized axis.

Parameter ``angle``:
    Angle expressed in radians.

Parameter ``axis``:
    Axis of the quaternion (must be normalized).)doc";

static const char *__doc_sgl_math_quat_from_euler_angles = R"doc(Build a quaternion from euler angles (pitch, yaw, roll), in radians.)doc";

static const char *__doc_sgl_math_quat_from_look_at =
R"doc(Build a look-at quaternion. If right handed, forward direction is
mapped onto -Z axis. If left handed, forward direction is mapped onto
+Z axis.

Parameter ``dir``:
    Forward direction (must to be normalized).

Parameter ``up``:
    Up vector (must be normalized).

Parameter ``handedness``:
    Coordinate system handedness.)doc";

static const char *__doc_sgl_math_quat_from_matrix = R"doc(Construct a quaternion from a 3x3 rotation matrix.)doc";

static const char *__doc_sgl_math_quat_from_rotation_between_vectors =
R"doc(Compute the rotation between two vectors.

Parameter ``from``:
    From vector (must to be normalized).

Parameter ``to``:
    To vector (must to be normalized).)doc";

static const char *__doc_sgl_math_quat_identity = R"doc(Identity quaternion.)doc";

static const char *__doc_sgl_math_quat_operator_array = R"doc()doc";

static const char *__doc_sgl_math_quat_operator_array_2 = R"doc()doc";

static const char *__doc_sgl_math_quat_quat = R"doc()doc";

static const char *__doc_sgl_math_quat_quat_2 = R"doc()doc";

static const char *__doc_sgl_math_quat_quat_3 = R"doc()doc";

static const char *__doc_sgl_math_quat_quat_4 = R"doc()doc";

static const char *__doc_sgl_math_quat_w = R"doc()doc";

static const char *__doc_sgl_math_quat_x = R"doc()doc";

static const char *__doc_sgl_math_quat_y = R"doc()doc";

static const char *__doc_sgl_math_quat_z = R"doc()doc";

static const char *__doc_sgl_math_radians = R"doc()doc";

static const char *__doc_sgl_math_radians_2 = R"doc()doc";

static const char *__doc_sgl_math_radians_3 = R"doc()doc";

static const char *__doc_sgl_math_radians_4 = R"doc(radians)doc";

static const char *__doc_sgl_math_rcp = R"doc()doc";

static const char *__doc_sgl_math_rcp_2 = R"doc()doc";

static const char *__doc_sgl_math_rcp_3 = R"doc()doc";

static const char *__doc_sgl_math_rcp_4 = R"doc(rcp)doc";

static const char *__doc_sgl_math_reflect = R"doc(reflect)doc";

static const char *__doc_sgl_math_roll = R"doc(Returns roll value of euler angles expressed in radians.)doc";

static const char *__doc_sgl_math_rotate = R"doc(Apply a rotation around an axis to a 4x4 matrix.)doc";

static const char *__doc_sgl_math_round = R"doc()doc";

static const char *__doc_sgl_math_round_2 = R"doc()doc";

static const char *__doc_sgl_math_round_3 = R"doc()doc";

static const char *__doc_sgl_math_round_4 = R"doc(round)doc";

static const char *__doc_sgl_math_rsqrt = R"doc()doc";

static const char *__doc_sgl_math_rsqrt_2 = R"doc()doc";

static const char *__doc_sgl_math_rsqrt_3 = R"doc()doc";

static const char *__doc_sgl_math_rsqrt_4 = R"doc(rsqrt)doc";

static const char *__doc_sgl_math_saturate = R"doc()doc";

static const char *__doc_sgl_math_saturate_2 = R"doc()doc";

static const char *__doc_sgl_math_saturate_3 = R"doc()doc";

static const char *__doc_sgl_math_saturate_4 = R"doc(saturate)doc";

static const char *__doc_sgl_math_scale = R"doc(Apply a scale to a 4x4 matrix.)doc";

static const char *__doc_sgl_math_sign = R"doc()doc";

static const char *__doc_sgl_math_sign_2 = R"doc(sign)doc";

static const char *__doc_sgl_math_sin = R"doc()doc";

static const char *__doc_sgl_math_sin_2 = R"doc()doc";

static const char *__doc_sgl_math_sin_3 = R"doc()doc";

static const char *__doc_sgl_math_sin_4 = R"doc(sin)doc";

static const char *__doc_sgl_math_sinh = R"doc()doc";

static const char *__doc_sgl_math_sinh_2 = R"doc()doc";

static const char *__doc_sgl_math_sinh_3 = R"doc()doc";

static const char *__doc_sgl_math_sinh_4 = R"doc(sinh)doc";

static const char *__doc_sgl_math_slerp = R"doc(Spherical linear interpolation.)doc";

static const char *__doc_sgl_math_smoothstep = R"doc()doc";

static const char *__doc_sgl_math_smoothstep_2 = R"doc(smoothstep)doc";

static const char *__doc_sgl_math_sqrt = R"doc()doc";

static const char *__doc_sgl_math_sqrt_2 = R"doc()doc";

static const char *__doc_sgl_math_sqrt_3 = R"doc()doc";

static const char *__doc_sgl_math_sqrt_4 = R"doc(sqrt)doc";

static const char *__doc_sgl_math_srgb_to_linear = R"doc()doc";

static const char *__doc_sgl_math_srgb_to_linear_2 = R"doc()doc";

static const char *__doc_sgl_math_step = R"doc()doc";

static const char *__doc_sgl_math_step_2 = R"doc()doc";

static const char *__doc_sgl_math_step_3 = R"doc()doc";

static const char *__doc_sgl_math_step_4 = R"doc(step)doc";

static const char *__doc_sgl_math_tan = R"doc()doc";

static const char *__doc_sgl_math_tan_2 = R"doc()doc";

static const char *__doc_sgl_math_tan_3 = R"doc()doc";

static const char *__doc_sgl_math_tan_4 = R"doc(tan)doc";

static const char *__doc_sgl_math_tanh = R"doc()doc";

static const char *__doc_sgl_math_tanh_2 = R"doc()doc";

static const char *__doc_sgl_math_tanh_3 = R"doc()doc";

static const char *__doc_sgl_math_tanh_4 = R"doc(tanh)doc";

static const char *__doc_sgl_math_to_string = R"doc()doc";

static const char *__doc_sgl_math_to_string_2 = R"doc()doc";

static const char *__doc_sgl_math_to_string_3 = R"doc()doc";

static const char *__doc_sgl_math_to_string_4 = R"doc()doc";

static const char *__doc_sgl_math_to_string_5 = R"doc()doc";

static const char *__doc_sgl_math_to_string_6 = R"doc()doc";

static const char *__doc_sgl_math_to_string_7 = R"doc()doc";

static const char *__doc_sgl_math_to_string_8 = R"doc()doc";

static const char *__doc_sgl_math_to_string_9 = R"doc()doc";

static const char *__doc_sgl_math_to_string_10 = R"doc(Convert vector to string.)doc";

static const char *__doc_sgl_math_transform_point =
R"doc(Transform a point by a 4x4 matrix. The point is treated as a column
vector with a 1 in the 4th component.)doc";

static const char *__doc_sgl_math_transform_vector = R"doc(Transform a vector by a 3x3 matrix.)doc";

static const char *__doc_sgl_math_transform_vector_2 =
R"doc(Transform a vector by a 4x4 matrix. The vector is treated as a column
vector with a 0 in the 4th component.)doc";

static const char *__doc_sgl_math_transform_vector_3 = R"doc(Transform a vector by a quaternion.)doc";

static const char *__doc_sgl_math_translate = R"doc(Apply a translation to a 4x4 matrix.)doc";

static const char *__doc_sgl_math_transpose = R"doc(Transpose a matrix.)doc";

static const char *__doc_sgl_math_trunc = R"doc()doc";

static const char *__doc_sgl_math_trunc_2 = R"doc()doc";

static const char *__doc_sgl_math_trunc_3 = R"doc()doc";

static const char *__doc_sgl_math_trunc_4 = R"doc(trunc)doc";

static const char *__doc_sgl_math_vector =
R"doc(Vector type.

The semantics are aligned with Slang: - Math operators are element-
wise (e.g. +, -, *, /) - Free standing functions for vector operations
(e.g. dot(), cross(), etc.)

Template parameter ``T``:
    Scalar type

Template parameter ``N``:
    Number of elements (1-4))doc";

static const char *__doc_sgl_math_yaw = R"doc(Returns yaw value of euler angles expressed in radians.)doc";

static const char *__doc_sgl_narrow_cast = R"doc()doc";

static const char *__doc_sgl_object_init_py =
R"doc(Install Python reference counting handlers

The `Object` class is designed so that the dependency on Python is
*optional*: the code compiles in ordinary C++ projects, in which case
the Python reference counting functionality will simply not be used.

Python binding code must invoke `object_init_py` and provide functions
that can be used to increase/decrease the Python reference count of an
instance (i.e., `Py_INCREF` / `Py_DECREF`).)doc";

static const char *__doc_sgl_operator_band = R"doc()doc";

static const char *__doc_sgl_operator_band_2 = R"doc()doc";

static const char *__doc_sgl_operator_band_3 = R"doc()doc";

static const char *__doc_sgl_operator_band_4 = R"doc()doc";

static const char *__doc_sgl_operator_band_5 = R"doc()doc";

static const char *__doc_sgl_operator_band_6 = R"doc()doc";

static const char *__doc_sgl_operator_band_7 = R"doc()doc";

static const char *__doc_sgl_operator_band_8 = R"doc()doc";

static const char *__doc_sgl_operator_band_9 = R"doc()doc";

static const char *__doc_sgl_operator_band_10 = R"doc()doc";

static const char *__doc_sgl_operator_band_11 = R"doc()doc";

static const char *__doc_sgl_operator_band_12 = R"doc()doc";

static const char *__doc_sgl_operator_band_13 = R"doc()doc";

static const char *__doc_sgl_operator_bnot = R"doc()doc";

static const char *__doc_sgl_operator_bnot_2 = R"doc()doc";

static const char *__doc_sgl_operator_bnot_3 = R"doc()doc";

static const char *__doc_sgl_operator_bnot_4 = R"doc()doc";

static const char *__doc_sgl_operator_bnot_5 = R"doc()doc";

static const char *__doc_sgl_operator_bnot_6 = R"doc()doc";

static const char *__doc_sgl_operator_bnot_7 = R"doc()doc";

static const char *__doc_sgl_operator_bnot_8 = R"doc()doc";

static const char *__doc_sgl_operator_bnot_9 = R"doc()doc";

static const char *__doc_sgl_operator_bnot_10 = R"doc()doc";

static const char *__doc_sgl_operator_bnot_11 = R"doc()doc";

static const char *__doc_sgl_operator_bnot_12 = R"doc()doc";

static const char *__doc_sgl_operator_bnot_13 = R"doc()doc";

static const char *__doc_sgl_operator_bor = R"doc()doc";

static const char *__doc_sgl_operator_bor_2 = R"doc()doc";

static const char *__doc_sgl_operator_bor_3 = R"doc()doc";

static const char *__doc_sgl_operator_bor_4 = R"doc()doc";

static const char *__doc_sgl_operator_bor_5 = R"doc()doc";

static const char *__doc_sgl_operator_bor_6 = R"doc()doc";

static const char *__doc_sgl_operator_bor_7 = R"doc()doc";

static const char *__doc_sgl_operator_bor_8 = R"doc()doc";

static const char *__doc_sgl_operator_bor_9 = R"doc()doc";

static const char *__doc_sgl_operator_bor_10 = R"doc()doc";

static const char *__doc_sgl_operator_bor_11 = R"doc()doc";

static const char *__doc_sgl_operator_bor_12 = R"doc()doc";

static const char *__doc_sgl_operator_bor_13 = R"doc()doc";

static const char *__doc_sgl_operator_iand = R"doc()doc";

static const char *__doc_sgl_operator_iand_2 = R"doc()doc";

static const char *__doc_sgl_operator_iand_3 = R"doc()doc";

static const char *__doc_sgl_operator_iand_4 = R"doc()doc";

static const char *__doc_sgl_operator_iand_5 = R"doc()doc";

static const char *__doc_sgl_operator_iand_6 = R"doc()doc";

static const char *__doc_sgl_operator_iand_7 = R"doc()doc";

static const char *__doc_sgl_operator_iand_8 = R"doc()doc";

static const char *__doc_sgl_operator_iand_9 = R"doc()doc";

static const char *__doc_sgl_operator_iand_10 = R"doc()doc";

static const char *__doc_sgl_operator_iand_11 = R"doc()doc";

static const char *__doc_sgl_operator_iand_12 = R"doc()doc";

static const char *__doc_sgl_operator_iand_13 = R"doc()doc";

static const char *__doc_sgl_operator_ior = R"doc()doc";

static const char *__doc_sgl_operator_ior_2 = R"doc()doc";

static const char *__doc_sgl_operator_ior_3 = R"doc()doc";

static const char *__doc_sgl_operator_ior_4 = R"doc()doc";

static const char *__doc_sgl_operator_ior_5 = R"doc()doc";

static const char *__doc_sgl_operator_ior_6 = R"doc()doc";

static const char *__doc_sgl_operator_ior_7 = R"doc()doc";

static const char *__doc_sgl_operator_ior_8 = R"doc()doc";

static const char *__doc_sgl_operator_ior_9 = R"doc()doc";

static const char *__doc_sgl_operator_ior_10 = R"doc()doc";

static const char *__doc_sgl_operator_ior_11 = R"doc()doc";

static const char *__doc_sgl_operator_ior_12 = R"doc()doc";

static const char *__doc_sgl_operator_ior_13 = R"doc()doc";

static const char *__doc_sgl_platform_FileDialogFilter = R"doc()doc";

static const char *__doc_sgl_platform_FileDialogFilter_FileDialogFilter = R"doc()doc";

static const char *__doc_sgl_platform_FileDialogFilter_FileDialogFilter_2 = R"doc()doc";

static const char *__doc_sgl_platform_FileDialogFilter_FileDialogFilter_3 = R"doc()doc";

static const char *__doc_sgl_platform_FileDialogFilter_name = R"doc(Readable name (e.g. "JPEG").)doc";

static const char *__doc_sgl_platform_FileDialogFilter_pattern = R"doc(File extension pattern (e.g. "*.jpg" or "*.jpg,*.jpeg").)doc";

static const char *__doc_sgl_platform_MemoryStats = R"doc()doc";

static const char *__doc_sgl_platform_MemoryStats_peak_rss = R"doc(Peak resident/working set size in bytes.)doc";

static const char *__doc_sgl_platform_MemoryStats_rss = R"doc(Current resident/working set size in bytes.)doc";

static const char *__doc_sgl_platform_ResolvedStackFrame = R"doc()doc";

static const char *__doc_sgl_platform_ResolvedStackFrame_address = R"doc()doc";

static const char *__doc_sgl_platform_ResolvedStackFrame_line = R"doc()doc";

static const char *__doc_sgl_platform_ResolvedStackFrame_module = R"doc()doc";

static const char *__doc_sgl_platform_ResolvedStackFrame_offset = R"doc()doc";

static const char *__doc_sgl_platform_ResolvedStackFrame_source = R"doc()doc";

static const char *__doc_sgl_platform_ResolvedStackFrame_symbol = R"doc()doc";

static const char *__doc_sgl_platform_app_data_directory = R"doc(The application data directory.)doc";

static const char *__doc_sgl_platform_backtrace = R"doc(Generate a backtrace.)doc";

static const char *__doc_sgl_platform_choose_folder_dialog =
R"doc(Show a folder selection dialog.

Returns:
    The selected folder path or nothing if the dialog was cancelled.)doc";

static const char *__doc_sgl_platform_create_junction = R"doc(Create a junction (soft link).)doc";

static const char *__doc_sgl_platform_debug_break = R"doc(Breaks in debugger (int 3 functionality).)doc";

static const char *__doc_sgl_platform_delete_junction = R"doc(Delete a junction (sof link).)doc";

static const char *__doc_sgl_platform_display_scale_factor = R"doc(The pixel scale factor of the primary display.)doc";

static const char *__doc_sgl_platform_executable_directory = R"doc(The current executable directory.)doc";

static const char *__doc_sgl_platform_executable_name = R"doc(The current executable name.)doc";

static const char *__doc_sgl_platform_executable_path = R"doc(The full path to the current executable.)doc";

static const char *__doc_sgl_platform_format_stacktrace = R"doc(Convert resolved stack trace to a human readable string.)doc";

static const char *__doc_sgl_platform_format_stacktrace_2 = R"doc(Convert resolved stack trace to a human readable string.)doc";

static const char *__doc_sgl_platform_get_environment_variable = R"doc(Get the content of a system environment variable.)doc";

static const char *__doc_sgl_platform_get_extension_from_path =
R"doc(Get the file extension from a path in lower-case and without the
leading '.' character.)doc";

static const char *__doc_sgl_platform_get_proc_address = R"doc(Get a function pointer from a library.)doc";

static const char *__doc_sgl_platform_has_extension =
R"doc(Check if a file path has a given file extension. Does a case-
insensitive comparison.)doc";

static const char *__doc_sgl_platform_home_directory = R"doc(The home directory.)doc";

static const char *__doc_sgl_platform_is_debugger_present = R"doc(Check if a debugger session is attached.)doc";

static const char *__doc_sgl_platform_is_python_active = R"doc(Check if the library is loaded from python.)doc";

static const char *__doc_sgl_platform_is_same_path =
R"doc(Compares two paths in their weakly canonical form, returning true if
they match. Operator == on path does a string comparison, ignoring the
fact that windows paths are case insensitive. STL's equivalent
comparator throws when either of the paths does not exist. This
complements the two by allowing comparing non-existent paths, but at
the same time ignoring case on windows.)doc";

static const char *__doc_sgl_platform_load_shared_library = R"doc(Load a shared library.)doc";

static const char *__doc_sgl_platform_memory_stats = R"doc(Get the current memory stats.)doc";

static const char *__doc_sgl_platform_open_file_dialog =
R"doc(Show a file open dialog.

Parameter ``filters``:
    List of file filters.

Returns:
    The selected file path or nothing if the dialog was cancelled.)doc";

static const char *__doc_sgl_platform_page_size = R"doc(The page size of the system.)doc";

static const char *__doc_sgl_platform_print_to_debug_window = R"doc(Print a message into the debug window.)doc";

static const char *__doc_sgl_platform_project_directory =
R"doc(The project source directory. Note that this is only valid during
development.)doc";

static const char *__doc_sgl_platform_release_shared_library = R"doc(Release a shared library.)doc";

static const char *__doc_sgl_platform_resolve_stacktrace = R"doc(Resolve a stack trace with symbol information.)doc";

static const char *__doc_sgl_platform_runtime_directory =
R"doc(The runtime directory. This is the path where the sgl runtime library
(sgl.dll, libsgl.so or libsgl.dynlib) resides.)doc";

static const char *__doc_sgl_platform_save_file_dialog =
R"doc(Show a file save dialog.

Parameter ``filters``:
    List of file filters.

Returns:
    The selected file path or nothing if the dialog was cancelled.)doc";

static const char *__doc_sgl_platform_set_keyboard_interrupt_handler =
R"doc(Setup a callback function to be called when Ctrl-C is detected. Use
nullptr to remove handler.)doc";

static const char *__doc_sgl_platform_set_python_active = R"doc(Inform the platform that the library is loaded from python.)doc";

static const char *__doc_sgl_platform_set_window_icon = R"doc(Set the window icon.)doc";

static const char *__doc_sgl_platform_static_init = R"doc(Initialize the platform layer.)doc";

static const char *__doc_sgl_platform_static_shutdown = R"doc(Shutdown the platform layer.)doc";

static const char *__doc_sgl_ref = R"doc()doc";

static const char *__doc_sgl_ref_2 =
R"doc(Reference counting helper.

The *ref* template is a simple wrapper to store a pointer to an
object. It takes care of increasing and decreasing the object's
reference count as needed. When the last reference goes out of scope,
the associated object will be deallocated.

This class follows similar semantics to the ``std::shared_ptr`` class
from the STL. In particular, we avoid implicit conversion to and from
raw pointers.)doc";

static const char *__doc_sgl_ref_dec_ref = R"doc()doc";

static const char *__doc_sgl_ref_get = R"doc(Return a pointer to the referenced object.)doc";

static const char *__doc_sgl_ref_inc_ref = R"doc()doc";

static const char *__doc_sgl_ref_m_ptr = R"doc()doc";

static const char *__doc_sgl_ref_operator_T0 = R"doc(Return a pointer to the referenced object.)doc";

static const char *__doc_sgl_ref_operator_assign = R"doc(Assign another reference into the current one.)doc";

static const char *__doc_sgl_ref_operator_assign_2 = R"doc(Assign another convertible reference into the current one.)doc";

static const char *__doc_sgl_ref_operator_assign_3 = R"doc(Move another reference into the current one.)doc";

static const char *__doc_sgl_ref_operator_assign_4 = R"doc(Move another convertible reference into the current one.)doc";

static const char *__doc_sgl_ref_operator_bool = R"doc(Check if the object is defined)doc";

static const char *__doc_sgl_ref_operator_eq = R"doc(Compare this reference to another reference.)doc";

static const char *__doc_sgl_ref_operator_eq_2 = R"doc(Compare this reference to a pointer.)doc";

static const char *__doc_sgl_ref_operator_eq_3 = R"doc(Compare this reference to a null pointer.)doc";

static const char *__doc_sgl_ref_operator_lt = R"doc(Compare this reference to another reference.)doc";

static const char *__doc_sgl_ref_operator_lt_2 = R"doc(Compare this reference to a null pointer.)doc";

static const char *__doc_sgl_ref_operator_mul = R"doc(Return a C++ reference to the referenced object.)doc";

static const char *__doc_sgl_ref_operator_ne = R"doc(Compare this reference to another reference.)doc";

static const char *__doc_sgl_ref_operator_ne_2 = R"doc(Compare this reference to a pointer.)doc";

static const char *__doc_sgl_ref_operator_ne_3 = R"doc(Compare this reference to a null pointer.)doc";

static const char *__doc_sgl_ref_operator_sub = R"doc(Access the object referenced by this reference.)doc";

static const char *__doc_sgl_ref_ref = R"doc(Default constructor (nullptr).)doc";

static const char *__doc_sgl_ref_ref_2 = R"doc(Construct a reference from a nullptr.)doc";

static const char *__doc_sgl_ref_ref_3 = R"doc(Construct a reference from a convertible pointer.)doc";

static const char *__doc_sgl_ref_ref_4 = R"doc(Copy constructor.)doc";

static const char *__doc_sgl_ref_ref_5 = R"doc(Construct a reference from a convertible reference.)doc";

static const char *__doc_sgl_ref_ref_6 = R"doc(Construct a reference from a pointer.)doc";

static const char *__doc_sgl_ref_ref_7 = R"doc(Move constructor.)doc";

static const char *__doc_sgl_ref_ref_8 = R"doc(Construct a reference by moving from a convertible reference.)doc";

static const char *__doc_sgl_ref_reset = R"doc(Overwrite this reference with a pointer to another object)doc";

static const char *__doc_sgl_ref_swap = R"doc(Swap this reference with another reference.)doc";

static const char *__doc_sgl_remove_ref = R"doc()doc";

static const char *__doc_sgl_renderdoc_end_frame_capture =
R"doc(End capturing a frame in RenderDoc.

This function will end capturing a frame (or some partial
compute/graphics workload) in RenderDoc.

Returns:
    True if the frame capture was ended successfully.)doc";

static const char *__doc_sgl_renderdoc_is_available =
R"doc(Check if RenderDoc is available.

This is typically the case when the application is running under the
RenderDoc.

Returns:
    True if RenderDoc is available.)doc";

static const char *__doc_sgl_renderdoc_is_frame_capturing =
R"doc(Check if a frame is currently being captured in RenderDoc.

Returns:
    True if a frame is currently being captured.)doc";

static const char *__doc_sgl_renderdoc_start_frame_capture =
R"doc(Start capturing a frame in RenderDoc.

This function will start capturing a frame (or some partial
compute/graphics workload) in RenderDoc.

To end the frame capture, call ``end_frame_capture``().

Parameter ``device``:
    The device to capture the frame for.

Parameter ``window``:
    The window to capture the frame for (optional).

Returns:
    True if the frame capture was started successfully.)doc";

static const char *__doc_sgl_report_assertion = R"doc(Report a failed assertion.)doc";

static const char *__doc_sgl_set_exception_diagnostics = R"doc(Set exception diagnostic options.)doc";

static const char *__doc_sgl_short_vector = R"doc()doc";

static const char *__doc_sgl_short_vector_2 =
R"doc(A vector that stores a small number of elements on the stack.

Template parameter ``T``:
    Element type

Template parameter ``N``:
    Size of the short vector)doc";

static const char *__doc_sgl_short_vector_back = R"doc()doc";

static const char *__doc_sgl_short_vector_back_2 = R"doc()doc";

static const char *__doc_sgl_short_vector_begin = R"doc()doc";

static const char *__doc_sgl_short_vector_begin_2 = R"doc()doc";

static const char *__doc_sgl_short_vector_capacity = R"doc()doc";

static const char *__doc_sgl_short_vector_clear = R"doc()doc";

static const char *__doc_sgl_short_vector_data = R"doc()doc";

static const char *__doc_sgl_short_vector_data_2 = R"doc()doc";

static const char *__doc_sgl_short_vector_emplace_back = R"doc()doc";

static const char *__doc_sgl_short_vector_empty = R"doc()doc";

static const char *__doc_sgl_short_vector_end = R"doc()doc";

static const char *__doc_sgl_short_vector_end_2 = R"doc()doc";

static const char *__doc_sgl_short_vector_front = R"doc()doc";

static const char *__doc_sgl_short_vector_front_2 = R"doc()doc";

static const char *__doc_sgl_short_vector_grow = R"doc()doc";

static const char *__doc_sgl_short_vector_m_capacity = R"doc()doc";

static const char *__doc_sgl_short_vector_m_data = R"doc()doc";

static const char *__doc_sgl_short_vector_m_short_data = R"doc()doc";

static const char *__doc_sgl_short_vector_m_size = R"doc()doc";

static const char *__doc_sgl_short_vector_operator_array = R"doc()doc";

static const char *__doc_sgl_short_vector_operator_array_2 = R"doc()doc";

static const char *__doc_sgl_short_vector_operator_assign = R"doc()doc";

static const char *__doc_sgl_short_vector_operator_assign_2 = R"doc()doc";

static const char *__doc_sgl_short_vector_pop_back = R"doc()doc";

static const char *__doc_sgl_short_vector_push_back = R"doc()doc";

static const char *__doc_sgl_short_vector_push_back_2 = R"doc()doc";

static const char *__doc_sgl_short_vector_reserve = R"doc()doc";

static const char *__doc_sgl_short_vector_short_vector = R"doc(Default constructor.)doc";

static const char *__doc_sgl_short_vector_short_vector_2 = R"doc(Size constructor.)doc";

static const char *__doc_sgl_short_vector_short_vector_3 = R"doc(Initializer list constructor.)doc";

static const char *__doc_sgl_short_vector_short_vector_4 = R"doc()doc";

static const char *__doc_sgl_short_vector_short_vector_5 = R"doc()doc";

static const char *__doc_sgl_short_vector_size = R"doc()doc";

static const char *__doc_sgl_slangpy_AccessType = R"doc()doc";

static const char *__doc_sgl_slangpy_AccessType_info = R"doc()doc";

static const char *__doc_sgl_slangpy_AccessType_none = R"doc()doc";

static const char *__doc_sgl_slangpy_AccessType_read = R"doc()doc";

static const char *__doc_sgl_slangpy_AccessType_readwrite = R"doc()doc";

static const char *__doc_sgl_slangpy_AccessType_write = R"doc()doc";

static const char *__doc_sgl_slangpy_CallContext = R"doc()doc";

static const char *__doc_sgl_slangpy_CallContext_CallContext = R"doc()doc";

static const char *__doc_sgl_slangpy_CallContext_call_mode = R"doc()doc";

static const char *__doc_sgl_slangpy_CallContext_call_shape = R"doc()doc";

static const char *__doc_sgl_slangpy_CallContext_device = R"doc()doc";

static const char *__doc_sgl_slangpy_CallContext_m_call_mode = R"doc()doc";

static const char *__doc_sgl_slangpy_CallContext_m_call_shape = R"doc()doc";

static const char *__doc_sgl_slangpy_CallContext_m_device = R"doc()doc";

static const char *__doc_sgl_slangpy_CallMode = R"doc()doc";

static const char *__doc_sgl_slangpy_CallMode_bwds = R"doc()doc";

static const char *__doc_sgl_slangpy_CallMode_fwds = R"doc()doc";

static const char *__doc_sgl_slangpy_CallMode_info = R"doc()doc";

static const char *__doc_sgl_slangpy_CallMode_prim = R"doc()doc";

static const char *__doc_sgl_slangpy_Shape = R"doc()doc";

static const char *__doc_sgl_slangpy_Shape_Shape = R"doc()doc";

static const char *__doc_sgl_slangpy_Shape_Shape_2 = R"doc(Constructor from optional 'tuple'.)doc";

static const char *__doc_sgl_slangpy_Shape_Shape_3 = R"doc(Constructor from initializer list)doc";

static const char *__doc_sgl_slangpy_Shape_Shape_4 = R"doc(Copy constructor.)doc";

static const char *__doc_sgl_slangpy_Shape_Shape_5 = R"doc(Move constructor.)doc";

static const char *__doc_sgl_slangpy_Shape_as_vector = R"doc(Access to internal vector.)doc";

static const char *__doc_sgl_slangpy_Shape_as_vector_2 = R"doc(Const access to internal vector.)doc";

static const char *__doc_sgl_slangpy_Shape_calc_contiguous_strides =
R"doc(Calculate the strides of a buffer of this shape, assuming it is
contiguous.)doc";

static const char *__doc_sgl_slangpy_Shape_concrete = R"doc(Check if concrete shape (no dimensions are -1).)doc";

static const char *__doc_sgl_slangpy_Shape_element_count = R"doc(Total element count (if this represented contiguous array))doc";

static const char *__doc_sgl_slangpy_Shape_m_shape = R"doc()doc";

static const char *__doc_sgl_slangpy_Shape_operator_add = R"doc(Add operator combines the 2 shapes.)doc";

static const char *__doc_sgl_slangpy_Shape_operator_array = R"doc(Indexers.)doc";

static const char *__doc_sgl_slangpy_Shape_operator_array_2 = R"doc()doc";

static const char *__doc_sgl_slangpy_Shape_operator_assign = R"doc(Assignment operator.)doc";

static const char *__doc_sgl_slangpy_Shape_operator_eq = R"doc()doc";

static const char *__doc_sgl_slangpy_Shape_size = R"doc(Get size (i.e. number of dimensions) of shape.)doc";

static const char *__doc_sgl_slangpy_Shape_to_string = R"doc(Convert to string)doc";

static const char *__doc_sgl_slangpy_Shape_valid = R"doc(Check if shape is valid (if the std::optional has a value).)doc";

static const char *__doc_sgl_slangpy_find_enum_info_adl = R"doc()doc";

static const char *__doc_sgl_slangpy_find_enum_info_adl_2 = R"doc()doc";

static const char *__doc_sgl_static_ref_cast = R"doc()doc";

static const char *__doc_sgl_static_vector = R"doc()doc";

static const char *__doc_sgl_static_vector_2 =
R"doc(A vector that stores up to a fixed number of elements.

Template parameter ``T``:
    Element type

Template parameter ``N``:
    Maximum size of the static vector)doc";

static const char *__doc_sgl_static_vector_back = R"doc()doc";

static const char *__doc_sgl_static_vector_back_2 = R"doc()doc";

static const char *__doc_sgl_static_vector_begin = R"doc()doc";

static const char *__doc_sgl_static_vector_begin_2 = R"doc()doc";

static const char *__doc_sgl_static_vector_capacity = R"doc()doc";

static const char *__doc_sgl_static_vector_clear = R"doc()doc";

static const char *__doc_sgl_static_vector_data = R"doc()doc";

static const char *__doc_sgl_static_vector_data_2 = R"doc()doc";

static const char *__doc_sgl_static_vector_emplace_back = R"doc()doc";

static const char *__doc_sgl_static_vector_empty = R"doc()doc";

static const char *__doc_sgl_static_vector_end = R"doc()doc";

static const char *__doc_sgl_static_vector_end_2 = R"doc()doc";

static const char *__doc_sgl_static_vector_front = R"doc()doc";

static const char *__doc_sgl_static_vector_front_2 = R"doc()doc";

static const char *__doc_sgl_static_vector_m_data = R"doc()doc";

static const char *__doc_sgl_static_vector_m_size = R"doc()doc";

static const char *__doc_sgl_static_vector_operator_array = R"doc()doc";

static const char *__doc_sgl_static_vector_operator_array_2 = R"doc()doc";

static const char *__doc_sgl_static_vector_operator_assign = R"doc()doc";

static const char *__doc_sgl_static_vector_operator_assign_2 = R"doc()doc";

static const char *__doc_sgl_static_vector_pop_back = R"doc()doc";

static const char *__doc_sgl_static_vector_push_back = R"doc()doc";

static const char *__doc_sgl_static_vector_push_back_2 = R"doc()doc";

static const char *__doc_sgl_static_vector_size = R"doc()doc";

static const char *__doc_sgl_static_vector_static_vector = R"doc(Default constructor.)doc";

static const char *__doc_sgl_static_vector_static_vector_2 = R"doc(Size constructor.)doc";

static const char *__doc_sgl_static_vector_static_vector_3 = R"doc(Initializer list constructor.)doc";

static const char *__doc_sgl_static_vector_static_vector_4 = R"doc()doc";

static const char *__doc_sgl_static_vector_static_vector_5 = R"doc()doc";

static const char *__doc_sgl_string_copy_to_cstr =
R"doc(Copy a string to a C-style string buffer. Truncates the input string
if it doesn't fit the buffer and always null-terminates the buffer.

Parameter ``dst``:
    C-style string buffer.

Parameter ``dst_len``:
    Length of the buffer.

Parameter ``src``:
    Input string.)doc";

static const char *__doc_sgl_string_decode_base64 =
R"doc(Decode data from base 64 encoding.

Parameter ``str``:
    Base 64 encoded string.

Returns:
    Decoded data.)doc";

static const char *__doc_sgl_string_encode_base64 =
R"doc(Encode data into base 64 encoding.

Parameter ``data``:
    Input data.

Parameter ``len``:
    Length of input data.

Returns:
    Base 64 encoded string.)doc";

static const char *__doc_sgl_string_encode_base64_2 =
R"doc(Encode data into base 64 encoding.

Parameter ``data``:
    Input data.

Returns:
    Base 64 encoded string.)doc";

static const char *__doc_sgl_string_format_byte_size =
R"doc(Convert a size in bytes to a human readable string: - prints bytes (B)
if size < 1000 bytes - prints kilobytes (KB) if size < 1000 kilobytes
- prints megabytes (MB) if size < 1000 megabytes - prints gigabytes
(GB) if size < 1000 gigabytes - otherwise prints terabytes (TB)

Parameter ``size``:
    Size in bytes.

Returns:
    Size as a human readable string.)doc";

static const char *__doc_sgl_string_format_duration =
R"doc(Convert a duration in seconds to a human readable string: - prints
nanoseconds (ns) if duration < 1 microsecond - prints microseconds
(us) if duration < 1 millisecond - prints milliseconds (ms) if
duration < 1 second - prints seconds (s) if duration < 60 seconds -
prints minutes (m) if duration < 60 minutes - prints hours (h) if
duration < 24 hours - otherwise prints days (d)

Parameter ``seconds``:
    Duration in seconds.

Returns:
    Duration as a human readable string.)doc";

static const char *__doc_sgl_string_has_prefix =
R"doc(Check is a string has the specified prefix.

Parameter ``str``:
    Input string.

Parameter ``prefix``:
    Prefix to check for.

Parameter ``case_sensitive``:
    Whether comparison should be case-sensitive.

Returns:
    True if the input string has the specified prefix.)doc";

static const char *__doc_sgl_string_has_suffix =
R"doc(Check is a string has the specified suffix.

Parameter ``str``:
    Input string.

Parameter ``suffix``:
    Suffix to check for.

Parameter ``case_sensitive``:
    Whether comparison should be case-sensitive.

Returns:
    True if the input string has the specified suffix.)doc";

static const char *__doc_sgl_string_hexlify =
R"doc(Convert data to a hexadecimal string.

Parameter ``data``:
    Input data.

Parameter ``len``:
    Length of input data.

Returns:
    Hexadecimal string.)doc";

static const char *__doc_sgl_string_hexlify_2 =
R"doc(Convert data to a hexadecimal string.

Parameter ``data``:
    Input data.

Returns:
    Hexadecimal string.)doc";

static const char *__doc_sgl_string_indent =
R"doc(Replace every newline with the specified indentation.

Parameter ``str``:
    Input string.

Parameter ``indentation``:
    Indentation string.

Returns:
    The indented string.)doc";

static const char *__doc_sgl_string_iterable_to_string = R"doc()doc";

static const char *__doc_sgl_string_join =
R"doc(Join an list of strings separated by a specified seperator.

Parameter ``strings``:
    Input strings.

Parameter ``separator``:
    String placed between each string to be joined.

Returns:
    Joined string.)doc";

static const char *__doc_sgl_string_list_to_flags =
R"doc(Convert a list of strings to a flags enum value. Throws if any of the
strings are not found in the registered enum information.)doc";

static const char *__doc_sgl_string_list_to_string =
R"doc(Convert a list of objects that have a to_string() method to a string.

Template parameter ``T``:
    Type of objects.

Parameter ``list``:
    List of objects.

Returns:
    The list of objects as a string.)doc";

static const char *__doc_sgl_string_list_to_string_2 =
R"doc(Convert a list of objects that have a free-standing to_string() method
to a string.

Template parameter ``T``:
    Type of objects.

Parameter ``list``:
    List of objects.

Returns:
    The list of objects as a string.)doc";

static const char *__doc_sgl_string_list_to_string_3 = R"doc()doc";

static const char *__doc_sgl_string_remove_leading_trailing_whitespace =
R"doc(Remove leading and trailing whitespace.

Parameter ``str``:
    Input string.

Parameter ``whitespace``:
    Whitespace characters.

Returns:
    String with leading and trailing whitespace removed.)doc";

static const char *__doc_sgl_string_remove_leading_whitespace =
R"doc(Remove leading whitespace.

Parameter ``str``:
    Input string.

Parameter ``whitespace``:
    Whitespace characters.

Returns:
    String with leading whitespace removed.)doc";

static const char *__doc_sgl_string_remove_trailing_whitespace =
R"doc(Remove trailing whitespace.

Parameter ``str``:
    Input string.

Parameter ``whitespace``:
    Whitespace characters.

Returns:
    String with trailing whitespace removed.)doc";

static const char *__doc_sgl_string_split =
R"doc(Split a string into a list of strings based on a list of delimiter
characters.

Parameter ``str``:
    Input string.

Parameter ``delimiters``:
    Delimiter characters.

Returns:
    List of split strings excluding the delimiter characters.)doc";

static const char *__doc_sgl_string_to_enum =
R"doc(Convert a string to an enum value. Throws if the string is not found
in the registered enum information.)doc";

static const char *__doc_sgl_string_to_lower =
R"doc(Convert a string to lower case.

Parameter ``str``:
    Input string.

Returns:
    Lower case string.)doc";

static const char *__doc_sgl_string_to_upper =
R"doc(Convert a string to upper case.

Parameter ``str``:
    Input string.

Returns:
    Upper case string.)doc";

static const char *__doc_sgl_string_to_wstring =
R"doc(Convert a string to wide string.

Parameter ``str``:
    Input string.

Returns:
    Wide string.)doc";

static const char *__doc_sgl_tev_show =
R"doc(Show a bitmap in the tev viewer (https://github.com/Tom94/tev).

This will block until the image is sent over.

Parameter ``bitmap``:
    Bitmap to show.

Parameter ``name``:
    Name of the image in tev. If not specified, a unique name will be
    generated.

Parameter ``host``:
    Host to connect to.

Parameter ``port``:
    Port to connect to.

Parameter ``max_retries``:
    Maximum number of retries.

Returns:
    True if successful.)doc";

static const char *__doc_sgl_tev_show_2 =
R"doc(Show texture in the tev viewer (https://github.com/Tom94/tev).

This will block until the image is sent over.

Parameter ``texture``:
    Texture to show.

Parameter ``name``:
    Name of the image in tev. If not specified, a unique name will be
    generated.

Parameter ``host``:
    Host to connect to.

Parameter ``port``:
    Port to connect to.

Parameter ``max_retries``:
    Maximum number of retries.

Returns:
    True if successful.)doc";

static const char *__doc_sgl_tev_show_async =
R"doc(Show a bitmap in the tev viewer (https://github.com/Tom94/tev).

This will return immediately and send the image asynchronously in the
background.

Parameter ``bitmap``:
    Bitmap to show.

Parameter ``name``:
    Name of the image in tev. If not specified, a unique name will be
    generated.

Parameter ``host``:
    Host to connect to.

Parameter ``port``:
    Port to connect to.

Parameter ``max_retries``:
    Maximum number of retries.)doc";

static const char *__doc_sgl_tev_show_async_2 =
R"doc(Show a texture in the tev viewer (https://github.com/Tom94/tev).

This will return immediately and send the image asynchronously in the
background.

Parameter ``bitmap``:
    Texture to show.

Parameter ``name``:
    Name of the image in tev. If not specified, a unique name will be
    generated.

Parameter ``host``:
    Host to connect to.

Parameter ``port``:
    Port to connect to.

Parameter ``max_retries``:
    Maximum number of retries.)doc";

static const char *__doc_sgl_thread_do_async = R"doc()doc";

static const char *__doc_sgl_thread_global_thread_pool = R"doc()doc";

static const char *__doc_sgl_thread_static_init = R"doc()doc";

static const char *__doc_sgl_thread_static_shutdown = R"doc()doc";

static const char *__doc_sgl_thread_wait_for_tasks = R"doc(Block until all scheduled tasks are completed.)doc";

static const char *__doc_sgl_throw_exception = R"doc(Throw an exception.)doc";

static const char *__doc_sgl_ui_Button = R"doc()doc";

static const char *__doc_sgl_ui_Button_2 = R"doc()doc";

static const char *__doc_sgl_ui_Button_Button = R"doc()doc";

static const char *__doc_sgl_ui_Button_callback = R"doc()doc";

static const char *__doc_sgl_ui_Button_class_name = R"doc()doc";

static const char *__doc_sgl_ui_Button_dispatch_event = R"doc()doc";

static const char *__doc_sgl_ui_Button_label = R"doc()doc";

static const char *__doc_sgl_ui_Button_m_callback = R"doc()doc";

static const char *__doc_sgl_ui_Button_m_label = R"doc()doc";

static const char *__doc_sgl_ui_Button_render = R"doc()doc";

static const char *__doc_sgl_ui_Button_set_callback = R"doc()doc";

static const char *__doc_sgl_ui_Button_set_label = R"doc()doc";

static const char *__doc_sgl_ui_CheckBox = R"doc()doc";

static const char *__doc_sgl_ui_CheckBox_2 = R"doc()doc";

static const char *__doc_sgl_ui_CheckBox_CheckBox = R"doc()doc";

static const char *__doc_sgl_ui_CheckBox_class_name = R"doc()doc";

static const char *__doc_sgl_ui_CheckBox_render = R"doc()doc";

static const char *__doc_sgl_ui_ComboBox = R"doc()doc";

static const char *__doc_sgl_ui_ComboBox_2 = R"doc()doc";

static const char *__doc_sgl_ui_ComboBox_ComboBox = R"doc()doc";

static const char *__doc_sgl_ui_ComboBox_class_name = R"doc()doc";

static const char *__doc_sgl_ui_ComboBox_items = R"doc()doc";

static const char *__doc_sgl_ui_ComboBox_m_items = R"doc()doc";

static const char *__doc_sgl_ui_ComboBox_render = R"doc()doc";

static const char *__doc_sgl_ui_ComboBox_set_items = R"doc()doc";

static const char *__doc_sgl_ui_Context = R"doc()doc";

static const char *__doc_sgl_ui_Context_2 = R"doc()doc";

static const char *__doc_sgl_ui_Context_Context = R"doc()doc";

static const char *__doc_sgl_ui_Context_class_name = R"doc()doc";

static const char *__doc_sgl_ui_Context_get_font = R"doc()doc";

static const char *__doc_sgl_ui_Context_get_pipeline = R"doc()doc";

static const char *__doc_sgl_ui_Context_handle_keyboard_event = R"doc()doc";

static const char *__doc_sgl_ui_Context_handle_mouse_event = R"doc()doc";

static const char *__doc_sgl_ui_Context_m_device = R"doc()doc";

static const char *__doc_sgl_ui_Context_m_font_texture = R"doc()doc";

static const char *__doc_sgl_ui_Context_m_fonts = R"doc()doc";

static const char *__doc_sgl_ui_Context_m_frame_index = R"doc()doc";

static const char *__doc_sgl_ui_Context_m_frame_timer = R"doc()doc";

static const char *__doc_sgl_ui_Context_m_imgui_context = R"doc()doc";

static const char *__doc_sgl_ui_Context_m_index_buffers = R"doc()doc";

static const char *__doc_sgl_ui_Context_m_input_layout = R"doc()doc";

static const char *__doc_sgl_ui_Context_m_pipelines = R"doc()doc";

static const char *__doc_sgl_ui_Context_m_program = R"doc()doc";

static const char *__doc_sgl_ui_Context_m_sampler = R"doc()doc";

static const char *__doc_sgl_ui_Context_m_screen = R"doc()doc";

static const char *__doc_sgl_ui_Context_m_vertex_buffers = R"doc()doc";

static const char *__doc_sgl_ui_Context_new_frame = R"doc()doc";

static const char *__doc_sgl_ui_Context_process_events = R"doc()doc";

static const char *__doc_sgl_ui_Context_render = R"doc()doc";

static const char *__doc_sgl_ui_Context_render_2 = R"doc()doc";

static const char *__doc_sgl_ui_Context_screen = R"doc()doc";

static const char *__doc_sgl_ui_DataTypeTraits = R"doc()doc";

static const char *__doc_sgl_ui_DataTypeTraits_2 = R"doc()doc";

static const char *__doc_sgl_ui_DataTypeTraits_3 = R"doc()doc";

static const char *__doc_sgl_ui_Drag = R"doc()doc";

static const char *__doc_sgl_ui_Drag_2 = R"doc()doc";

static const char *__doc_sgl_ui_Drag_Drag = R"doc()doc";

static const char *__doc_sgl_ui_Drag_class_name = R"doc()doc";

static const char *__doc_sgl_ui_Drag_flags = R"doc()doc";

static const char *__doc_sgl_ui_Drag_format = R"doc()doc";

static const char *__doc_sgl_ui_Drag_m_flags = R"doc()doc";

static const char *__doc_sgl_ui_Drag_m_format = R"doc()doc";

static const char *__doc_sgl_ui_Drag_m_max = R"doc()doc";

static const char *__doc_sgl_ui_Drag_m_min = R"doc()doc";

static const char *__doc_sgl_ui_Drag_m_speed = R"doc()doc";

static const char *__doc_sgl_ui_Drag_max = R"doc()doc";

static const char *__doc_sgl_ui_Drag_min = R"doc()doc";

static const char *__doc_sgl_ui_Drag_render = R"doc()doc";

static const char *__doc_sgl_ui_Drag_set_flags = R"doc()doc";

static const char *__doc_sgl_ui_Drag_set_format = R"doc()doc";

static const char *__doc_sgl_ui_Drag_set_max = R"doc()doc";

static const char *__doc_sgl_ui_Drag_set_min = R"doc()doc";

static const char *__doc_sgl_ui_Drag_set_speed = R"doc()doc";

static const char *__doc_sgl_ui_Drag_speed = R"doc()doc";

static const char *__doc_sgl_ui_Event = R"doc()doc";

static const char *__doc_sgl_ui_Event_sender = R"doc()doc";

static const char *__doc_sgl_ui_Group = R"doc()doc";

static const char *__doc_sgl_ui_Group_2 = R"doc()doc";

static const char *__doc_sgl_ui_Group_Group = R"doc()doc";

static const char *__doc_sgl_ui_Group_class_name = R"doc()doc";

static const char *__doc_sgl_ui_Group_label = R"doc()doc";

static const char *__doc_sgl_ui_Group_m_label = R"doc()doc";

static const char *__doc_sgl_ui_Group_render = R"doc()doc";

static const char *__doc_sgl_ui_Group_set_label = R"doc()doc";

static const char *__doc_sgl_ui_Input = R"doc()doc";

static const char *__doc_sgl_ui_InputText = R"doc()doc";

static const char *__doc_sgl_ui_InputText_2 = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_allow_tab_input = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_always_overwrite = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_auto_select_all = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_callback_always = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_callback_char_filter = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_callback_completion = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_callback_history = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_chars_decimal = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_chars_hexadecimal = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_chars_no_blank = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_chars_scientific = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_chars_uppercase = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_ctrl_enter_for_new_line = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_enter_returns_true = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_escape_clears_all = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_no_horizontal_scroll = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_no_undo_redo = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_none = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_password = R"doc()doc";

static const char *__doc_sgl_ui_InputTextFlags_read_only = R"doc()doc";

static const char *__doc_sgl_ui_InputText_InputText = R"doc()doc";

static const char *__doc_sgl_ui_InputText_class_name = R"doc()doc";

static const char *__doc_sgl_ui_InputText_m_flags = R"doc()doc";

static const char *__doc_sgl_ui_InputText_m_multi_line = R"doc()doc";

static const char *__doc_sgl_ui_InputText_render = R"doc()doc";

static const char *__doc_sgl_ui_Input_Input = R"doc()doc";

static const char *__doc_sgl_ui_Input_class_name = R"doc()doc";

static const char *__doc_sgl_ui_Input_flags = R"doc()doc";

static const char *__doc_sgl_ui_Input_format = R"doc()doc";

static const char *__doc_sgl_ui_Input_m_flags = R"doc()doc";

static const char *__doc_sgl_ui_Input_m_format = R"doc()doc";

static const char *__doc_sgl_ui_Input_m_step = R"doc()doc";

static const char *__doc_sgl_ui_Input_m_step_fast = R"doc()doc";

static const char *__doc_sgl_ui_Input_render = R"doc()doc";

static const char *__doc_sgl_ui_Input_set_flags = R"doc()doc";

static const char *__doc_sgl_ui_Input_set_format = R"doc()doc";

static const char *__doc_sgl_ui_Input_set_step = R"doc()doc";

static const char *__doc_sgl_ui_Input_set_step_fast = R"doc()doc";

static const char *__doc_sgl_ui_Input_step = R"doc()doc";

static const char *__doc_sgl_ui_Input_step_fast = R"doc()doc";

static const char *__doc_sgl_ui_ListBox = R"doc()doc";

static const char *__doc_sgl_ui_ListBox_2 = R"doc()doc";

static const char *__doc_sgl_ui_ListBox_ListBox = R"doc()doc";

static const char *__doc_sgl_ui_ListBox_class_name = R"doc()doc";

static const char *__doc_sgl_ui_ListBox_height_in_items = R"doc()doc";

static const char *__doc_sgl_ui_ListBox_items = R"doc()doc";

static const char *__doc_sgl_ui_ListBox_m_height_in_items = R"doc()doc";

static const char *__doc_sgl_ui_ListBox_m_items = R"doc()doc";

static const char *__doc_sgl_ui_ListBox_render = R"doc()doc";

static const char *__doc_sgl_ui_ListBox_set_height_in_items = R"doc()doc";

static const char *__doc_sgl_ui_ListBox_set_items = R"doc()doc";

static const char *__doc_sgl_ui_ProgressBar = R"doc()doc";

static const char *__doc_sgl_ui_ProgressBar_2 = R"doc()doc";

static const char *__doc_sgl_ui_ProgressBar_ProgressBar = R"doc()doc";

static const char *__doc_sgl_ui_ProgressBar_class_name = R"doc()doc";

static const char *__doc_sgl_ui_ProgressBar_fraction = R"doc()doc";

static const char *__doc_sgl_ui_ProgressBar_m_fraction = R"doc()doc";

static const char *__doc_sgl_ui_ProgressBar_render = R"doc()doc";

static const char *__doc_sgl_ui_ProgressBar_set_fraction = R"doc()doc";

static const char *__doc_sgl_ui_PropertyValue = R"doc()doc";

static const char *__doc_sgl_ui_ScopedDisable = R"doc()doc";

static const char *__doc_sgl_ui_ScopedDisable_2 = R"doc(Scoped begin/end for disabling ImGUI widgets.)doc";

static const char *__doc_sgl_ui_ScopedDisable_ScopedDisable = R"doc()doc";

static const char *__doc_sgl_ui_ScopedDisable_m_disabled = R"doc()doc";

static const char *__doc_sgl_ui_ScopedID = R"doc()doc";

static const char *__doc_sgl_ui_ScopedID_2 = R"doc(Scoped push/pop of ImGui ID.)doc";

static const char *__doc_sgl_ui_ScopedID_ScopedID = R"doc()doc";

static const char *__doc_sgl_ui_Screen = R"doc()doc";

static const char *__doc_sgl_ui_Screen_2 =
R"doc(This is the main widget that represents the screen. It is intended to
be used as the parent for ``Window`` widgets.)doc";

static const char *__doc_sgl_ui_Screen_Screen = R"doc()doc";

static const char *__doc_sgl_ui_Screen_class_name = R"doc()doc";

static const char *__doc_sgl_ui_Screen_dispatch_events = R"doc()doc";

static const char *__doc_sgl_ui_Screen_m_events = R"doc()doc";

static const char *__doc_sgl_ui_Screen_record_event = R"doc()doc";

static const char *__doc_sgl_ui_Screen_render = R"doc()doc";

static const char *__doc_sgl_ui_Slider = R"doc()doc";

static const char *__doc_sgl_ui_Slider_2 = R"doc()doc";

static const char *__doc_sgl_ui_SliderFlags = R"doc()doc";

static const char *__doc_sgl_ui_SliderFlags_always_clamp = R"doc()doc";

static const char *__doc_sgl_ui_SliderFlags_logarithmic = R"doc()doc";

static const char *__doc_sgl_ui_SliderFlags_no_input = R"doc()doc";

static const char *__doc_sgl_ui_SliderFlags_no_round_to_format = R"doc()doc";

static const char *__doc_sgl_ui_SliderFlags_none = R"doc()doc";

static const char *__doc_sgl_ui_Slider_Slider = R"doc()doc";

static const char *__doc_sgl_ui_Slider_class_name = R"doc()doc";

static const char *__doc_sgl_ui_Slider_flags = R"doc()doc";

static const char *__doc_sgl_ui_Slider_format = R"doc()doc";

static const char *__doc_sgl_ui_Slider_m_flags = R"doc()doc";

static const char *__doc_sgl_ui_Slider_m_format = R"doc()doc";

static const char *__doc_sgl_ui_Slider_m_max = R"doc()doc";

static const char *__doc_sgl_ui_Slider_m_min = R"doc()doc";

static const char *__doc_sgl_ui_Slider_max = R"doc()doc";

static const char *__doc_sgl_ui_Slider_min = R"doc()doc";

static const char *__doc_sgl_ui_Slider_render = R"doc()doc";

static const char *__doc_sgl_ui_Slider_set_flags = R"doc()doc";

static const char *__doc_sgl_ui_Slider_set_format = R"doc()doc";

static const char *__doc_sgl_ui_Slider_set_max = R"doc()doc";

static const char *__doc_sgl_ui_Slider_set_min = R"doc()doc";

static const char *__doc_sgl_ui_Text = R"doc()doc";

static const char *__doc_sgl_ui_Text_2 = R"doc()doc";

static const char *__doc_sgl_ui_Text_Text = R"doc()doc";

static const char *__doc_sgl_ui_Text_class_name = R"doc()doc";

static const char *__doc_sgl_ui_Text_m_text = R"doc()doc";

static const char *__doc_sgl_ui_Text_render = R"doc()doc";

static const char *__doc_sgl_ui_Text_set_text = R"doc()doc";

static const char *__doc_sgl_ui_Text_text = R"doc()doc";

static const char *__doc_sgl_ui_ValueProperty = R"doc()doc";

static const char *__doc_sgl_ui_ValueProperty_ValueProperty = R"doc()doc";

static const char *__doc_sgl_ui_ValueProperty_callback = R"doc()doc";

static const char *__doc_sgl_ui_ValueProperty_dispatch_event = R"doc()doc";

static const char *__doc_sgl_ui_ValueProperty_label = R"doc()doc";

static const char *__doc_sgl_ui_ValueProperty_m_callback = R"doc()doc";

static const char *__doc_sgl_ui_ValueProperty_m_label = R"doc()doc";

static const char *__doc_sgl_ui_ValueProperty_m_value = R"doc()doc";

static const char *__doc_sgl_ui_ValueProperty_set_callback = R"doc()doc";

static const char *__doc_sgl_ui_ValueProperty_set_label = R"doc()doc";

static const char *__doc_sgl_ui_ValueProperty_set_value = R"doc()doc";

static const char *__doc_sgl_ui_ValueProperty_value = R"doc()doc";

static const char *__doc_sgl_ui_VectorTraits = R"doc()doc";

static const char *__doc_sgl_ui_VectorTraits_2 = R"doc()doc";

static const char *__doc_sgl_ui_VectorTraits_3 = R"doc()doc";

static const char *__doc_sgl_ui_VectorTraits_4 = R"doc()doc";

static const char *__doc_sgl_ui_VectorTraits_5 = R"doc()doc";

static const char *__doc_sgl_ui_VectorTraits_6 = R"doc()doc";

static const char *__doc_sgl_ui_VectorTraits_7 = R"doc()doc";

static const char *__doc_sgl_ui_VectorTraits_8 = R"doc()doc";

static const char *__doc_sgl_ui_VectorTraits_9 = R"doc()doc";

static const char *__doc_sgl_ui_Widget = R"doc()doc";

static const char *__doc_sgl_ui_Widget_2 = R"doc(Base class for Python UI widgets. Widgets own their children.)doc";

static const char *__doc_sgl_ui_Widget_3 = R"doc(Base class for Python UI widgets. Widgets own their children.)doc";

static const char *__doc_sgl_ui_Widget_Widget = R"doc()doc";

static const char *__doc_sgl_ui_Widget_add_child = R"doc()doc";

static const char *__doc_sgl_ui_Widget_add_child_at = R"doc()doc";

static const char *__doc_sgl_ui_Widget_child_at = R"doc()doc";

static const char *__doc_sgl_ui_Widget_child_count = R"doc()doc";

static const char *__doc_sgl_ui_Widget_child_index = R"doc()doc";

static const char *__doc_sgl_ui_Widget_children = R"doc()doc";

static const char *__doc_sgl_ui_Widget_class_name = R"doc()doc";

static const char *__doc_sgl_ui_Widget_clear_children = R"doc()doc";

static const char *__doc_sgl_ui_Widget_dispatch_event = R"doc()doc";

static const char *__doc_sgl_ui_Widget_enabled = R"doc()doc";

static const char *__doc_sgl_ui_Widget_m_children = R"doc()doc";

static const char *__doc_sgl_ui_Widget_m_enabled = R"doc()doc";

static const char *__doc_sgl_ui_Widget_m_parent = R"doc()doc";

static const char *__doc_sgl_ui_Widget_m_visible = R"doc()doc";

static const char *__doc_sgl_ui_Widget_parent = R"doc()doc";

static const char *__doc_sgl_ui_Widget_parent_2 = R"doc()doc";

static const char *__doc_sgl_ui_Widget_record_event = R"doc()doc";

static const char *__doc_sgl_ui_Widget_remove_all_children = R"doc()doc";

static const char *__doc_sgl_ui_Widget_remove_child = R"doc()doc";

static const char *__doc_sgl_ui_Widget_remove_child_at = R"doc()doc";

static const char *__doc_sgl_ui_Widget_render = R"doc()doc";

static const char *__doc_sgl_ui_Widget_set_enabled = R"doc()doc";

static const char *__doc_sgl_ui_Widget_set_parent = R"doc()doc";

static const char *__doc_sgl_ui_Widget_set_visible = R"doc()doc";

static const char *__doc_sgl_ui_Widget_visible = R"doc()doc";

static const char *__doc_sgl_ui_Window = R"doc()doc";

static const char *__doc_sgl_ui_Window_2 = R"doc()doc";

static const char *__doc_sgl_ui_Window_Window = R"doc()doc";

static const char *__doc_sgl_ui_Window_class_name = R"doc()doc";

static const char *__doc_sgl_ui_Window_close = R"doc()doc";

static const char *__doc_sgl_ui_Window_m_position = R"doc()doc";

static const char *__doc_sgl_ui_Window_m_set_position = R"doc()doc";

static const char *__doc_sgl_ui_Window_m_set_size = R"doc()doc";

static const char *__doc_sgl_ui_Window_m_size = R"doc()doc";

static const char *__doc_sgl_ui_Window_m_title = R"doc()doc";

static const char *__doc_sgl_ui_Window_position = R"doc()doc";

static const char *__doc_sgl_ui_Window_render = R"doc()doc";

static const char *__doc_sgl_ui_Window_set_position = R"doc()doc";

static const char *__doc_sgl_ui_Window_set_size = R"doc()doc";

static const char *__doc_sgl_ui_Window_set_title = R"doc()doc";

static const char *__doc_sgl_ui_Window_show = R"doc()doc";

static const char *__doc_sgl_ui_Window_size = R"doc()doc";

static const char *__doc_sgl_ui_Window_title = R"doc()doc";

static const char *__doc_sgl_ui_flip_bit = R"doc()doc";

static const char *__doc_sgl_ui_flip_bit_2 = R"doc()doc";

static const char *__doc_sgl_ui_is_set = R"doc()doc";

static const char *__doc_sgl_ui_is_set_2 = R"doc()doc";

static const char *__doc_sgl_ui_operator_band = R"doc()doc";

static const char *__doc_sgl_ui_operator_band_2 = R"doc()doc";

static const char *__doc_sgl_ui_operator_bnot = R"doc()doc";

static const char *__doc_sgl_ui_operator_bnot_2 = R"doc()doc";

static const char *__doc_sgl_ui_operator_bor = R"doc()doc";

static const char *__doc_sgl_ui_operator_bor_2 = R"doc()doc";

static const char *__doc_sgl_ui_operator_iand = R"doc()doc";

static const char *__doc_sgl_ui_operator_iand_2 = R"doc()doc";

static const char *__doc_sgl_ui_operator_ior = R"doc()doc";

static const char *__doc_sgl_ui_operator_ior_2 = R"doc()doc";

static const char *__doc_std_numeric_limits = R"doc()doc";

static const char *__doc_std_numeric_limits_denorm_min = R"doc()doc";

static const char *__doc_std_numeric_limits_epsilon = R"doc()doc";

static const char *__doc_std_numeric_limits_infinity = R"doc()doc";

static const char *__doc_std_numeric_limits_lowest = R"doc()doc";

static const char *__doc_std_numeric_limits_max = R"doc()doc";

static const char *__doc_std_numeric_limits_min = R"doc()doc";

static const char *__doc_std_numeric_limits_quiet_NaN = R"doc()doc";

static const char *__doc_std_numeric_limits_round_error = R"doc()doc";

static const char *__doc_std_numeric_limits_signaling_NaN = R"doc()doc";

static const char *__doc_std_swap = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

