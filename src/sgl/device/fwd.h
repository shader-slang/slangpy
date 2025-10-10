// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/object.h"

namespace sgl {

// device.h

struct DeviceDesc;
class Device;

// surface.h

struct SurfaceInfo;
struct SurfaceConfig;
class Surface;

// buffer_cursor.h

class BufferCursor;
class BufferElementCursor;

// resource.h

class Resource;

struct BufferDesc;
class Buffer;

struct BufferViewDesc;
class BufferView;

struct TextureDesc;
class Texture;

struct TextureViewDesc;
class TextureView;

// sampler.h

struct SamplerDesc;
class Sampler;

// fence.h

struct FenceDesc;
class Fence;

// shader.h

struct SlangSessionDesc;
class SlangSession;

class SlangModule;
class SlangComponentType;
class SlangGlobalScope;
class SlangEntryPoint;

class ShaderProgram;

// reflection.h

class DeclReflection;
class DeclReflectionChildList;
class DeclReflectionIndexedChildList;
class TypeReflection;
class TypeReflectionFieldList;
class TypeLayoutReflection;
class TypeLayoutReflectionFieldList;
class FunctionReflection;
class FunctionReflectionParameterList;
class FunctionReflectionOverloadList;
class VariableReflection;
class VariableLayoutReflection;
class Attribute;
class EntryPointLayout;
class EntryPointLayoutParameterList;
class ProgramLayout;
class ProgramLayoutParameterList;
class ProgramLayoutEntryPointList;

// kernel.h

class Kernel;
struct ComputeKernelDesc;
class ComputeKernel;

// persistent_cache.h

class PersistentCache;

// pipeline.h

class Pipeline;
struct ComputePipelineDesc;
class ComputePipeline;
struct RenderPipelineDesc;
class RenderPipeline;
struct RayTracingPipelineDesc;
class RayTracingPipeline;

// query.h

struct QueryPoolDesc;
class QueryPool;

// raytracing.h

struct AccelerationStructureDesc;
class AccelerationStructure;
class AccelerationStructureInstanceList;
struct ShaderTableDesc;
class ShaderTable;

// command.h

class CommandQueue;
class CommandEncoder;
class RenderPassEncoder;
class ComputePassEncoder;
class RayTracingPassEncoder;
class CommandBuffer;

// shader_cursor.h

class ShaderCursor;
class ShaderObject;

// input_layout.h

struct InputLayoutDesc;
class InputLayout;

// blit.h

class Blitter;

// texture_loader.h

class TextureLoader;

// hot_reload.h
class HotReload;

// coopvec.h
class CoopVec;

// cuda_interop.h

namespace cuda {
    struct TensorView;
    class InteropBuffer;
} // namespace cuda

// cuda_utils.h

namespace cuda {
    class Device;
    class ExternalMemory;
    class ExternalSemaphore;
} // namespace cuda

} // namespace sgl
