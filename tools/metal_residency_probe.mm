// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Standalone diagnostic: bypass RHI's GPU-family gate and test the Metal API directly.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>

int main()
{
    @autoreleasepool {
        printf("OS: %s\n", NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
        const char* forced = std::getenv("SLANG_RHI_METAL_NO_RESIDENCY_SET");
        printf("SLANG_RHI_METAL_NO_RESIDENCY_SET: %s\n", forced ? forced : "<unset>");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No default Metal device\n");
            return 1;
        }
        printf("Device: %s\n", device.name.UTF8String);
        printf("Unified memory: %d\n", device.hasUnifiedMemory);
        printf("Argument buffer tier: %lu\n", (unsigned long)device.argumentBuffersSupport);
        for (int family = 1001; family <= 1010; ++family)
            printf("Apple%d: %d\n", family - 1000, [device supportsFamily:(MTLGPUFamily)family]);
        printf("Mac2: %d\n", [device supportsFamily:MTLGPUFamilyMac2]);
        printf("Metal3: %d\n", [device supportsFamily:MTLGPUFamilyMetal3]);
        fflush(stdout);

        if (@available(macOS 15.0, *)) {
            @try {
                printf(
                    "Residency-set selector: %d\n",
                    [device respondsToSelector:@selector(newResidencySetWithDescriptor:error:)]
                );
                NSError* error = nil;
                MTLResidencySetDescriptor* desc = [MTLResidencySetDescriptor new];
                id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:desc error:&error];
                printf("Residency-set creation: %s\n", residency ? "SUCCESS" : "UNAVAILABLE");
                if (error)
                    printf(
                        "NSError: domain=%s code=%ld description=%s\n",
                        error.domain.UTF8String,
                        (long)error.code,
                        error.localizedDescription.UTF8String
                    );
                fflush(stdout);
                // An unsupported capability is a diagnostic result, not a broken probe.
                if (!residency)
                    return 0;

                id<MTLCommandQueue> queue = [device newCommandQueue];
                id<MTLBuffer> buffer = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
                if (!queue || !buffer) {
                    fprintf(stderr, "Queue or buffer allocation failed\n");
                    return 1;
                }
                *(uint32_t*)buffer.contents = 0;
                [residency addAllocation:buffer];
                [residency commit];
                [queue addResidencySet:residency];

                // Pass only address bits; the buffer is never bound or declared with useResource.
                NSString* source = @"#include <metal_stdlib>\n"
                                    "using namespace metal;\n"
                                    "kernel void probe(constant ulong& address [[buffer(0)]]) {\n"
                                    "    *((device uint*)address) = 42;\n"
                                    "}\n";
                error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
                if (!library) {
                    fprintf(stderr, "Shader compilation failed: %s\n", error.description.UTF8String);
                    return 1;
                }
                id<MTLComputePipelineState> pipeline =
                    [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"probe"] error:&error];
                if (!pipeline) {
                    fprintf(stderr, "Pipeline creation failed: %s\n", error.description.UTF8String);
                    return 1;
                }
                id<MTLCommandBuffer> commands = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
                uint64_t address = buffer.gpuAddress;
                printf("Buffer GPU address: 0x%llx\n", (unsigned long long)address);
                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&address length:sizeof(address) atIndex:0];
                [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                [encoder endEncoding];
                [commands commit];
                [commands waitUntilCompleted];
                printf("Command status: %lu\n", (unsigned long)commands.status);
                if (commands.error)
                    printf("Command error: %s\n", commands.error.description.UTF8String);
                uint32_t result = *(uint32_t*)buffer.contents;
                printf("Pointer write: %u (expected 42)\n", result);
                [queue removeResidencySet:residency];
                return commands.status == MTLCommandBufferStatusCompleted && result == 42 ? 0 : 1;
            } @catch (NSException* exception) {
                fprintf(stderr, "Metal exception: %s: %s\n", exception.name.UTF8String, exception.reason.UTF8String);
                return 1;
            }
        } else {
            printf("Residency sets require macOS 15 or newer\n");
            return 0;
        }
    }
}
