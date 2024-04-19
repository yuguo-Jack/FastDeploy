/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#ifdef WITH_DCU
#include <hip/hip_runtime.h>
#define GPU(str) hip##str
#define GPURAND(str) hiprand##str
#define GPUSOLVER(str) hipsolver##str
#define GPUFFT(str) hipfft##str
#define GPUMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define GPUMaxThreadsPerMultiProcessor hipDeviceAttributeMaxThreadsPerMultiProcessor
#define GPUMaxSharedMemoryPerBlockOptin hipDeviceAttributeSharedMemPerBlockOptin
#endif

#ifdef WITH_GPU
#include <cuda.h>
#include <cuda_runtime_api.h>
#define GPU(str) cuda##str
#define GPURAND(str) curand##str
#define GPUSOLVER(str) cusolver##str
#define GPUFFT(str) cufft##str
#define GPUMultiProcessorCount cudaDevAttrMultiProcessorCount
#define GPUMaxThreadsPerMultiProcessor cudaDevAttrMaxThreadsPerMultiProcessor
#define GPUMaxSharedMemoryPerBlockOptin cudaDevAttrMaxSharedMemoryPerBlockOptin
#endif
