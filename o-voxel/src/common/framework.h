#ifndef NVDR_FRAMEWORK_H
#define NVDR_FRAMEWORK_H

// 1. SHIM & MACROS
#include "cuda_runtime.h"

// 2. TORCH HEADERS
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>

// 3. ROCm CONTEXT
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>

// 4. MANUAL IMPLEMENTATIONS
namespace at {
    namespace cuda {
        static inline int current_device() {
            int dev = 0;
            (void)hipGetDevice(&dev);
            return dev;
        }

        static inline bool check_device(at::ArrayRef<at::Tensor> tensors) {
            if (tensors.empty()) return true;
            auto device = tensors[0].device();
            for (const auto& t : tensors) {
                if (t.device() != device) return false;
            }
            return true;
        }
    }
}

// 5. HIPIFY COMPATIBILITY WRAPPERS
namespace c10 {
    namespace hip {
        struct OptionalHIPGuardMasqueradingAsCUDA : public c10::OptionalDeviceGuard {
            using c10::OptionalDeviceGuard::OptionalDeviceGuard;
            explicit OptionalHIPGuardMasqueradingAsCUDA(int index) 
                : c10::OptionalDeviceGuard(c10::Device(c10::DeviceType::CUDA, index)) {}
        };
        
        static inline auto getCurrentHIPStreamMasqueradingAsCUDA() {
            return c10::hip::getCurrentHIPStream();
        }
    }
}

namespace at {
    namespace hip {
        using OptionalHIPGuardMasqueradingAsCUDA = c10::hip::OptionalHIPGuardMasqueradingAsCUDA;
        
        static inline auto getCurrentHIPStreamMasqueradingAsCUDA() {
            return c10::hip::getCurrentHIPStream();
        }
    }
}

// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
// ... (Standard License Header) ...

#ifdef NVDR_TORCH
    #ifndef __HIPCC__
        #if !defined(USE_ROCM) && !defined(__HIP_PLATFORM_AMD__)
            #include <ATen/cuda/CUDAContext.h>
            #include <ATen/cuda/CUDAUtils.h>
            #include <c10/cuda/CUDAGuard.h>
        #endif
        #include <pybind11/numpy.h>
    #endif

    #define NVDR_CHECK(COND, ERR) do { TORCH_CHECK(COND, ERR) } while(0)
    #define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) do { hipError_t err = CUDA_CALL; TORCH_CHECK(!err, "Cuda error: ", hipGetLastError(), "[", #CUDA_CALL, ";]"); } while(0)
#endif

#endif // NVDR_FRAMEWORK_H
