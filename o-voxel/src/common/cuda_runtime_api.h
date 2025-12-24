#ifndef CUDA_RUNTIME_SHIM_H
#define CUDA_RUNTIME_SHIM_H

#include <hip/hip_runtime.h>

// --- Types ---
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaMemcpyKind hipMemcpyKind

// --- Constants ---
#define cudaSuccess hipSuccess
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaEventDisableTiming hipEventDisableTiming

// --- Functions ---
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy hipMemcpy
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamQuery hipStreamQuery
#define cudaStreamGetPriority hipStreamGetPriority
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventRecord hipEventRecord
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime

#endif
#ifndef cudaErrorNotReady
#define cudaErrorNotReady hipErrorNotReady
#endif

// --- Stream Capture Support ---
#define cudaStreamCaptureMode hipStreamCaptureMode
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed


// --- Stream Capture Functions ---
#define cudaThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode


// --- Stream Capture Status ---
#define cudaStreamCaptureStatus hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusInvalid hipStreamCaptureStatusInvalid

#define cudaStreamCaptureStatusInvalidated hipStreamCaptureStatusInvalidated

// --- Missing Graph & Event Aliases ---
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaEventDefault hipEventDefault


// --- Event Synchronization ---
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaEventQuery hipEventQuery

#define cudaEventSynchronize hipEventSynchronize

// --- Fixes for PyTorch 2.9+ ROCm Headers ---

// 1. Map NVIDIA Solver Handle to HIP
#define cusolverDnHandle_t hipsolverDnHandle_t

// 2. Fix 'current_device' visibility in CUDAUtils.h
// The header uses unqualified 'current_device()', but on ROCm it lives in c10::cuda
