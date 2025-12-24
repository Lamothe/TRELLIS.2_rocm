#include <hipsolver/hipsolver.h>

// 1. Map the types
#define cusolverStatus_t hipsolverStatus_t

// 2. Define a dummy error string function
inline const char* _compat_cusolverGetErrorMessage(hipsolverStatus_t status) { 
    return "HIP Solver Error (Compat)"; 
}

// 3. Macro-swap the name so PyTorch uses our function
#define cusolverGetErrorMessage _compat_cusolverGetErrorMessage

