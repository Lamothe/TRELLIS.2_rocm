export CC="$(pwd)/scripts/fixed_hipcc"
export CXX="$(pwd)/scripts/fixed_hipcc"
export CFLAGS="-D__HIP_PLATFORM_AMD__ -I$(pwd)/csrc/common"
export CXXFLAGS="-D__HIP_PLATFORM_AMD__ -I$(pwd)/csrc/common"
export TORCH_CUDA_ARCH_LIST="11.0"

export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
export CONV_BACKEND=spconv
export SPARSE_CONV_BACKEND=flex_gemm
export HIP_PLATFORM=amd
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export CUDA_HOME=/usr/lib64/rocm
export FLASH_ATTENTION_INTERNAL_CK_ENABLE="FALSE"
export TRITON_F32_AS_TF32=0
export HSA_ENABLE_SDMA=0

# --- Strix Halo Impersonation Mode (gfx1100) ---
# Impersonate 7900 XTX
export GPU_ARCHS="gfx1100"
export HSA_OVERRIDE_GFX_VERSION=11.0.0

echo "Starting TRELLIS (Impersonating gfx1100 on Strix Halo)..."
python example.py
