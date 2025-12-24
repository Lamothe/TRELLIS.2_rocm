# --- Strix Halo Impersonation Mode (gfx1100) ---
export GPU_ARCHS="gfx1100"
export TORCH_CUDA_ARCH_LIST="11.0"

export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
export CONV_BACKEND=spconv
export SPARSE_CONV_BACKEND=flex_gemm
export HIP_PLATFORM=amd
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# CRITICAL: Impersonate 7900 XTX
export HSA_OVERRIDE_GFX_VERSION=11.0.0

echo "Starting TRELLIS (Impersonating gfx1100 on Strix Halo)..."
python example.py
