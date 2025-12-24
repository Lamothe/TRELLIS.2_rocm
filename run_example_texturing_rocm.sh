export HIP_PLATFORM=amd
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export CUDA_HOME=/usr/lib64/rocm
export FLASH_ATTENTION_INTERNAL_CK_ENABLE="FALSE"
export TRITON_F32_AS_TF32=0
export HSA_ENABLE_SDMA=0
export MIOPEN_DEBUG_GCN_ASM_KERNELS=0  # Disable buggy assembly kernels
export MIOPEN_DEBUG_DISABLE_FIND_DB=1 # Prevent DB corruption crashes
export MIOPEN_FIND_MODE=1             # Force generic kernel selection
export AMD_SERIALIZE_KERNEL=1         # Force sequential kernel execution (debug only)
export TORCH_CUDA_ARCH_LIST="11.0"
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:False"

# --- Strix Halo Impersonation Mode (gfx1100) ---
# Impersonate 7900 XTX
export GPU_ARCHS="gfx1100"
export TORCH_ROCM_ARCH_LIST="gfx1100"
export HSA_OVERRIDE_GFX_VERSION=11.0.0

echo "Starting TRELLIS..."
python -X faulthandler example_texturing.py
