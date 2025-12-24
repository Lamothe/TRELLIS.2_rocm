# Launch the web app
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Hardware Stability
export HSA_ENABLE_SDMA=0

# MIOpen "Safety Mode" (CRITICAL: Disable Assembly Kernels)
# This forces generic kernels that are compatible with Strix Halo
export MIOPEN_DEBUG_GCN_ASM_KERNELS=0
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export MIOPEN_FIND_MODE=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export TRITON_F32_AS_TF32=0

# 4. PyTorch Allocator
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

# Launch
python -X faulthandler app.py
