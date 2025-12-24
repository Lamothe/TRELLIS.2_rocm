
# 1. Deactivate any accidental environment
deactivate 2>/dev/null

# 2. Delete the "wrong" Python 3.14 venv
cd ~/Projects/TRELLIS.2_rocm
rm -rf .venv

# 3. Create a new venv EXPLICITLY using Python 3.12
# (Make sure you have python3.12 installed)
python3.12 -m venv .venv

# 4. Activate it
source .venv/bin/activate

# 5. Install PyTorch for ROCm (The Foundation)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

# 6. Install Dependencies
# (Installing easydict explicitly first just to be safe)
pip install easydict ninja packaging
pip install gradio wheel
pip install opencv-python-headless
pip install rembg
pip install onnxruntime
pip install transformers accelerate safetensors huggingface_hub
pip install open3d
pip install utils3d==0.1.0 --no-deps
pip install scipy numpy

# ---------------------------------------------------------
# 7. THE EXTENSION RE-COMPILE (Batch Job)
# We set the variables once and rebuild your custom kernels.
# ---------------------------------------------------------

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export TORCH_ROCM_ARCH_LIST="gfx1100"
export PYTORCH_ROCM_ARCH="gfx1100"
export MAX_JOBS=20  # Safety first!

# Rebuild FlexGEMM
echo "Building FlexGEMM..."
cd extensions/FlexGEMM_rocm
rm -rf build/ dist/ *.egg-info
pip install . --no-build-isolation

# Rebuild o_voxel
echo "Building o_voxel..."
cd ../o-voxel
rm -rf build/ dist/ *.egg-info
pip install . --no-build-isolation

# Rebuild nvdiffrast
echo "Building nvdiffrast..."
cd ../nvdiffrast_rocm
rm -rf build/ dist/ *.egg-info
pip install . --no-build-isolation

# Rebuild cumesh
echo "Building cumesh..."
cd ../cumesh
rm -rf build/ dist/ *.egg-info
pip install . --no-build-isolation

echo "Building flash-attention..."
cd ../flash-attention
rm -rf build/ dist/ *.egg-info
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install

# Return to root
cd ../../
