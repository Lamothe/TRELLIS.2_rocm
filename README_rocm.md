# TRELLIS.2 on ROCm
Tested on:
* AMD Strix Halo 128 GB (64/64) - Framework Desktop
* Fedora 43
* ROCm 6.4
* Python 3.12

You will need to request and be granted access to:
* https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
* https://huggingface.co/briaai/RMBG-2.0

# Install system dependencies
sudo dnf install hipblaslt-devel eigen3-devel libjpeg-turbo-devel rust cargo pkgconf openssl-devel mesa-libGL-devel mesa-libEGL-devel libglvnd-devel -y

# I did this but I don't think that we're using it right now.  Background removal is being done by the CPU.
sudo dnf install /usr/bin/execstack
execstack -c /home/michael/Projects/TRELLIS.2/.venv/lib64/python3.12/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.so

Use:
* setup_rocm.sh to install
* example_rocm.sh to run the example
