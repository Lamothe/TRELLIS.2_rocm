from typing import *
from PIL import Image
import rembg
import onnxruntime as ort

class BiRefNet:
    """
    Shim for BiRefNet that FORCES CPU execution.
    This is critical to prevent ONNX Runtime from initializing a ROCm context 
    that conflicts with PyTorch's native gfx1151 context.
    """
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        # STRICTLY force CPU. Do not allow ROCMExecutionProvider.
        providers = ['CPUExecutionProvider']
        
        print(f"[BiRefNet-Shim] Forcing ONNX Providers: {providers}")

        try:
            self.session = rembg.new_session(model_name="birefnet-general", providers=providers)
            print("[BiRefNet-Shim] Successfully loaded 'birefnet-general' on CPU.")
        except Exception as e:
            print(f"[BiRefNet-Shim] 'birefnet-general' failed ({e}). Fallback to 'u2net'.")
            self.session = rembg.new_session("u2net", providers=providers)

    def to(self, device: str):
        pass

    def cuda(self):
        pass

    def cpu(self):
        pass
        
    def __call__(self, image: Image.Image) -> Image.Image:
        # This now runs purely on CPU RAM, leaving the GPU free for Trellis
        return rembg.remove(image, session=self.session)