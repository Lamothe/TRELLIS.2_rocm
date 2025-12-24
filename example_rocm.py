import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# Setup Environment Map
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# Load Pipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# Load Image & Run
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image)[0]
mesh.simplify(200000) # nvdiffrast limit

print("Saving geometry-only backup to 'safety_mesh.obj'...")
with open("safety_mesh.obj", "w") as f:
    verts = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    for v in verts:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for face in faces:
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
print("Backup saved!")

# Render Video
#video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
#imageio.mimsave("sample.mp4", video, fps=15)

# 5. Export to GLB
print("Attempting minimal GLB export...")
glb = o_voxel.postprocess.to_glb(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    
    # CRITICAL CHANGE 1: Drop face count to 100k
    decimation_target   =   100000,
    
    # CRITICAL CHANGE 2: Drop texture to 1024 (or even 512 if this fails)
    texture_size        =   1024,
    
    # CRITICAL CHANGE 3: Disable remeshing (Uses original geometry, much safer)
    remesh              =   False,
    # remesh_band       =   1,  <-- Comment out or ignore
    # remesh_project    =   0,  <-- Comment out or ignore
    
    verbose             =   True
)
glb.export("sample.glb", extension_webp=True)