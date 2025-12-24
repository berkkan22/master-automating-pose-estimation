import cv2
import torch
import os
import sys
import numpy as np

# Add Depth-Anything-V2 to the Python path so depth_anything_v2 can be imported
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEPTH_ANYTHING_ROOT = os.path.join(PROJECT_ROOT, 'Depth-Anything-V2')
if DEPTH_ANYTHING_ROOT not in sys.path:
    sys.path.append(DEPTH_ANYTHING_ROOT)

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits'  # or 'vitb', 'vitl', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load('/data/9katirci/master-automating-pose-estimation/depth_anything_v2_vits.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

img_path = '/data/9katirci/master-automating-pose-estimation/datasets/trudi_ds_yolo11_instand_segmentation/test/images/20240503_124230.jpg'
# img_path = '/data/9katirci/master-automating-pose-estimation/datasets/synthetic_data-v2-coco/test/images/0.jpg'
raw_img = cv2.imread(img_path)

# Run depth inference (HxW raw depth map in numpy)
depth = model.infer_image(raw_img)

# Normalize depth to 0-255 and convert to uint8
depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_uint8 = depth_norm.astype(np.uint8)

# Apply a colormap for visualization
depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

# Build output path under results/depth_anything
outdir = os.path.join(PROJECT_ROOT, 'results', 'depth_anything')
os.makedirs(outdir, exist_ok=True)
basename = os.path.splitext(os.path.basename(img_path))[0]

# Ensure depth_color matches the original image size
depth_color_resized = cv2.resize(depth_color, (raw_img.shape[1], raw_img.shape[0]))

# 1) Save pure depth visualization
depth_path = os.path.join(outdir, f'{basename}_depth.png')
cv2.imwrite(depth_path, depth_color_resized)

# 2) Save side-by-side original and depth
side_by_side = cv2.hconcat([raw_img, depth_color_resized])
side_by_side_path = os.path.join(outdir, f'{basename}_depth_side_by_side.png')
cv2.imwrite(side_by_side_path, side_by_side)

# 3) Save overlay of depth on top of original with opacity
alpha = 0.6  # weight for original image
beta = 0.4   # weight for depth map
overlay = cv2.addWeighted(raw_img, alpha, depth_color_resized, beta, 0)
overlay_path = os.path.join(outdir, f'{basename}_depth_overlay.png')
cv2.imwrite(overlay_path, overlay)

print(f'Depth map saved to: {depth_path}')
print(f'Side-by-side image saved to: {side_by_side_path}')
print(f'Overlay image saved to: {overlay_path}')