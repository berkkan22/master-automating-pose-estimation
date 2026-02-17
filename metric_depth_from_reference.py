import argparse
import os
import sys

import cv2
import numpy as np
import torch

# Add Depth-Anything-V2 to the Python path so depth_anything_v2 can be imported
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEPTH_ANYTHING_ROOT = os.path.join(PROJECT_ROOT, 'Depth-Anything-V2')
if DEPTH_ANYTHING_ROOT not in sys.path:
    sys.path.append(DEPTH_ANYTHING_ROOT)

from depth_anything_v2.dpt import DepthAnythingV2


def load_model(encoder: str, checkpoint_path: str, device: str) -> DepthAnythingV2:
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }

    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    depth_anything = depth_anything.to(device).eval()
    return depth_anything


def compute_fx_pixels(focal_length_mm: float, sensor_width_mm: float, image_width_px: int) -> float:
    """Approximate focal length in pixels from EXIF focal length and sensor width."""
    return (focal_length_mm / sensor_width_mm) * image_width_px


def main() -> None:
    parser = argparse.ArgumentParser(description="Approximate metric depth from Depth Anything using a reference object")

    parser.add_argument("--img-path", type=str, required=True, help="Path to input image")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"], help="Depth Anything encoder scale")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(PROJECT_ROOT, "depth_anything_v2_vits.pth"), help="Path to Depth Anything checkpoint")

    parser.add_argument("--ref-xmin", type=int, required=True, help="Left pixel coordinate of reference object bbox")
    parser.add_argument("--ref-ymin", type=int, required=True, help="Top pixel coordinate of reference object bbox")
    parser.add_argument("--ref-xmax", type=int, required=True, help="Right pixel coordinate of reference object bbox")
    parser.add_argument("--ref-ymax", type=int, required=True, help="Bottom pixel coordinate of reference object bbox")

    parser.add_argument("--ref-height-m", type=float, required=True, help="Real-world height of the reference object in meters")

    parser.add_argument("--focal-mm", type=float, required=True, help="Camera focal length in millimeters (from EXIF)")
    parser.add_argument("--sensor-width-mm", type=float, required=True, help="Physical sensor width in millimeters")

    parser.add_argument("--outdir", type=str, default=os.path.join(PROJECT_ROOT, "results", "metric_depth"), help="Output directory")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    os.makedirs(args.outdir, exist_ok=True)

    img = cv2.imread(args.img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {args.img_path}")

    h, w = img.shape[:2]

    model = load_model(args.encoder, args.checkpoint, device)

    # Run depth inference (relative depth, arbitrary scale)
    depth_rel = model.infer_image(img)

    # Extract relative depth for the reference object region
    x0, y0, x1, y1 = args.ref_xmin, args.ref_ymin, args.ref_xmax, args.ref_ymax
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid reference bounding box after clipping to image bounds")

    ref_depth_patch = depth_rel[y0:y1, x0:x1]
    ref_depth_rel_mean = float(ref_depth_patch.mean())

    # Estimate camera focal length in pixels
    fx_pixels = compute_fx_pixels(args.focal_mm, args.sensor_width_mm, w)

    # Pixel height of the reference object in the image
    ref_height_px = float(y1 - y0)

    # Pinhole approximation: Z = f * H / h
    ref_distance_m = fx_pixels * args.ref_height_m / ref_height_px

    # Scale relative depth so that the mean depth in the reference region matches this distance
    scale = ref_distance_m / ref_depth_rel_mean
    depth_m = depth_rel * scale

    basename = os.path.splitext(os.path.basename(args.img_path))[0]

    # Save raw metric depth as .npy (meters)
    depth_npy_path = os.path.join(args.outdir, f"{basename}_depth_meters.npy")
    np.save(depth_npy_path, depth_m.astype(np.float32))

    # Also save metric depth as a human-readable text file (meters)
    depth_txt_path = os.path.join(args.outdir, f"{basename}_depth_meters.txt")
    np.savetxt(depth_txt_path, depth_m.astype(np.float32), fmt="%.4f")

    # Create visualizations from metric depth (for display only; still relative contrast)
    depth_norm = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    depth_color_resized = cv2.resize(depth_color, (w, h))

    depth_vis_path = os.path.join(args.outdir, f"{basename}_depth_vis.png")
    cv2.imwrite(depth_vis_path, depth_color_resized)

    side_by_side = cv2.hconcat([img, depth_color_resized])
    side_by_side_path = os.path.join(args.outdir, f"{basename}_depth_side_by_side.png")
    cv2.imwrite(side_by_side_path, side_by_side)

    alpha = 0.6
    beta = 0.4
    overlay = cv2.addWeighted(img, alpha, depth_color_resized, beta, 0)
    overlay_path = os.path.join(args.outdir, f"{basename}_depth_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    print(f"Saved metric depth array (meters) to: {depth_npy_path}")
    print(f"Saved metric depth TXT (meters) to: {depth_txt_path}")
    print(f"Saved depth visualization to: {depth_vis_path}")
    print(f"Saved side-by-side visualization to: {side_by_side_path}")
    print(f"Saved overlay visualization to: {overlay_path}")
    print(f"Estimated distance to reference object: {ref_distance_m:.3f} m")


if __name__ == "__main__":
    main()
