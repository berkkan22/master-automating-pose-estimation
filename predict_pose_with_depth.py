"""Predict keypoints with YOLO pose, estimate metric depth with Depth Anything,
 and save per-keypoint depths to TXT files.

Workflow per image:
- Run YOLO pose model to get detections + keypoints.
- Run Depth Anything V2 to get a relative depth map.
- Use a reference detection (first container) with known real-world height
  and camera intrinsics to convert relative depth to approximate meters.
- Sample depth (meters) at each keypoint location and write to TXT.
"""

import argparse
import os
import sys
from glob import glob
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add Depth-Anything-V2 to the Python path so depth_anything_v2 can be imported
DEPTH_ANYTHING_ROOT = os.path.join(PROJECT_ROOT, "Depth-Anything-V2")
if DEPTH_ANYTHING_ROOT not in sys.path:
    sys.path.append(DEPTH_ANYTHING_ROOT)

from depth_anything_v2.dpt import DepthAnythingV2


# Default paths (can be overridden by CLI)
DEFAULT_POSE_MODEL_PATH = os.path.join("runs", "pose", "train", "weights", "best.pt")
DEFAULT_DA_CHECKPOINT = os.path.join(PROJECT_ROOT, "depth_anything_v2_vits.pth")

# Target classes to keep; leave empty to keep all
TARGET_LABELS = {0}  # dataset only has one class (transportation_units)


def load_depth_anything(encoder: str, checkpoint_path: str, device: str) -> DepthAnythingV2:
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    depth_anything = depth_anything.to(device).eval()
    return depth_anything


def compute_fx_pixels(focal_length_mm: float, sensor_width_mm: float, image_width_px: int) -> float:
    """Approximate focal length in pixels from EXIF focal length and sensor width."""
    return (focal_length_mm / sensor_width_mm) * image_width_px


def prediction_to_struct(res, shape_hw: Tuple[int, int]) -> List[Dict]:
    """Convert YOLO result to a list with bbox + keypoints in image pixels."""
    H, W = shape_hw
    out: List[Dict] = []
    boxes = getattr(res, "boxes", None)
    kpts = getattr(res, "keypoints", None)
    if boxes is None or kpts is None:
        return out

    xywhn = boxes.xywhn.cpu().numpy()
    cls_arr = boxes.cls.cpu().numpy() if boxes.cls is not None else None
    kpt_xyn = kpts.xyn.cpu().numpy() if hasattr(kpts, "xyn") else None

    for i, (xc, yc, w, h) in enumerate(xywhn):
        cls_id = int(cls_arr[i]) if cls_arr is not None else -1
        if TARGET_LABELS and cls_id not in TARGET_LABELS:
            continue
        # bbox in pixel coordinates
        x1 = (xc - w / 2.0) * W
        y1 = (yc - h / 2.0) * H
        x2 = (xc + w / 2.0) * W
        y2 = (yc + h / 2.0) * H
        bbox = (float(x1), float(y1), float(x2), float(y2))

        keypoints = []  # (x, y, v)
        if kpt_xyn is not None:
            for (kx, ky) in kpt_xyn[i]:
                keypoints.append((float(kx * W), float(ky * H), 2.0))

        out.append({"cls": cls_id, "bbox": bbox, "keypoints": keypoints})

    return out


def sample_depth_at_keypoints(
    depth_m: np.ndarray, items: List[Dict]
) -> List[Dict]:
    """For each detection and keypoint, sample depth (meters) from depth map."""
    H, W = depth_m.shape
    out: List[Dict] = []
    for det_idx, it in enumerate(items):
        cls_id = it.get("cls", -1)
        kpts_info = []
        for k_idx, (x, y, v) in enumerate(it.get("keypoints", [])):
            if v <= 0:
                continue
            px = int(round(x))
            py = int(round(y))
            if px < 0 or px >= W or py < 0 or py >= H:
                continue
            d = float(depth_m[py, px])
            kpts_info.append({
                "kpt_index": k_idx,
                "x": float(x),
                "y": float(y),
                "depth_m": d,
            })
        if kpts_info:
            out.append({
                "det_index": det_idx,
                "cls": cls_id,
                "keypoints": kpts_info,
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict YOLO pose keypoints on a single image, estimate metric depth via "
            "Depth Anything, and save per-keypoint depths plus visualizations."
        ),
    )

    parser.add_argument("--img-path", type=str, required=True, help="Path to input image")
    parser.add_argument("--pose-model", type=str, default=DEFAULT_POSE_MODEL_PATH, help="Path to YOLO pose model")

    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"], help="Depth Anything encoder scale")
    parser.add_argument("--da-checkpoint", type=str, default=DEFAULT_DA_CHECKPOINT, help="Path to Depth Anything checkpoint")

    parser.add_argument("--ref-height-m", type=float, required=True, help="Real-world height of the reference object in meters")
    parser.add_argument("--focal-mm", type=float, required=True, help="Camera focal length in millimeters (from EXIF)")
    parser.add_argument("--sensor-width-mm", type=float, required=True, help="Physical sensor width in millimeters")

    parser.add_argument("--outdir", type=str, default=os.path.join(PROJECT_ROOT, "results", "pose_depth"), help="Output directory for TXT files")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    if not os.path.isfile(args.img_path):
        raise FileNotFoundError(f"Input image not found: {args.img_path}")

    os.makedirs(args.outdir, exist_ok=True)

    # Load models once
    pose_model = YOLO(args.pose_model)
    depth_model = load_depth_anything(args.encoder, args.da_checkpoint, device)

    img = cv2.imread(args.img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {args.img_path}")
    H, W = img.shape[:2]

    base = os.path.splitext(os.path.basename(args.img_path))[0]

    # 1) Pose prediction
    res = pose_model([args.img_path], verbose=False, conf=args.conf)[0]
    dets = prediction_to_struct(res, (H, W))
    if not dets:
        print(f"[{base}] No pose detections, skipping depth sampling.")
        return

    # 2) Depth prediction (relative)
    depth_rel = depth_model.infer_image(img)

    # 3) Use first detection as reference object for metric scaling
    ref = dets[0]
    x1, y1, x2, y2 = ref["bbox"]
    x0_i = max(0, int(round(x1)))
    y0_i = max(0, int(round(y1)))
    x1_i = min(W, int(round(x2)))
    y1_i = min(H, int(round(y2)))
    if x1_i <= x0_i or y1_i <= y0_i:
        print(f"[{base}] Invalid reference bbox, skipping.")
        return

    ref_patch = depth_rel[y0_i:y1_i, x0_i:x1_i]
    ref_depth_rel_mean = float(ref_patch.mean())

    fx_pixels = compute_fx_pixels(args.focal_mm, args.sensor_width_mm, W)
    ref_height_px = float(y1_i - y0_i)
    ref_distance_m = fx_pixels * args.ref_height_m / ref_height_px

    scale = ref_distance_m / ref_depth_rel_mean
    depth_m = depth_rel * scale

    # 4) Sample depth at keypoints
    kp_depths = sample_depth_at_keypoints(depth_m, dets)
    if not kp_depths:
        print(f"[{base}] No valid keypoints to sample depth for.")
        return

    # 5) Save per-image TXT (keypoint depths)
    out_txt = os.path.join(args.outdir, f"{base}_keypoint_depths.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# det_index cls kpt_index x y depth_m\n")
        for det in kp_depths:
            det_idx = det["det_index"]
            cls_id = det["cls"]
            for kp in det["keypoints"]:
                line = f"{det_idx} {cls_id} {kp['kpt_index']} {kp['x']:.2f} {kp['y']:.2f} {kp['depth_m']:.4f}\n"
                f.write(line)

    # 6) Create and save visualizations: pose-only, depth-only, and combined
    # Pose-only visualization on RGB
    pose_vis = img.copy()
    for det in dets:
        x1b, y1b, x2b, y2b = det["bbox"]
        cv2.rectangle(pose_vis, (int(x1b), int(y1b)), (int(x2b), int(y2b)), (0, 255, 0), 2)
        for kp in det["keypoints"]:
            kx, ky, _ = kp
            cv2.circle(pose_vis, (int(kx), int(ky)), 3, (0, 0, 255), -1)

    pose_path = os.path.join(args.outdir, f"{base}_pose.png")
    cv2.imwrite(pose_path, pose_vis)

    # Depth-only visualization from metric depth
    depth_norm = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    depth_color_resized = cv2.resize(depth_color, (W, H))

    depth_path = os.path.join(args.outdir, f"{base}_depth.png")
    cv2.imwrite(depth_path, depth_color_resized)

    # Combined visualization: overlay depth on RGB, then draw keypoints
    alpha = 0.6  # original weight
    beta = 0.4   # depth weight
    overlay = cv2.addWeighted(img, alpha, depth_color_resized, beta, 0)
    combo_vis = overlay.copy()
    for det in dets:
        x1b, y1b, x2b, y2b = det["bbox"]
        cv2.rectangle(combo_vis, (int(x1b), int(y1b)), (int(x2b), int(y2b)), (0, 255, 0), 2)
        for kp in det["keypoints"]:
            kx, ky, _ = kp
            cv2.circle(combo_vis, (int(kx), int(ky)), 3, (0, 0, 255), -1)

    combo_path = os.path.join(args.outdir, f"{base}_pose_depth_combined.png")
    cv2.imwrite(combo_path, combo_vis)

    print(f"[{base}] Saved keypoint depths to: {out_txt} (ref distance ~ {ref_distance_m:.3f} m)")
    print(f"[{base}] Saved pose visualization to: {pose_path}")
    print(f"[{base}] Saved depth visualization to: {depth_path}")
    print(f"[{base}] Saved combined visualization to: {combo_path}")


if __name__ == "__main__":
    main()
