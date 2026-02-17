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
"""Interactive pipeline: segmentation → pose → Depth Pro → 3D keypoints.

Features:
- Runs YOLO segmentation and pose on a single image.
- Runs Depth Pro to get a metric depth map in meters.
- Computes 3D camera-space coordinates for every keypoint using
  the pinhole projection formula P = (K^{-1} p_2d) * depth.
- Saves all keypoints with 2D + 3D coordinates into a TXT file.
- Opens an interactive window where you can click on the image:
  prints the clicked 2D coordinate, its depth, and 3D coordinate.
"""

import argparse
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import depth_pro


BASE_PATH = "./runs"
POSE_MODEL_PATH = f"{BASE_PATH}/pose/train/weights/best.pt"
SEG_MODEL_PATH = f"{BASE_PATH}/segment/train/weights/best.pt"

OUT_DIR = "./results/pose_depth"
os.makedirs(OUT_DIR, exist_ok=True)

# Target classes to keep; leave empty to keep all
TARGET_LABELS = {0}


def load_depthpro_model(device: torch.device, precision: torch.dtype):
    model, transform = depth_pro.create_model_and_transforms(
        device=device,
        precision=precision,
    )
    model.eval()
    return model, transform


def run_depthpro(
    model,
    transform,
    img_path: str,
    device: torch.device,
    focal_px_override: float | None = None,
) -> Tuple[np.ndarray, float | None]:
    """Run Depth Pro and return (depth_m, f_px_used).

    depth_m has shape [H, W] in meters. f_px_used is the focal length
    in pixels actually used by the model (if available).
    """
    image, _, f_px_exif = depth_pro.load_rgb(img_path)

    if focal_px_override is not None:
        f_px = float(focal_px_override)
    else:
        f_px = f_px_exif

    img_tensor = transform(image)

    with torch.no_grad():
        prediction = model.infer(img_tensor, f_px=f_px)

    depth = prediction["depth"].detach().cpu().numpy().squeeze()

    if prediction["focallength_px"] is not None:
        f_px_used = float(prediction["focallength_px"].detach().cpu().item())
    else:
        f_px_used = f_px

    return depth, f_px_used


def prediction_to_struct(res, shape_hw: Tuple[int, int]) -> List[Dict]:
    """Convert YOLO pose result to a list of dicts with bbox + keypoints."""
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

        x1 = (xc - w / 2.0) * W
        y1 = (yc - h / 2.0) * H
        x2 = (xc + w / 2.0) * W
        y2 = (yc + h / 2.0) * H
        bbox = (float(x1), float(y1), float(x2), float(y2))

        keypoints = []
        if kpt_xyn is not None:
            for (kx, ky) in kpt_xyn[i]:
                keypoints.append((float(kx * W), float(ky * H), 2.0))

        out.append({"cls": cls_id, "bbox": bbox, "keypoints": keypoints})

    return out


def draw_pose_on_image(img: np.ndarray, dets: List[Dict]) -> np.ndarray:
    vis = img.copy()
    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        for idx, (kx, ky, _) in enumerate(det.get("keypoints", [])):
            cv2.circle(vis, (int(kx), int(ky)), 3, (0, 0, 255), -1)
            cv2.putText(
                vis,
                str(idx),
                (int(kx) + 3, int(ky) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return vis


def project_2d_to_3d(u: float, v: float, depth_m: float, k_inv: np.ndarray) -> np.ndarray:
    """Project 2D pixel + depth into 3D camera coordinates.

    p_2d = [u, v, 1]^T (homogeneous). The back-projected ray is r = K^{-1} p_2d.
    The 3D point is then P = r * depth (in meters).
    """
    p = np.array([u, v, 1.0], dtype=np.float64)
    ray = k_inv @ p
    return ray * float(depth_m)


def compute_all_keypoints_3d(
    depth_m: np.ndarray,
    dets: List[Dict],
    k_inv: np.ndarray,
) -> List[Dict]:
    H, W = depth_m.shape
    all_kp: List[Dict] = []
    for det_idx, det in enumerate(dets):
        cls_id = det.get("cls", -1)
        for k_idx, (x, y, v) in enumerate(det.get("keypoints", [])):
            if v <= 0:
                continue
            px = int(round(x))
            py = int(round(y))
            if px < 0 or px >= W or py < 0 or py >= H:
                continue
            d = float(depth_m[py, px])
            if not np.isfinite(d) or d <= 0:
                continue
            p3d = project_2d_to_3d(x, y, d, k_inv)
            all_kp.append(
                {
                    "det_index": det_idx,
                    "cls": cls_id,
                    "kpt_index": k_idx,
                    "u": float(x),
                    "v": float(y),
                    "depth_m": d,
                    "X": float(p3d[0]),
                    "Y": float(p3d[1]),
                    "Z": float(p3d[2]),
                }
            )
    return all_kp


def save_keypoints_txt(path: str, image_name: str, keypoints: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# image det_index cls kpt_index u v depth_m X Y Z\n")
        for kp in keypoints:
            line = (
                f"{image_name} {kp['det_index']} {kp['cls']} {kp['kpt_index']} "
                f"{kp['u']:.2f} {kp['v']:.2f} {kp['depth_m']:.4f} "
                f"{kp['X']:.4f} {kp['Y']:.4f} {kp['Z']:.4f}\n"
            )
            f.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO segmentation + pose, Depth Pro depth, and compute 3D "
            "coordinates for all keypoints. Also allows interactive clicking "
            "to query 2D→3D on the image."
        )
    )

    parser.add_argument("--img-path", type=str, required=True, help="Path to input image")
    parser.add_argument("--pose-model", type=str, default=POSE_MODEL_PATH, help="Path to YOLO pose model")
    parser.add_argument("--seg-model", type=str, default=SEG_MODEL_PATH, help="Path to YOLO segmentation model")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")

    parser.add_argument(
        "--focal-px",
        type=float,
        default=None,
        help=(
            "Optional focal length in pixels to force Depth Pro and the 3D "
            "projection. If omitted, Depth Pro will use EXIF/its estimate and "
            "we reuse that value for K."
        ),
    )

    parser.add_argument(
        "--half-precision",
        action="store_true",
        help="Run Depth Pro in FP16 (if supported by GPU).",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.img_path):
        raise FileNotFoundError(f"Input image not found: {args.img_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torch.half if args.half_precision and torch.cuda.is_available() else torch.float32

    # Load models
    pose_model = YOLO(args.pose_model)
    seg_model = YOLO(args.seg_model)
    depth_model, depth_transform = load_depthpro_model(device, precision)

    img_bgr = cv2.imread(args.img_path)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {args.img_path}")
    H, W = img_bgr.shape[:2]

    base_name = os.path.splitext(os.path.basename(args.img_path))[0]

    # 1) Segmentation
    seg_res = seg_model(args.img_path, conf=args.conf, verbose=False)[0]
    seg_vis = seg_res.plot()  # BGR image with masks/boxes drawn

    # 2) Pose
    pose_res = pose_model(args.img_path, conf=args.conf, verbose=False)[0]
    dets = prediction_to_struct(pose_res, (H, W))

    # Draw pose on top of segmentation visualization
    vis = draw_pose_on_image(seg_vis, dets)

    # 3) Depth Pro
    depth_m, f_px_used = run_depthpro(
        depth_model,
        depth_transform,
        args.img_path,
        device=device,
        focal_px_override=args.focal_px,
    )

    if depth_m.shape != (H, W):
        depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_LINEAR)

    if f_px_used is None:
        raise RuntimeError(
            "Depth Pro did not provide a focal length in pixels. "
            "Please pass --focal-px explicitly."
        )

    fx = fy = float(f_px_used)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    # 4) Compute 3D for all keypoints and save to TXT
    kp_3d = compute_all_keypoints_3d(depth_m, dets, K_inv)
    if kp_3d:
        out_txt = os.path.join(OUT_DIR, f"{base_name}_keypoints_3d.txt")
        save_keypoints_txt(out_txt, base_name, kp_3d)
        print(f"Saved keypoint 2D/3D coordinates to {out_txt}")
    else:
        print("No valid keypoints found for 3D computation.")

    # 5) Interactive clicking: print 2D + 3D for clicked pixel
    click_log_path = os.path.join(OUT_DIR, f"{base_name}_clicks_3d.txt")

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x < 0 or x >= W or y < 0 or y >= H:
            return

        d = float(depth_m[y, x])
        if not np.isfinite(d) or d <= 0:
            print(f"Clicked at (u={x}, v={y}) but depth is invalid: {d}")
            return

        p3d = project_2d_to_3d(float(x), float(y), d, K_inv)
        X, Y, Z = p3d.tolist()
        print(
            f"Clicked at (u={x}, v={y}) -> depth={d:.3f} m, "
            f"3D ≈ ({X:.3f}, {Y:.3f}, {Z:.3f}) m"
        )

        with open(click_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{base_name} {x} {y} {d:.4f} "
                f"{X:.4f} {Y:.4f} {Z:.4f}\n"
            )

        cv2.circle(vis, (x, y), 4, (255, 0, 0), -1)
        cv2.imshow("seg_pose_depth", vis)

    cv2.namedWindow("seg_pose_depth", cv2.WINDOW_NORMAL)
    cv2.imshow("seg_pose_depth", vis)
    cv2.setMouseCallback("seg_pose_depth", on_mouse)

    print("Left-click on the image to query 2D → 3D. Press 'q' to quit.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
