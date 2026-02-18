"""Pose + Depth Pro: compute 3D for all detected keypoints.

This script:
- Runs a YOLO pose model on an image to get keypoints.
- Runs Depth Pro to obtain a metric depth map (meters).
- Uses a simple pinhole camera model with focal length in pixels
  to back-project every keypoint into 3D camera coordinates.
- Saves all keypoints with 2D+3D coordinates into a TXT file.
- Saves an image with each keypoint drawn and labeled with
  its 2D and 3D coordinates.

The Depth Pro usage is based on depth_estimation/depthpro_interactive_depth_test.py.
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

OUT_DIR = "./results/pose_depth_all"
os.makedirs(OUT_DIR, exist_ok=True)

# Target classes to keep; leave empty to keep all
TARGET_LABELS = {0}


def load_depthpro_model(device: torch.device, precision: torch.dtype):
    """Load Depth Pro model and its preprocessing transform."""
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


def draw_and_label_keypoints(
    img_bgr: np.ndarray,
    keypoints_3d: List[Dict],
) -> np.ndarray:
    """Draw keypoints and label each with 2D and 3D coordinates."""
    vis = img_bgr.copy()
    for kp in keypoints_3d:
        u = int(round(kp["u"]))
        v = int(round(kp["v"]))
        text = (
            f"id {kp['kpt_index']}: u={kp['u']:.1f}, v={kp['v']:.1f} "
            f"X={kp['X']:.2f}, Y={kp['Y']:.2f}, Z={kp['Z']:.2f}"
        )
        cv2.circle(vis, (u, v), 3, (0, 0, 255), -1)
        cv2.putText(
            vis,
            text,
            (u + 5, v - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO pose + Depth Pro and compute 3D coordinates for "
            "all detected keypoints. Saves TXT and annotated image."
        )
    )

    parser.add_argument("--img-path", type=str,
                        required=True, help="Path to input image")
    parser.add_argument(
        "--pose-model",
        type=str,
        default=POSE_MODEL_PATH,
        help="Path to YOLO pose model (e.g. runs/pose/train/weights/best.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.5,
                        help="YOLO confidence threshold")

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

    parser.add_argument(
        "--out-dir",
        type=str,
        default=OUT_DIR,
        help="Output directory for TXT and annotated image",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.img_path):
        raise FileNotFoundError(f"Input image not found: {args.img_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torch.half if args.half_precision and torch.cuda.is_available() else torch.float32

    # Load models
    pose_model = YOLO(args.pose_model)
    depth_model, depth_transform = load_depthpro_model(device, precision)

    # Pose estimation
    pose_res = pose_model(args.img_path, conf=args.conf, verbose=False)[0]

    # Use Depth Pro's RGB loader so depth map and RGB share the same resolution
    rgb_image, _, _ = depth_pro.load_rgb(args.img_path)
    H, W = rgb_image.shape[:2]

    dets = prediction_to_struct(pose_res, (H, W))

    # Depth estimation
    depth_m, f_px_used = run_depthpro(
        depth_model,
        depth_transform,
        args.img_path,
        device=device,
        focal_px_override=args.focal_px,
    )

    if depth_m.shape != (H, W):
        depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_LINEAR)

    # Intrinsic matrix K (simple pinhole model)
    if args.focal_px is not None:
        fx = fy = float(args.focal_px)
    elif f_px_used is not None:
        fx = fy = float(f_px_used)
    else:
        raise RuntimeError(
            "Depth Pro did not provide a focal length in pixels. "
            "Please pass --focal-px explicitly."
        )

    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [
                 0.0, 0.0, 1.0]], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    # Compute 3D for all keypoints and save to TXT
    base_name = os.path.splitext(os.path.basename(args.img_path))[0]
    kp_3d = compute_all_keypoints_3d(depth_m, dets, K_inv)

    if kp_3d:
        txt_path = os.path.join(args.out_dir, f"{base_name}_keypoints_3d.txt")
        save_keypoints_txt(txt_path, base_name, kp_3d)
        print(f"Saved keypoint 2D/3D coordinates to {txt_path}")
    else:
        print("No valid keypoints found for 3D computation.")

    # Annotated image with keypoints and labels
    rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    vis = draw_and_label_keypoints(rgb_bgr, kp_3d)
    img_out_path = os.path.join(
        args.out_dir, f"{base_name}_pose_depth_annotated.png")
    cv2.imwrite(img_out_path, vis)
    print(f"Saved annotated image to {img_out_path}")


if __name__ == "__main__":
    main()
