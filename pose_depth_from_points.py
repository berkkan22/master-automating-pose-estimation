"""Depth Pro 3D query for user-specified 2D keypoints.

This script:
- Runs Depth Pro on an image to obtain a metric depth map (meters).
- Takes 2D pixel coordinates provided via CLI.
- Uses a pinhole camera model (with focal length in pixels) to
  back-project each 2D keypoint into 3D camera coordinates.
- Saves all queried keypoints with 2D+3D coordinates into a TXT file.
- Saves an image with each queried keypoint drawn and labeled with
  its 2D and 3D coordinates.

Keypoints are passed as a single string, for example:
    --keypoints "100,200;150,250;320,400"
"""

import argparse
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

import depth_pro


OUT_DIR = "./results/pose_depth_points"
os.makedirs(OUT_DIR, exist_ok=True)


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
    """Run Depth Pro and return (depth_m, f_px_used)."""
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


def project_2d_to_3d(u: float, v: float, depth_m: float, k_inv: np.ndarray) -> np.ndarray:
    """Project 2D pixel + depth into 3D camera coordinates."""
    p = np.array([u, v, 1.0], dtype=np.float64)
    ray = k_inv @ p
    return ray * float(depth_m)


def parse_keypoints(spec: str) -> List[Tuple[float, float]]:
    """Parse keypoints from "x,y;x,y;..." string."""
    points: List[Tuple[float, float]] = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        xy = part.split(",")
        if len(xy) != 2:
            raise ValueError(
                f"Invalid keypoint format: '{part}' (expected x,y)")
        x, y = float(xy[0]), float(xy[1])
        points.append((x, y))
    return points


def save_keypoints_txt(path: str, image_name: str, keypoints: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# image idx u v depth_m X Y Z\n")
        for idx, kp in enumerate(keypoints):
            line = (
                f"{image_name} {idx} {kp['u']:.2f} {kp['v']:.2f} {kp['depth_m']:.4f} "
                f"{kp['X']:.4f} {kp['Y']:.4f} {kp['Z']:.4f}\n"
            )
            f.write(line)


def draw_and_label_keypoints(
    img_bgr: np.ndarray,
    keypoints_3d: List[Dict],
) -> np.ndarray:
    vis = img_bgr.copy()
    for idx, kp in enumerate(keypoints_3d):
        u = int(round(kp["u"]))
        v = int(round(kp["v"]))
        text = (
            f"id {idx}: u={kp['u']:.1f}, v={kp['v']:.1f} "
            f"X={kp['X']:.2f}, Y={kp['Y']:.2f}, Z={kp['Z']:.2f}"
        )
        cv2.circle(vis, (u, v), 4, (255, 0, 0), -1)
        cv2.putText(
            vis,
            text,
            (u + 5, v - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Depth Pro on an image and compute 3D coordinates for "
            "user-specified 2D keypoints. Saves TXT and annotated image."
        )
    )

    parser.add_argument("--img-path", type=str,
                        required=True, help="Path to input image")
    parser.add_argument(
        "--keypoints",
        type=str,
        required=True,
        help="Keypoints as 'x,y;x,y;...' in pixel coordinates",
    )

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

    keypoints_2d = parse_keypoints(args.keypoints)

    # Load Depth Pro
    depth_model, depth_transform = load_depthpro_model(device, precision)
    depth_m, f_px_used = run_depthpro(
        depth_model,
        depth_transform,
        args.img_path,
        device=device,
        focal_px_override=args.focal_px,
    )

    # Use Depth Pro's RGB loader for resolution
    rgb_image, _, _ = depth_pro.load_rgb(args.img_path)
    H, W = rgb_image.shape[:2]

    if depth_m.shape != (H, W):
        depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_LINEAR)

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

    base_name = os.path.splitext(os.path.basename(args.img_path))[0]

    keypoints_3d: List[Dict] = []
    for (u, v) in keypoints_2d:
        px = int(round(u))
        py = int(round(v))
        if px < 0 or px >= W or py < 0 or py >= H:
            continue
        d = float(depth_m[py, px])
        if not np.isfinite(d) or d <= 0:
            continue
        p3d = project_2d_to_3d(u, v, d, K_inv)
        keypoints_3d.append(
            {
                "u": float(u),
                "v": float(v),
                "depth_m": d,
                "X": float(p3d[0]),
                "Y": float(p3d[1]),
                "Z": float(p3d[2]),
            }
        )

    if keypoints_3d:
        txt_path = os.path.join(
            args.out_dir, f"{base_name}_manual_keypoints_3d.txt")
        save_keypoints_txt(txt_path, base_name, keypoints_3d)
        print(f"Saved keypoint 2D/3D coordinates to {txt_path}")
    else:
        print("No valid keypoints found (inside image with valid depth).")

    rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    vis = draw_and_label_keypoints(rgb_bgr, keypoints_3d)
    img_out_path = os.path.join(
        args.out_dir, f"{base_name}_manual_keypoints_3d.png")
    cv2.imwrite(img_out_path, vis)
    print(f"Saved annotated image to {img_out_path}")


if __name__ == "__main__":
    main()
