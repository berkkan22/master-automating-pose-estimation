import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import numpy as np
import torch

import depth_pro


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
    # Load image and EXIF focal length (if available)
    image, _, f_px_exif = depth_pro.load_rgb(img_path)

    if focal_px_override is not None:
        f_px = float(focal_px_override)
    else:
        f_px = f_px_exif

    # Prepare tensor
    img_tensor = transform(image)  # already moves to device inside transform

    # Inference; Depth Pro returns metric depth in meters
    with torch.no_grad():
        prediction = model.infer(img_tensor, f_px=f_px)

    depth = prediction["depth"].detach().cpu(
    ).numpy().squeeze()  # [H,W] in meters

    if prediction["focallength_px"] is not None:
        f_px_used = float(prediction["focallength_px"].detach().cpu().item())
    else:
        f_px_used = f_px

    return depth, f_px_used


def depth_to_pointcloud(
    depth_m: np.ndarray,
    f_px: float,
    stride: int = 1,
    max_depth: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert depth map (in meters) to 3D point cloud using pinhole camera model.

    Coordinates are in camera space with Z pointing forward.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")

    h, w = depth_m.shape

    # Pixel coordinates
    v_indices, u_indices = np.indices((h, w))  # v = row (y), u = col (x)
    v_indices = v_indices[::stride, ::stride]
    u_indices = u_indices[::stride, ::stride]

    z = depth_m[::stride, ::stride]

    # Valid depth mask
    mask = np.isfinite(z) & (z > 0)
    if max_depth is not None:
        mask &= z <= max_depth

    if not np.any(mask):
        raise ValueError("No valid depth values after masking.")

    u = u_indices[mask].astype(np.float32)
    v = v_indices[mask].astype(np.float32)
    z = z[mask].astype(np.float32)

    # Intrinsics (assume principal point at image center, square pixels)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    fx = f_px
    fy = f_px

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return x, y, z


def plot_pointcloud(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    img_path: str,
    stride: int,
) -> None:
    """Interactive 3D scatter plot of the point cloud in Matplotlib."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x,
        y,
        z,
        c=z,
        s=1,
        cmap="turbo",
        marker=".",
        linewidth=0,
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(
        f"Depth Pro point cloud\n{os.path.basename(img_path)} (stride={stride})"
    )

    # Try to make aspect roughly equal
    try:
        max_range = np.array([
            x.max() - x.min(),
            y.max() - y.min(),
            z.max() - z.min(),
        ]).max()
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    except Exception:
        pass

    fig.colorbar(sc, ax=ax, label="Depth Z [m]")

    print("Rotate the view with mouse drag; scroll to zoom.")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Depth Pro 3D point cloud viewer: predict metric depth and "
            "visualize a 3D point cloud in Matplotlib."
        )
    )
    parser.add_argument(
        "--img-path",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--focal-px",
        type=float,
        default=None,
        help=(
            "Optional focal length in pixels. If not given, Depth Pro uses EXIF "
            "if available or estimates it."
        ),
    )
    parser.add_argument(
        "--half-precision",
        action="store_true",
        help="Run model in FP16 (may be faster on GPU, keep off on CPU).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "Subsample factor for pixels when building the point cloud. "
            "1 = every pixel, 2 = every 2nd pixel, etc."
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Optional maximum depth in meters to keep in the point cloud.",
    )
    parser.add_argument(
        "--save-npz",
        type=str,
        default=None,
        help=(
            "If set, save point cloud to this .npz file "
            "(arrays x, y, z, plus metadata)."
        ),
    )

    args = parser.parse_args()

    if not os.path.isfile(args.img_path):
        raise FileNotFoundError(f"Input image not found: {args.img_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = (
        torch.half if args.half_precision and torch.cuda.is_available() else torch.float32
    )

    print(f"Loading Depth Pro model on {device} (precision={precision}) ...")
    model, transform = load_depthpro_model(device, precision)

    print("Running Depth Pro inference ...")
    depth_m, f_px_used = run_depthpro(
        model,
        transform,
        args.img_path,
        device,
        focal_px_override=args.focal_px,
    )

    if f_px_used is None:
        raise RuntimeError(
            "Depth Pro did not return a focal length; please provide --focal-px."
        )

    print(f"Focal length used (pixels): {f_px_used:.2f}")
    print(
        "Depth stats (m): min={:.4f}, max={:.4f}, mean={:.4f}".format(
            float(depth_m.min()), float(depth_m.max()), float(depth_m.mean())
        )
    )

    print("Converting depth map to 3D point cloud ...")
    x, y, z = depth_to_pointcloud(
        depth_m,
        f_px=f_px_used,
        stride=args.stride,
        max_depth=args.max_depth,
    )
    print(f"Point cloud size: {x.size} points (stride={args.stride})")

    if args.save_npz is not None:
        np.savez(
            args.save_npz,
            x=x,
            y=y,
            z=z,
            f_px=f_px_used,
            img_path=args.img_path,
            stride=args.stride,
            max_depth=args.max_depth,
        )
        print(f"Saved point cloud to {args.save_npz}")

    plot_pointcloud(x, y, z, img_path=args.img_path, stride=args.stride)


if __name__ == "__main__":
    main()
