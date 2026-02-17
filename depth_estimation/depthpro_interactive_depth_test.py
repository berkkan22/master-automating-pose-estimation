import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
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
) -> Tuple[np.ndarray, np.ndarray, float | None]:
    """Run Depth Pro and return (depth_m, inverse_depth_norm, f_px_used)."""
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
    f_px_used = None
    if prediction["focallength_px"] is not None:
        f_px_used = float(prediction["focallength_px"].detach().cpu().item())

    inverse_depth = 1.0 / np.clip(depth, 1e-4, 1e4)
    # Normalize inverse depth for visualization (similar to CLI example)
    max_inv = min(inverse_depth.max(), 1.0 / 0.1)
    min_inv = max(inverse_depth.min(), 1.0 / 250.0)
    inv_norm = (inverse_depth - min_inv) / (max_inv - min_inv + 1e-8)

    return depth, inv_norm, f_px_used


def interactive_view(image: np.ndarray, depth_m: np.ndarray, inv_norm: np.ndarray) -> None:
    """Show RGB + Depth Pro inverse depth and print depth on clicks."""
    h, w = image.shape[:2]

    fig, (ax_img, ax_depth) = plt.subplots(1, 2, figsize=(14, 6))

    ax_img.imshow(image)
    ax_img.set_title("RGB image (click to query depth)")

    im = ax_depth.imshow(inv_norm, cmap="turbo")
    ax_depth.set_title("Depth Pro inverse depth (normalized)")
    plt.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.04)

    print("Click on the RGB image to read depths. Close the window to exit.")
    print(
        "Depth stats (m): min={:.4f}, max={:.4f}, mean={:.4f}".format(
            float(depth_m.min()), float(depth_m.max()), float(depth_m.mean())
        )
    )

    def onclick(event):
        if event.inaxes is not ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if x < 0 or x >= w or y < 0 or y >= h:
            return

        d = float(depth_m[y, x])
        print(f"Clicked at (x={x}, y={y}) -> depth≈{d:.3f} m")
        ax_img.plot(x, y, "ro")
        ax_img.text(
            x + 5,
            y - 5,
            f"{d:.2f} m",
            color="yellow",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="black", alpha=0.5),
        )
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show()
    fig.canvas.mpl_disconnect(cid)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive Depth Pro test: load Depth Pro, "
            "predict metric depth for an image, and click to inspect depths."
        )
    )
    parser.add_argument("--img-path", type=str,
                        required=True, help="Path to input image")
    parser.add_argument(
        "--focal-px",
        type=float,
        default=None,
        help=(
            "Optional focal length in pixels. "
            "If not given, Depth Pro will use EXIF if available or estimate it."
        ),
    )
    parser.add_argument(
        "--half-precision",
        action="store_true",
        help="Run model in FP16 (may be faster on GPU, keep off on pure CPU).",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.img_path):
        raise FileNotFoundError(f"Input image not found: {args.img_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torch.half if args.half_precision and torch.cuda.is_available() else torch.float32

    print(f"Loading Depth Pro model on {device} (precision={precision}) ...")
    model, transform = load_depthpro_model(device, precision)

    print("Running Depth Pro inference ...")
    depth_m, inv_norm, f_px_used = run_depthpro(
        model,
        transform,
        args.img_path,
        device,
        focal_px_override=args.focal_px,
    )

    if f_px_used is not None:
        print(f"Focal length used (pixels): {f_px_used:.2f}")

    # Reload image purely for display (depth_pro.load_rgb already did, but we keep it simple)
    image, _, _ = depth_pro.load_rgb(args.img_path)

    interactive_view(image, depth_m, inv_norm)


if __name__ == "__main__":
    main()
