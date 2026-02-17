import argparse
import os
import sys

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Add Depth-Anything-V2 to the Python path so depth_anything_v2 can be imported
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEPTH_ANYTHING_ROOT = os.path.join(PROJECT_ROOT, "Depth-Anything-V2")
if DEPTH_ANYTHING_ROOT not in sys.path:
    sys.path.append(DEPTH_ANYTHING_ROOT)

from depth_anything_v2.dpt import DepthAnythingV2


def load_model(encoder: str, checkpoint_path: str, device: str) -> DepthAnythingV2:
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


def select_reference_bbox(img_rgb: np.ndarray):
    """Let the user draw a reference rectangle in matplotlib and return bbox coords."""
    h, w, _ = img_rgb.shape

    fig, ax = plt.subplots(1, 1)
    ax.imshow(img_rgb)
    ax.set_title("Draw reference object rectangle, then close the window")

    ref_coords = {"x0": None, "y0": None, "x1": None, "y1": None}

    def onselect(eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if x0 is None or y0 is None or x1 is None or y1 is None:
            return
        ref_coords["x0"] = int(round(min(x0, x1)))
        ref_coords["y0"] = int(round(min(y0, y1)))
        ref_coords["x1"] = int(round(max(x0, x1)))
        ref_coords["y1"] = int(round(max(y0, y1)))

    rect_selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],  # left click
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )

    plt.show()

    if ref_coords["x0"] is None:
        raise RuntimeError("No reference rectangle selected.")

    # Clip to image bounds
    x0 = max(0, min(w - 1, ref_coords["x0"]))
    y0 = max(0, min(h - 1, ref_coords["y0"]))
    x1 = max(0, min(w, ref_coords["x1"]))
    y1 = max(0, min(h, ref_coords["y1"]))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("Invalid reference rectangle after clipping.")

    return x0, y0, x1, y1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive Depth Anything test: select a reference object in the image, "
            "enter its real height and camera intrinsics, then click anywhere to read depth in meters."
        )
    )

    parser.add_argument("--img-path", type=str, required=True, help="Path to input image")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"], help="Depth Anything encoder scale")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(PROJECT_ROOT, "depth_anything_v2_vits.pth"), help="Path to Depth Anything checkpoint")

    # Optional CLI camera params; if not given, ask interactively
    parser.add_argument("--focal-mm", type=float, default=None, help="Camera focal length in millimeters (from EXIF)")
    parser.add_argument("--sensor-width-mm", type=float, default=None, help="Physical sensor width in millimeters")

    args = parser.parse_args()

    if not os.path.isfile(args.img_path):
        raise FileNotFoundError(f"Input image not found: {args.img_path}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    img_bgr = cv2.imread(args.img_path)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {args.img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 1) Let user draw reference rectangle
    print("Draw a rectangle around a reference object whose real height you know.")
    x0, y0, x1, y1 = select_reference_bbox(img_rgb)
    print(f"Selected reference bbox: x0={x0}, y0={y0}, x1={x1}, y1={y1}")

    # 2) Ask for real-world height and camera intrinsics (if not provided)
    if args.focal_mm is None:
        focal_mm = float(input("Enter camera focal length in mm (from EXIF): "))
    else:
        focal_mm = args.focal_mm

    if args.sensor_width_mm is None:
        sensor_width_mm = float(input("Enter camera sensor width in mm: "))
    else:
        sensor_width_mm = args.sensor_width_mm

    ref_height_m = float(input("Enter real-world height of reference object in meters: "))

    # 3) Run Depth Anything once
    model = load_model(args.encoder, args.checkpoint, device)
    depth_rel = model.infer_image(img_bgr)
    if depth_rel.shape != (h, w):
        depth_rel = cv2.resize(depth_rel, (w, h), interpolation=cv2.INTER_LINEAR)

    # 4) Compute metric scaling using the reference box
    ref_patch = depth_rel[y0:y1, x0:x1]
    ref_depth_rel_mean = float(ref_patch.mean())

    fx_pixels = compute_fx_pixels(focal_mm, sensor_width_mm, w)
    ref_height_px = float(y1 - y0)
    ref_distance_m = fx_pixels * ref_height_m / ref_height_px

    scale = ref_distance_m / ref_depth_rel_mean
    depth_m = depth_rel * scale

    print(f"Estimated distance to reference object: {ref_distance_m:.3f} m")
    print("Now click anywhere on the image window to read depth in meters. Close the window to exit.")

    # 5) Interactive click-to-read depth display
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img_rgb)
    ax.set_title("Click to query depth (meters). Close window to exit.")

    # Draw the reference bbox for context
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="lime",
            linewidth=2,
            label="reference",
        )
    )

    def onclick(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if x < 0 or x >= w or y < 0 or y >= h:
            return
        d = float(depth_m[y, x])
        print(f"Clicked at (x={x}, y={y}) -> depth ≈ {d:.3f} m")
        ax.plot(x, y, "ro")
        ax.text(
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


if __name__ == "__main__":
    main()
