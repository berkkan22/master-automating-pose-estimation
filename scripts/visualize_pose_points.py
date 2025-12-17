"""Visualize pose.jsonl keypoints on the scene image.

Usage (from repo root):
    python scripts/visualize_pose_points.py \
        --scene-dir synthetic_data-v2/synthetic_data-v2/v1_default_0 \
        --out ./viz_pose

Notes:
- Expects rgb.jpg and pose.jsonl inside the scene directory.
- Draws numbered circles for each available image_position.
- Saves an annotated image; optionally show it in a window with --show.
"""

import argparse
import json
import os
import cv2
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, help="Scene folder containing rgb.jpg and pose.jsonl")
    ap.add_argument("--out", default="./viz_pose", help="Output directory for annotated image")
    ap.add_argument("--show", action="store_true", help="Display the image with OpenCV window")
    return ap.parse_args()


def load_pose(pose_path):
    objs = []
    with open(pose_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return objs


def main():
    args = parse_args()

    img_path = os.path.join(args.scene_dir, "rgb.jpg")
    pose_path = os.path.join(args.scene_dir, "pose.jsonl")

    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"rgb.jpg not found in {args.scene_dir}")
    if not os.path.isfile(pose_path):
        raise FileNotFoundError(f"pose.jsonl not found in {args.scene_dir}")

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    objs = load_pose(pose_path)

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_rgb)
    ax.axis('off')

    point_color = '#ffd500'  # yellow
    text_color = '#000000'   # black
    label_color = '#ff0000'  # red

    count = 0
    for obj in objs:
        label = obj.get("label", "")
        first_plotted = False
        for corner in obj.get("corners", []):
            pos = corner.get("image_position")
            if pos is None or len(pos) != 2:
                continue
            x, y = float(pos[0]), float(pos[1])
            ax.scatter(x, y, c=point_color, s=20, edgecolors='k', linewidths=0.5)
            ax.text(x + 3, y - 3, str(count), color=text_color, fontsize=7,
                    ha='left', va='bottom', backgroundcolor='white')
            if label and not first_plotted:
                ax.text(x, y - 12, label, color=label_color, fontsize=8,
                        ha='left', va='bottom', backgroundcolor='white')
                first_plotted = True
            count += 1

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, os.path.basename(args.scene_dir.rstrip('/')) + "_pose_overlay.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path} (points drawn: {count})")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()