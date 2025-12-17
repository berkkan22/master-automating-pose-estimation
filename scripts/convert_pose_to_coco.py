"""Convert synthetic pose.jsonl scenes to COCO and deterministic splits.

Usage (from repo root):
    python scripts/convert_pose_to_coco.py \
        --src-root synthetic_data-v2/synthetic_data-v2 \
        --out-root datasets/synth_pose_coco \
        --seed 42 --train 0.8 --val 0.1 --test 0.1

Behavior:
- Scans scene folders under --src-root (expects rgb.jpg and pose.jsonl).
- Converts pose.jsonl entries to COCO annotations (bbox + rect segmentation).
- Splits images into train/val/test deterministically with the given seed.
- Persists split manifest to reuse the same split on reruns (split_manifest.json).
- Copies images into out_root/images/{split}/ and writes COCO JSON to out_root/annotations/.
"""

import argparse
import json
import os
import random
import shutil
from glob import glob
from typing import Dict, List, Tuple

import cv2
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True, help="Root containing v1_default_* scene folders")
    ap.add_argument("--out-root", required=True, help="Output root for COCO dataset")
    ap.add_argument("--seed", type=int, default=42, help="Seed for deterministic split")
    ap.add_argument("--train", type=float, default=0.8, help="Train ratio")
    ap.add_argument("--val", type=float, default=0.1, help="Val ratio")
    ap.add_argument("--test", type=float, default=0.1, help="Test ratio")
    ap.add_argument("--force-resplit", action="store_true", help="Ignore existing split manifest and reshuffle")
    return ap.parse_args()


def clamp_bbox(xmin, ymin, xmax, ymax, w, h):
    xmin = max(0.0, min(float(xmin), w))
    xmax = max(0.0, min(float(xmax), w))
    ymin = max(0.0, min(float(ymin), h))
    ymax = max(0.0, min(float(ymax), h))
    return xmin, ymin, xmax, ymax


def pose_to_annotations(pose_path: str, img_w: int, img_h: int) -> Tuple[List[Dict], Dict[str, int]]:
    """Convert pose.jsonl to annotation dicts and collect category frequencies.

    We build rectangle segmentations/bboxes from available 2D image_position points.
    Entries with fewer than 2 points or zero-area boxes are skipped.
    """
    annos = []
    cat_counts: Dict[str, int] = {}
    with open(pose_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: could not parse line {idx} in {pose_path}")
                continue

            label = obj.get("label", "unknown")
            pts = []
            for c in obj.get("corners", []):
                pos = c.get("image_position")
                if pos is None:
                    continue
                if len(pos) != 2:
                    continue
                pts.append((float(pos[0]), float(pos[1])))

            if len(pts) < 2:
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            xmin, ymin, xmax, ymax = clamp_bbox(min(xs), min(ys), max(xs), max(ys), img_w - 1, img_h - 1)
            w_box = xmax - xmin
            h_box = ymax - ymin
            if w_box <= 0 or h_box <= 0:
                continue

            bbox = [xmin, ymin, w_box, h_box]
            seg = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
            annos.append(
                {
                    "label": label,
                    "bbox": bbox,
                    "segmentation": [seg],
                    "area": w_box * h_box,
                }
            )
            cat_counts[label] = cat_counts.get(label, 0) + 1
    return annos, cat_counts


def collect_scenes(src_root: str):
    scenes = []
    for scene_dir in sorted(glob(os.path.join(src_root, "v1_default_*"))):
        pose_path = os.path.join(scene_dir, "pose.jsonl")
        img_path = os.path.join(scene_dir, "rgb.jpg")
        if not os.path.isfile(pose_path) or not os.path.isfile(img_path):
            continue
        scenes.append({"name": os.path.basename(scene_dir), "img": img_path, "pose": pose_path})
    return scenes


def load_split_manifest(manifest_path: str):
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_split_manifest(manifest_path: str, manifest: Dict):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def split_scenes(scenes: List[Dict], seed: int, ratios: Tuple[float, float, float]):
    train_r, val_r, test_r = ratios
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "Train/val/test ratios must sum to 1."
    random.seed(seed)
    shuffled = scenes[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def ensure_dirs(out_root: str):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(out_root, "images", split), exist_ok=True)
    os.makedirs(os.path.join(out_root, "annotations"), exist_ok=True)


def build_categories(cat_counts: Dict[str, int]):
    categories = []
    for idx, name in enumerate(sorted(cat_counts.keys())):
        categories.append({"id": idx + 1, "name": name, "supercategory": "object"})
    cat_to_id = {c["name"]: c["id"] for c in categories}
    return categories, cat_to_id


def process_split(split_name: str, split_scenes: List[Dict], out_root: str, cat_to_id: Dict[str, int]):
    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    for scene in split_scenes:
        img_path = scene["img"]
        pose_path = scene["pose"]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read image {img_path}, skipping scene {scene['name']}")
            continue
        h, w = img.shape[:2]
        annos, _ = pose_to_annotations(pose_path, w, h)

        # Copy image
        fname = f"{scene['name']}.jpg"
        dst_img = os.path.join(out_root, "images", split_name, fname)
        shutil.copy2(img_path, dst_img)

        images.append({
            "id": img_id,
            "file_name": fname,
            "width": w,
            "height": h,
        })

        for a in annos:
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_to_id.get(a["label"], 0),
                "bbox": a["bbox"],
                "segmentation": a["segmentation"],
                "area": a["area"],
                "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cid, "name": name, "supercategory": "object"}
            for name, cid in sorted(cat_to_id.items(), key=lambda kv: kv[1])
        ],
    }

    out_json = os.path.join(out_root, "annotations", f"instances_{split_name}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    print(f"Wrote {out_json}: {len(images)} images, {len(annotations)} annotations")


def main():
    args = parse_args()

    scenes = collect_scenes(args.src_root)
    if not scenes:
        raise SystemExit(f"No scenes found under {args.src_root}")

    ensure_dirs(args.out_root)

    manifest_path = os.path.join(args.out_root, "split_manifest.json")
    manifest = None if args.force_resplit else load_split_manifest(manifest_path)

    if manifest is None:
        manifest = split_scenes(scenes, args.seed, (args.train, args.val, args.test))
        save_split_manifest(manifest_path, manifest)
        print(f"Created split manifest with seed {args.seed}: {manifest_path}")
    else:
        print(f"Using existing split manifest: {manifest_path}")

    # Build categories from all scenes to ensure consistency across splits
    cat_counts: Dict[str, int] = {}
    for s in scenes:
        img = cv2.imread(s["img"])
        if img is None:
            continue
        h, w = img.shape[:2]
        _, counts = pose_to_annotations(s["pose"], w, h)
        for k, v in counts.items():
            cat_counts[k] = cat_counts.get(k, 0) + v

    categories, cat_to_id = build_categories(cat_counts)
    print(f"Categories: {', '.join([c['name'] for c in categories])}")

    for split_name in ["train", "val", "test"]:
        process_split(split_name, manifest.get(split_name, []), args.out_root, cat_to_id)


if __name__ == "__main__":
    main()