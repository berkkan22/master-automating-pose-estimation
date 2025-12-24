"""
Pose prediction and visualization script using a YOLOv11 pose model.
Mirrors the segmentation predictor but tailored for keypoints.
"""
import os
import json
from glob import glob
from typing import List, Dict, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Configuration
BASE_PATH = "./runs"  # folder containing YOLO training outputs
MODEL_PATH = f"{BASE_PATH}/pose/train/weights/best.pt"

# Test dataset paths
TEST_ROOT = "./datasets/synthetic_data-v2-coco-v2"
TEST_IMAGES_DIR = os.path.join(TEST_ROOT, "images/test")
GT_LABELS_DIR = os.path.join(TEST_ROOT, "labels/test")  # YOLO keypoint labels

# Output directory (per-image visualizations + summary)
OUT_DIR = "./results/pose_predictions_train"
os.makedirs(OUT_DIR, exist_ok=True)

# Target classes to keep; leave empty to keep all
TARGET_LABELS = {0}  # dataset only has one class (transportation_units)

# Visualization colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Metrics
IOU_THR = 0.5           # bbox IoU threshold to match predictions to GT
PCK_THR = 0.05          # keypoint correct if within 5% of max(image_w, image_h)


def overlay_text_block(image, lines, org=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale=0.7, color=(255, 255, 255), thickness=2,
                       bg_color=(0, 0, 0), alpha=0.5, line_height=24):
    """Draw a translucent text block for readability."""
    max_w = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_w = max(max_w, w)
    bg_w = max_w + 20
    bg_h = line_height * len(lines) + 10
    x, y = org
    overlay = image.copy()
    cv2.rectangle(overlay, (x - 10, y - 20),
                  (x - 10 + bg_w, y - 20 + bg_h), bg_color, -1)
    out = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    for i, line in enumerate(lines):
        yy = y + i * line_height
        cv2.putText(out, line, (x, yy), font, font_scale,
                    color, thickness, cv2.LINE_AA)
    return out


def xywhn_to_xyxy(xc, yc, w, h, W, H):
    x1 = (xc - w / 2.0) * W
    y1 = (yc - h / 2.0) * H
    x2 = (xc + w / 2.0) * W
    y2 = (yc + h / 2.0) * H
    return x1, y1, x2, y2


def load_gt_labels(label_path: str, shape_hw: Tuple[int, int]):
    """Parse YOLO keypoint labels -> list of dicts with bbox and keypoints."""
    H, W = shape_hw
    out = []
    if not os.path.exists(label_path):
        return out
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if TARGET_LABELS and cls not in TARGET_LABELS:
            continue
        xc, yc, w, h = map(float, parts[1:5])
        bbox = xywhn_to_xyxy(xc, yc, w, h, W, H)
        kpt_vals = parts[5:]
        keypoints = []  # (x, y, v)
        for i in range(0, len(kpt_vals), 3):
            try:
                kx = float(kpt_vals[i]) * W
                ky = float(kpt_vals[i + 1]) * H
                v = float(kpt_vals[i + 2])
                keypoints.append((kx, ky, v))
            except (ValueError, IndexError):
                break
        out.append({"cls": cls, "bbox": bbox, "keypoints": keypoints})
    return out


def prediction_to_struct(res, shape_hw: Tuple[int, int]):
    """Convert YOLO result to structured list with bbox + keypoints."""
    H, W = shape_hw
    out = []
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
        bbox = xywhn_to_xyxy(float(xc), float(yc), float(w), float(h), W, H)
        keypoints = []
        if kpt_xyn is not None:
            for (kx, ky) in kpt_xyn[i]:
                keypoints.append((float(kx * W), float(ky * H), 2.0))
        out.append({"cls": cls_id, "bbox": bbox, "keypoints": keypoints})
    return out


def bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def greedy_match(preds, gts, thr=IOU_THR):
    """Match predictions to GT by bbox IoU (greedy highest-first)."""
    matches = []  # (pi, gi, iou)
    used_p = set()
    used_g = set()
    for _ in range(min(len(preds), len(gts))):
        best = (-1, -1)
        best_iou = -1.0
        for i, p in enumerate(preds):
            if i in used_p:
                continue
            for j, g in enumerate(gts):
                if j in used_g:
                    continue
                iou = bbox_iou(p["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best = (i, j)
        if best_iou < thr or best == (-1, -1):
            break
        used_p.add(best[0])
        used_g.add(best[1])
        matches.append((best[0], best[1], best_iou))
    return matches


def keypoint_metrics(pred_item, gt_item, shape_hw: Tuple[int, int]):
    """Compute per-instance keypoint distances and PCK."""
    H, W = shape_hw
    max_dim = max(H, W)
    pred_k = pred_item.get("keypoints", [])
    gt_k = gt_item.get("keypoints", [])
    n = min(len(pred_k), len(gt_k))
    if n == 0:
        return {"count": 0, "pck_hits": 0, "mean_dist_norm": 0.0}
    dists = []
    hits = 0
    for i in range(n):
        gx, gy, gv = gt_k[i]
        px, py, _ = pred_k[i]
        if gv <= 0:  # invisible
            continue
        dist = np.sqrt((gx - px) ** 2 + (gy - py) ** 2)
        dists.append(dist / max_dim)
        if dist <= PCK_THR * max_dim:
            hits += 1
    if not dists:
        return {"count": 0, "pck_hits": 0, "mean_dist_norm": 0.0}
    return {
        "count": len(dists),
        "pck_hits": hits,
        "mean_dist_norm": float(np.mean(dists)),
    }


def draw_keypoints(img, items, color, radius=4, thickness=2):
    out = img.copy()
    for it in items:
        x1, y1, x2, y2 = map(int, it["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        for idx, (kx, ky, _) in enumerate(it.get("keypoints", [])):
            cv2.circle(out, (int(kx), int(ky)), radius, color, -1)
            cv2.putText(out, str(idx), (int(kx) + 3, int(ky) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness, cv2.LINE_AA)
    return out


def fmt_counts(items: List[Dict]):
    counts = {}
    for it in items:
        cls = it.get("cls", -1)
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def main():
    if not os.path.isdir(TEST_IMAGES_DIR):
        raise FileNotFoundError(f"Test images directory not found: {TEST_IMAGES_DIR}")

    image_paths = [p for p in sorted(glob(os.path.join(TEST_IMAGES_DIR, '*'))) if os.path.isfile(p)]
    if not image_paths:
        raise RuntimeError(f"No test images found in {TEST_IMAGES_DIR}")

    model = YOLO(MODEL_PATH)

    list_path = os.path.join(OUT_DIR, "per_image_summary.txt")
    summary_lines = ["image; gt_counts; pred_counts\n"]

    agg = {
        "images": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "mean_pck": 0.0,
        "mean_kpt_dist": 0.0,
        "kpt_instances": 0,
    }

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read image {img_path}, skipping.")
            continue
        H, W = img.shape[:2]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(GT_LABELS_DIR, f"{base}.txt")

        gt_items = load_gt_labels(label_path, (H, W))
        if not gt_items:
            print(f"[{base}] No GT labels found. Skipping prediction and save.")
            continue

        # Predict
        res = model([img_path], verbose=False, conf=0.5)[0]
        pred_items = prediction_to_struct(res, (H, W))

        # Visuals
        gt_vis = draw_keypoints(img, gt_items, color=RED)
        pred_vis = draw_keypoints(img, pred_items, color=GREEN)

        # Metrics
        matches = greedy_match(pred_items, gt_items, thr=IOU_THR)
        tp = len(matches)
        fp = max(0, len(pred_items) - tp)
        fn = max(0, len(gt_items) - tp)

        # Keypoint metrics per matched pair
        pck_hits = 0
        kpt_total = 0
        dist_sum = 0.0
        for pi, gi, _ in matches:
            km = keypoint_metrics(pred_items[pi], gt_items[gi], (H, W))
            pck_hits += km["pck_hits"]
            kpt_total += km["count"]
            dist_sum += km["mean_dist_norm"] * km["count"]
        pck = (pck_hits / kpt_total) if kpt_total > 0 else 0.0
        mean_dist = (dist_sum / kpt_total) if kpt_total > 0 else 0.0

        lines = [
            f"Image: {base}",
            f"Pred={len(pred_items)} GT={len(gt_items)}",
            f"TP={tp} FP={fp} FN={fn}",
            f"PCK@{PCK_THR*100:.0f}%={pck:.3f} mean_kpt_dist_norm={mean_dist:.3f}",
        ]
        # Side-by-side
        h = max(gt_vis.shape[0], pred_vis.shape[0])
        w = gt_vis.shape[1] + pred_vis.shape[1]
        combined = np.zeros((h, w, 3), dtype=np.uint8)
        combined[:gt_vis.shape[0], :gt_vis.shape[1]] = gt_vis
        combined[:pred_vis.shape[0], gt_vis.shape[1]:gt_vis.shape[1] + pred_vis.shape[1]] = pred_vis
        combined = overlay_text_block(combined, lines, org=(20, 50))

        # Save outputs
        out_pred = os.path.join(OUT_DIR, f"{base}_pred.jpg")
        out_gt = os.path.join(OUT_DIR, f"{base}_gt.jpg")
        out_combined = os.path.join(OUT_DIR, f"{base}_gt_vs_pred.jpg")
        cv2.imwrite(out_pred, pred_vis)
        cv2.imwrite(out_gt, gt_vis)
        cv2.imwrite(out_combined, combined)

        # Aggregate
        agg["images"] += 1
        agg["tp"] += tp
        agg["fp"] += fp
        agg["fn"] += fn
        if kpt_total > 0:
            agg["mean_pck"] += pck
            agg["mean_kpt_dist"] += mean_dist
            agg["kpt_instances"] += 1

        summary_lines.append(
            f"{base}; GT: {fmt_counts(gt_items)}; Pred: {fmt_counts(pred_items)}\n"
        )

    if agg["images"] > 0:
        # Per-image summary
        with open(list_path, "w", encoding="utf-8") as summary_file:
            summary_file.writelines(summary_lines)
        summary = {
            "images": agg["images"],
            "tp": agg["tp"],
            "fp": agg["fp"],
            "fn": agg["fn"],
            "mean_pck": (agg["mean_pck"] / agg["kpt_instances"]) if agg["kpt_instances"] > 0 else 0.0,
            "mean_kpt_dist_norm": (agg["mean_kpt_dist"] / agg["kpt_instances"]) if agg["kpt_instances"] > 0 else 0.0,
        }
        summary_path = os.path.join(OUT_DIR, "summary_metrics.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        agg_txt_path = os.path.join(OUT_DIR, "summary_metrics.txt")
        with open(agg_txt_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(summary, indent=2))
    else:
        print("No images processed.")


if __name__ == "__main__":
    main()
