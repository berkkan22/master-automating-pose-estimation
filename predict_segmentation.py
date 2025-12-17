from ultralytics import YOLO
import os
import json
import cv2
import numpy as np
from glob import glob

# Configuration
BASE_PATH = "./runs"  # folder containing YOLO training outputs
MODEL_PATH = f"{BASE_PATH}/segment/train3/weights/best.pt"

# Test dataset paths
TEST_ROOT = "./datasets/trudi_ds_yolo11_instand_segmentation/test"
TEST_IMAGES_DIR = os.path.join(TEST_ROOT, "images")
GT_JSON_DIR = "./trudi_ds/data"  # ground-truth jsons live here

# Output directory (per-image visualizations + summary)
OUT_DIR = "./results/segmentation_predictions_train3"
os.makedirs(OUT_DIR, exist_ok=True)

# Labels of interest (filter both prediction + GT)
TARGET_LABELS = {"container", "freight_car", "semi_trailer", "tank_container", "trailer"}

# Color constants
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_polygons(img, polygons, color, thickness=2, alpha=0.4):
    overlay = img.copy()
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], isClosed=True,
                      color=color, thickness=thickness)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def load_gt_polygons(json_path):
    """Return polygons only (filtered by TARGET_LABELS)."""
    shapes = load_gt_shapes(json_path)
    return [s["points"] for s in shapes]


def load_gt_shapes(json_path):
    """Return list of shapes dicts {label, points} filtered by TARGET_LABELS."""
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = []
        for s in data.get("shapes", []):
            lbl = s.get("label")
            pts = s.get("points")
            if lbl in TARGET_LABELS and pts:
                out.append({"label": lbl, "points": pts})
        return out
    except Exception as e:
        print(f"Failed to read {json_path}: {e}")
        return []


def prediction_to_polygons(ultra_result):
    polys = []
    if getattr(ultra_result, "masks", None) and ultra_result.masks is not None:
        # Filter predictions by target labels if class names are available
        try:
            names = ultra_result.names  # dict or list
            boxes = ultra_result.boxes
            if boxes is not None and hasattr(boxes, 'cls') and boxes.cls is not None:
                for i, xy in enumerate(ultra_result.masks.xy):
                    if len(xy) < 3:
                        continue
                    cls_idx = int(boxes.cls[i].item())
                    cls_name = names[cls_idx] if isinstance(
                        names, (list, dict)) else str(cls_idx)
                    if cls_name in TARGET_LABELS:
                        polys.append(xy.tolist())
            else:
                # Fallback: take all masks
                for xy in ultra_result.masks.xy:
                    if len(xy) >= 3:
                        polys.append(xy.tolist())
        except Exception:
            # Robust fallback: take all masks
            for xy in ultra_result.masks.xy:
                if len(xy) >= 3:
                    polys.append(xy.tolist())
    return polys


def prediction_label_counts(ultra_result):
    """Return dict of predicted counts per class (filtered by TARGET_LABELS)."""
    counts = {k: 0 for k in TARGET_LABELS}
    try:
        names = ultra_result.names
        boxes = ultra_result.boxes
        if boxes is not None and hasattr(boxes, 'cls') and boxes.cls is not None:
            for i in range(len(boxes.cls)):
                cls_idx = int(boxes.cls[i].item())
                cls_name = names[cls_idx] if isinstance(names, (list, dict)) else str(cls_idx)
                if cls_name in counts:
                    counts[cls_name] += 1
    except Exception:
        pass
    return counts


def poly_to_mask(poly, shape_hw):
    """Rasterize a single polygon to a binary mask of shape (H, W)."""
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


def polys_to_masks(polys, shape_hw):
    return [poly_to_mask(p, shape_hw) for p in polys]


def iou_matrix(pred_masks, gt_masks):
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    P, G = len(pred_masks), len(gt_masks)
    mat = np.zeros((P, G), dtype=np.float32)
    for i in range(P):
        pm = pred_masks[i].astype(bool)
        psum = pm.sum()
        if psum == 0:
            continue
        for j in range(G):
            gm = gt_masks[j].astype(bool)
            isum = np.logical_and(pm, gm).sum()
            usum = psum + gm.sum() - isum
            mat[i, j] = (isum / usum) if usum > 0 else 0.0
    return mat


def greedy_match(iou_mat, thr=0.5):
    used_pred = set()
    used_gt = set()
    matches = []  # (pi, gi, iou)
    if iou_mat.size == 0:
        return matches
    while True:
        # find max IoU among unused indices
        max_iou = -1.0
        best = (-1, -1)
        for i in range(iou_mat.shape[0]):
            if i in used_pred:
                continue
            for j in range(iou_mat.shape[1]):
                if j in used_gt:
                    continue
                val = iou_mat[i, j]
                if val > max_iou:
                    max_iou = val
                    best = (i, j)
        if max_iou < thr or best == (-1, -1):
            break
        pi, gi = best
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, float(max_iou)))
    return matches


def compute_metrics(pred_polys, gt_polys, shape_hw, iou_thr=0.5):
    pred_masks = polys_to_masks(pred_polys, shape_hw)
    gt_masks = polys_to_masks(gt_polys, shape_hw)
    M = iou_matrix(pred_masks, gt_masks)
    matches = greedy_match(M, thr=iou_thr)
    tp = len(matches)
    fp = max(0, len(pred_masks) - tp)
    fn = max(0, len(gt_masks) - tp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    miou = float(np.mean([m[2] for m in matches])) if tp > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "miou": miou,
        "matches": matches,
    }


def overlay_text_block(image, lines, org=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale=0.7, color=(255, 255, 255), thickness=2,
                       bg_color=(0, 0, 0), alpha=0.5, line_height=24):
    # Compute background rectangle size
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
    # Put text lines
    for i, line in enumerate(lines):
        yy = y + i * line_height
        cv2.putText(out, line, (x, yy), font, font_scale,
                    color, thickness, cv2.LINE_AA)
    return out


def main():
    if not os.path.isdir(TEST_IMAGES_DIR):
        raise FileNotFoundError(f"Test images directory not found: {TEST_IMAGES_DIR}")

    # Gather test images
    exts = {".jpg", ".jpeg", ".png"}
    image_paths = [p for p in sorted(glob(os.path.join(TEST_IMAGES_DIR, '*'))) if os.path.splitext(p)[1].lower() in exts]
    if not image_paths:
        raise RuntimeError(f"No test images found in {TEST_IMAGES_DIR}")

    # Load model once
    model = YOLO(MODEL_PATH)

    # Prepare summary log file
    list_path = os.path.join(OUT_DIR, 'per_image_summary.txt')
    summary_lines = ["image; gt_counts; pred_counts\n"]

    # Run predictions lazily per image to allow skipping when no GT labels
    # (We avoid predicting images that have no GT labels.)
    results = []

    # Accumulators for aggregate metrics
    agg = {
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'precision_sum': 0.0,
        'recall_sum': 0.0,
        'f1_sum': 0.0,
        'miou_sum': 0.0,
        'count': 0
    }

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read image {img_path}, skipping.")
            continue

        base = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(GT_JSON_DIR, base + '.json')

        # Ground truth shapes and counts; skip if no labels
        gt_shapes = load_gt_shapes(json_path)
        if not gt_shapes:
            # No GT labels -> do not predict and do not save
            print(f"[{base}] No GT labels found. Skipping prediction and save.")
            continue
        gt_polys = [s["points"] for s in gt_shapes]
        gt_vis = draw_polygons(img.copy(), gt_polys, color=RED, thickness=2, alpha=0.35)
        # GT counts per class
        gt_counts = {k: 0 for k in TARGET_LABELS}
        for s in gt_shapes:
            if s["label"] in gt_counts:
                gt_counts[s["label"]] += 1

        # Run prediction now (since we have GT labels)
        # Run prediction quietly to avoid console clutter
        res = model([img_path], conf=0.80, verbose=False)[0]
        pred_polys = prediction_to_polygons(res)
        pred_vis = draw_polygons(img.copy(), pred_polys, color=GREEN, thickness=2, alpha=0.35)
        pred_counts = prediction_label_counts(res)

        # Side-by-side
        h = max(gt_vis.shape[0], pred_vis.shape[0])
        w = gt_vis.shape[1] + pred_vis.shape[1]
        combined = np.zeros((h, w, 3), dtype=np.uint8)
        combined[:gt_vis.shape[0], :gt_vis.shape[1]] = gt_vis
        combined[:pred_vis.shape[0], gt_vis.shape[1]:gt_vis.shape[1] + pred_vis.shape[1]] = pred_vis

        # Metrics (only if GT available)
        H, W = img.shape[:2]
        metrics = compute_metrics(pred_polys, gt_polys, (H, W), iou_thr=0.5) if gt_polys else {
            'tp': 0, 'fp': len(pred_polys), 'fn': 0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'miou': 0.0
        }
        lines = [
            f"Image: {base}",
            f"Pred={len(pred_polys)} GT={len(gt_polys)}",
            f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}",
            f"Precision={metrics['precision']:.3f} Recall={metrics['recall']:.3f}",
            f"F1={metrics['f1']:.3f} mIoU={metrics['miou']:.3f}",
        ]
        combined = overlay_text_block(combined, lines, org=(20, 50))

        # Save
        out_pred = os.path.join(OUT_DIR, f"{base}_pred.jpg")
        out_gt = os.path.join(OUT_DIR, f"{base}_gt.jpg")
        out_combined = os.path.join(OUT_DIR, f"{base}_gt_vs_pred.jpg")
        cv2.imwrite(out_pred, pred_vis)
        cv2.imwrite(out_gt, gt_vis)
        cv2.imwrite(out_combined, combined)

        # Aggregate
        agg['tp'] += metrics['tp']
        agg['fp'] += metrics['fp']
        agg['fn'] += metrics['fn']
        agg['precision_sum'] += metrics['precision']
        agg['recall_sum'] += metrics['recall']
        agg['f1_sum'] += metrics['f1']
        agg['miou_sum'] += metrics['miou']
        agg['count'] += 1

        # Build human-readable class counts summary line
        def fmt_counts(d):
            parts = []
            for cls in sorted(TARGET_LABELS):
                parts.append(f"{d.get(cls,0)} {cls}")
            return "; ".join(parts)

        line = f"{base}; GT: {fmt_counts(gt_counts)}; Pred: {fmt_counts(pred_counts)}\n"
        summary_lines.append(line)

    # Summary
    if agg['count'] > 0:
        # Write per-image summary once at the end
        with open(list_path, 'w', encoding='utf-8') as summary_file:
            summary_file.writelines(summary_lines)
        summary = {
            'images': agg['count'],
            'tp': agg['tp'],
            'fp': agg['fp'],
            'fn': agg['fn'],
            'precision_mean': agg['precision_sum'] / agg['count'],
            'recall_mean': agg['recall_sum'] / agg['count'],
            'f1_mean': agg['f1_sum'] / agg['count'],
            'miou_mean': agg['miou_sum'] / agg['count'],
        }
        # Write summary JSON
        summary_path = os.path.join(OUT_DIR, 'summary_metrics.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        # Also write aggregate summary to a txt alongside JSON
        agg_txt_path = os.path.join(OUT_DIR, 'summary_metrics.txt')
        with open(agg_txt_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(summary, indent=2))
    else:
        print("No images processed.")


if __name__ == "__main__":
    main()
