import argparse
import json
import os
from typing import List, Tuple

import cv2
import numpy as np

from models import PoseEstimationResult, Keypoint, BoundingBox, PoseDepth3D
from yolo_estimator import YoloPoseEstimator
from depth_estimator import DepthAnything3Estimator


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two axis-aligned bounding boxes in xyxy format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    denom = float(area_a + area_b - inter_area)
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def best_fit_transform_with_scale(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute similarity transform (s, R, t) that best aligns A to B.

    Solves:  B \approx s * R @ A + t

    Returns (R, t, s).
    """
    assert A.shape == B.shape
    assert A.shape[1] == 3

    # Subtract centroids
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    # Compute covariance matrix
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle possible reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Compute scale
    var_A = np.sum(AA ** 2)
    if var_A <= 0:
        s = 1.0
    else:
        s = np.sum(S) / var_A

    # Translation
    t = centroid_B - s * (R @ centroid_A)

    return R, t, float(s)


def load_gt_objects(pose_jsonl_path: str) -> List[dict]:
    """Load synthetic GT objects from pose.jsonl.

    Each object has keys:
      - 'bbox_xyxy': (x1, y1, x2, y2) from visible image_position corners
      - 'points_3d': np.ndarray (8, 3) with world_position coordinates
    Objects without any visible image_position are discarded.
    """
    objects: List[dict] = []
    if not os.path.exists(pose_jsonl_path):
        raise FileNotFoundError(f"pose.jsonl not found at {pose_jsonl_path}")

    with open(pose_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            corners = obj.get("corners", [])
            if len(corners) != 8:
                continue

            pts3d: List[Tuple[float, float, float]] = []
            xs: List[float] = []
            ys: List[float] = []

            # Ensure we follow the same corner ordering as in training
            # world_position is [idx, [X, Y, Z]]
            for corner in corners:
                wp = corner.get("world_position", None)
                ip = corner.get("image_position", None)
                if not isinstance(wp, list) or len(wp) != 2:
                    break
                coords = wp[1]
                if not isinstance(coords, list) or len(coords) != 3:
                    break
                X, Y, Z = float(coords[0]), float(coords[1]), float(coords[2])
                pts3d.append((X, Y, Z))

                if ip is not None and len(ip) == 2:
                    # image_position is [x, y]
                    x, y = float(ip[0]), float(ip[1])
                    xs.append(x)
                    ys.append(y)

            if len(pts3d) != 8:
                continue

            if not xs or not ys:
                # No visible keypoints -> skip
                continue

            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            if x2 <= x1 or y2 <= y1:
                continue

            objects.append(
                {
                    "bbox_xyxy": (int(x1), int(y1), int(x2), int(y2)),
                    "points_3d": np.asarray(pts3d, dtype=np.float32),
                }
            )

    return objects


def group_predicted_keypoints(frame_3d: PoseDepth3D) -> List[np.ndarray]:
    """Group PoseDepth3D.keypoints_3d into per-detection arrays.

    Assumes that keypoints are ordered as in YoloPoseEstimator:
    first all keypoints of det 0, then of det 1, ...
    """
    num_boxes = len(frame_3d.pose_result.boundingBoxes)
    num_kpts = len(frame_3d.keypoints_3d)
    if num_boxes == 0 or num_kpts == 0:
        return []

    kpts_per_box = num_kpts // num_boxes
    if kpts_per_box == 0:
        return []

    grouped: List[np.ndarray] = []
    for det_idx in range(num_boxes):
        start = det_idx * kpts_per_box
        end = start + kpts_per_box
        pts: List[Tuple[float, float, float]] = []
        for kp in frame_3d.keypoints_3d[start:end]:
            # Use camera-space coordinates (X, Y, Z)
            if kp.X is None or kp.Y is None or kp.Z is None:
                continue
            pts.append((float(kp.X), float(kp.Y), float(kp.Z)))
        if pts:
            grouped.append(np.asarray(pts, dtype=np.float32))
    return grouped


def _xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, W: int, H: int) -> Tuple[int, int, int, int]:
    """Convert normalized YOLO (xc,yc,w,h) to pixel-space (x1,y1,x2,y2)."""
    x1 = (xc - w / 2.0) * W
    y1 = (yc - h / 2.0) * H
    x2 = (xc + w / 2.0) * W
    y2 = (yc + h / 2.0) * H
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _load_sorted_folders(base_path: str) -> List[str]:
    return [
        os.path.join(base_path, d)
        for d in sorted(os.listdir(base_path))
        if os.path.isdir(os.path.join(base_path, d))
    ]


def _load_split_order_indices(split_txt: str, total_folders: int) -> List[int]:
    if not os.path.exists(split_txt):
        raise FileNotFoundError(f"Split order file not found: {split_txt}")

    indices: List[int] = []
    with open(split_txt, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            stem = os.path.splitext(ln)[0]
            try:
                idx = int(stem)
            except ValueError:
                continue
            if 0 <= idx < total_folders:
                indices.append(idx)
    return indices


def _compute_splits(order: List[int], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
    total = len(order)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_order = order[:train_end]
    val_order = order[train_end:val_end]
    test_order = order[val_end:]
    return train_order, val_order, test_order


def _resolve_scene_from_dataset_index(
    base_synth_root: str,
    split_order_path: str,
    split: str,
    index: int,
) -> str:
    """Map a YOLO dataset (split, index) back to its synthetic_data-v2 scene folder.

    This mirrors the logic in scripts/convert_synthetic_data-v2_to_coco_split_order.py
    so that index k in images/{split} corresponds to the same scene folder
    that was used when generating k.jpg and k.txt.
    """
    folders_sorted = _load_sorted_folders(base_synth_root)
    total = len(folders_sorted)
    order_indices = _load_split_order_indices(split_order_path, total)
    if not order_indices:
        raise RuntimeError("No valid indices loaded from split_order_synthetic.txt")

    train_folders, val_folders, test_folders = _compute_splits(order_indices)

    split = split.lower()
    if split == "train":
        split_indices = train_folders
    elif split == "val":
        split_indices = val_folders
    elif split == "test":
        split_indices = test_folders
    else:
        raise ValueError(f"Invalid split '{split}', expected one of ['train', 'val', 'test'].")

    if index < 0 or index >= len(split_indices):
        raise IndexError(f"Index {index} out of range for split '{split}' (len={len(split_indices)}).")

    folder_idx = split_indices[index]
    return folders_sorted[folder_idx]


def load_gt_from_yolo_labels(
    label_path: str,
    image_shape: Tuple[int, int],
    pose_jsonl_path: str,
    iou_thresh: float = 0.0,
) -> List[dict]:
    """Build GT objects using YOLO pose labels + synthetic 3D from pose.jsonl.

    - Parses YOLO keypoint labels to get 2D GT boxes.
    - Loads pose.jsonl and builds 3D corner points per object.
    - Matches each YOLO GT box to a pose.jsonl object by IoU.

    Returns list of dicts with keys:
      - 'bbox_xyxy': 2D GT bbox from YOLO label (pixels)
      - 'points_3d': (8,3) world coordinates from pose.jsonl
    """
    H, W = image_shape

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"YOLO label file not found: {label_path}")

    # Load YOLO GT boxes + 2D keypoints
    yolo_boxes: List[dict] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 5:
                continue
            # cls = int(float(parts[0]))  # class id, currently unused
            xc, yc, w, h = map(float, parts[1:5])
            bbox = _xywhn_to_xyxy(xc, yc, w, h, W, H)
            kpt_vals = parts[5:]
            keypoints_2d: List[Tuple[float, float, float]] = []
            for i in range(0, len(kpt_vals), 3):
                try:
                    kx = float(kpt_vals[i]) * W
                    ky = float(kpt_vals[i + 1]) * H
                    v = float(kpt_vals[i + 2])
                    keypoints_2d.append((kx, ky, v))
                except (ValueError, IndexError):
                    break
            yolo_boxes.append({"bbox_xyxy": bbox, "keypoints_2d": keypoints_2d})

    if not yolo_boxes:
        return []

    # Load pose-based GT objects (with 3D points and their own 2D bboxes)
    pose_objs = load_gt_objects(pose_jsonl_path)
    if not pose_objs:
        return []

    gt_objects: List[dict] = []
    for yobj in yolo_boxes:
        y_box = yobj["bbox_xyxy"]
        best_iou = 0.0
        best_idx = -1
        for idx, pobj in enumerate(pose_objs):
            iou = compute_iou(y_box, pobj["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_thresh:
            gt_objects.append(
                {
                    "bbox_xyxy": y_box,
                    "points_3d": pose_objs[best_idx]["points_3d"],
                    # 2D keypoints (pixels) from YOLO labels; used for dev GT-2D comparison
                    "keypoints_2d": yobj.get("keypoints_2d"),
                }
            )

    return gt_objects


def match_detections_to_gt(
    pose_result: PoseEstimationResult,
    gt_objects: List[dict],
    iou_thresh: float = 0.0,
) -> List[Tuple[int, int]]:
    """Match predicted boxes to GT objects using IoU.

    Returns list of (pred_idx, gt_idx) pairs.
    """
    matches: List[Tuple[int, int]] = []

    gt_used = set()
    for pred_idx, box in enumerate(pose_result.boundingBoxes):
        pred_box = (box.x1, box.y1, box.x2, box.y2)
        best_iou = 0.0
        best_gt = -1
        for gt_idx, gt in enumerate(gt_objects):
            if gt_idx in gt_used:
                continue
            iou = compute_iou(pred_box, gt["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= iou_thresh:
            matches.append((pred_idx, best_gt))
            gt_used.add(best_gt)

    return matches

def _build_pose_result_from_gt_2d(
    image_shape: Tuple[int, int],
    gt_objects: List[dict],
) -> PoseEstimationResult:
    """Construct a PoseEstimationResult using GT 2D keypoints from YOLO labels.

    This is used in a dev mode to compare ADD when using "perfect" 2D keypoints
    (from labels) versus predicted 2D keypoints from YOLO.
    """
    keypoints: List[Keypoint] = []
    boxes: List[BoundingBox] = []

    for obj in gt_objects:
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
        kpts2d = obj.get("keypoints_2d") or []
        for kp_idx, (x, y, _v) in enumerate(kpts2d):
            keypoints.append(Keypoint(id=kp_idx, x=float(x), y=float(y)))

    return PoseEstimationResult(imageShape=image_shape, keypoints=keypoints, boundingBoxes=boxes)


def _compute_add_for_pose_result(
    pose_result: PoseEstimationResult,
    depth_result: "DepthAnything3Estimator",
    gt_objects: List[dict],
    aligned: bool,
    label: str,
) -> List[float]:
    """Compute ADD values for a given pose_result against GT 3D points.

    Returns list of per-object ADD values. Prints intermediate diagnostics
    with the provided label (e.g. "PRED" or "GT2D").
    """
    if not pose_result.keypoints or not pose_result.boundingBoxes:
        print(f"[{label}] No pose detections / keypoints available.")
        return []

    frame_3d = PoseDepth3D(pose_result, depth_result)
    frame_3d.compute_camera_coordinates()

    pred_groups = group_predicted_keypoints(frame_3d)
    if not pred_groups:
        print(f"[{label}] No 3D keypoints available after depth fusion.")
        return []

    matches = match_detections_to_gt(pose_result, gt_objects)
    print(
        f"[{label}] Pred boxes: {len(pose_result.boundingBoxes)}, "
        f"GT objects: {len(gt_objects)}, matched pairs: {len(matches)}"
    )
    if not matches:
        print(f"[{label}] No predicted boxes matched any GT objects (IoU too low?).")
        return []

    all_add_values: List[float] = []

    for pred_idx, gt_idx in matches:
        if pred_idx >= len(pred_groups):
            continue
        pred_pts = pred_groups[pred_idx]
        gt_pts = gt_objects[gt_idx]["points_3d"]

        if pred_pts.shape[0] != gt_pts.shape[0]:
            n = min(pred_pts.shape[0], gt_pts.shape[0])
            pred_pts = pred_pts[:n]
            gt_pts = gt_pts[:n]

        if aligned:
            R, t, s = best_fit_transform_with_scale(pred_pts, gt_pts)
            pred_aligned = (s * (R @ pred_pts.T)).T + t
            diffs = pred_aligned - gt_pts
        else:
            diffs = pred_pts - gt_pts

        dists = np.linalg.norm(diffs, axis=1)
        add_val = float(dists.mean())
        all_add_values.append(add_val)

        print(f"[{label}] Object match (pred {pred_idx}, gt {gt_idx}): ADD = {add_val:.6f} (N={len(dists)})")

    if not all_add_values:
        print(f"[{label}] No ADD values computed.")
        return []

    add_mean = float(np.mean(all_add_values))
    add_std = float(np.std(all_add_values))
    print(f"[{label}] Summary over matched objects:")
    print(f"  ADD mean = {add_mean:.6f}")
    print(f"  ADD std  = {add_std:.6f}")

    return all_add_values


def _run_add_on_image(
    image: np.ndarray,
    gt_objects: List[dict],
    model_path: str,
    aligned: bool = True,
    conf: float = 0.8,
    dev_compare_gt_2d: bool = False,
    gt_pose_result: PoseEstimationResult | None = None,
) -> Tuple[List[float], List[float]]:
    """Shared core: run pose+depth on an image and compute ADD vs GT 3D points.

    If ``dev_compare_gt_2d`` is True and ``gt_pose_result`` is provided, the
    function will compute ADD twice:
      1) Using predicted 2D keypoints from YOLO.
      2) Using GT 2D keypoints from YOLO labels (via ``gt_pose_result``).
    """
    # 1) Pose estimation (predicted keypoints)
    pose_estimator = YoloPoseEstimator(model_path=model_path, conf=conf)
    pose_result_pred = pose_estimator.predict(image)

    if not pose_result_pred.keypoints or not pose_result_pred.boundingBoxes:
        raise RuntimeError("No pose detections found by YOLO.")

    # 2) Depth estimation (shared for both modes)
    depth_estimator = DepthAnything3Estimator(output_dir=None)
    depth_result = depth_estimator.predict(image)

    # 3) Compute ADD using predicted 2D keypoints
    print("=== ADD using predicted 2D keypoints (YOLO) ===")
    add_pred = _compute_add_for_pose_result(
        pose_result=pose_result_pred,
        depth_result=depth_result,
        gt_objects=gt_objects,
        aligned=aligned,
        label="PRED",
    )
    if not add_pred:
        raise RuntimeError("Could not compute ADD for predicted keypoints.")

    # 4) Optionally: compute ADD using GT 2D keypoints from labels
    add_gt: List[float] = []
    if dev_compare_gt_2d and gt_pose_result is not None:
        print("\n=== ADD using GT 2D keypoints (YOLO labels) ===")
        add_gt = _compute_add_for_pose_result(
            pose_result=gt_pose_result,
            depth_result=depth_result,
            gt_objects=gt_objects,
            aligned=aligned,
            label="GT2D",
        )
        if add_gt:
            print("\n=== Comparison of mean ADD ===")
            print(f"  Predicted 2D keypoints: {float(np.mean(add_pred)):.6f}")
            print(f"  GT 2D keypoints       : {float(np.mean(add_gt)):.6f}")

    return add_pred, add_gt


def evaluate_add_for_scene(
    scene_dir: str,
    model_path: str,
    aligned: bool = True,
    conf: float = 0.5,
) -> None:
    """Run the full pipeline on one synthetic scene folder (rgb.jpg + pose.jsonl)."""
    img_path = os.path.join(scene_dir, "rgb.jpg")
    pose_jsonl_path = os.path.join(scene_dir, "pose.jsonl")

    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"rgb.jpg not found in {scene_dir}")

    gt_objects = load_gt_objects(pose_jsonl_path)
    if not gt_objects:
        raise RuntimeError(f"No valid GT objects found in {pose_jsonl_path}")

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    _run_add_on_image(image, gt_objects, model_path=model_path, aligned=aligned, conf=conf)


def evaluate_add_for_dataset_entry(
    data_root: str,
    split: str,
    index: int,
    model_path: str,
    aligned: bool = True,
    conf: float = 0.5,
    dev_compare_gt_2d: bool = False,
    base_synth_root: str = "./synthetic_data-v2/synthetic_data-v2",
    split_order_path: str = "./split_order_synthetic.txt",
    ) -> Tuple[List[float], List[float]]:
    """Run ADD evaluation for one image/label pair from the YOLO pose dataset.

    - Uses labels in datasets/synthetic_data_yolo11_pose_new to get 2D GT boxes.
    - Maps (split, index) back to the original synthetic_data-v2 scene folder.
    - Uses that scene's pose.jsonl to obtain 3D GT keypoints.
    - Runs pose+depth pipeline on the dataset image and computes ADD.
    """
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of 'train', 'val', 'test'")

    img_path = os.path.join(data_root, "images", split, f"{index}.jpg")
    label_path = os.path.join(data_root, "labels", split, f"{index}.txt")

    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Label not found: {label_path}")

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    H, W = image.shape[:2]

    # Map dataset index back to synthetic scene folder to read pose.jsonl (3D GT).
    scene_dir = _resolve_scene_from_dataset_index(
        base_synth_root=base_synth_root,
        split_order_path=split_order_path,
        split=split,
        index=index,
    )
    pose_jsonl_path = os.path.join(scene_dir, "pose.jsonl")

    gt_objects = load_gt_from_yolo_labels(label_path, (H, W), pose_jsonl_path)
    if not gt_objects:
        raise RuntimeError(
            f"No valid GT objects constructed from labels in {label_path} "
            f"and pose data in {pose_jsonl_path}."
        )

    gt_pose_result = None
    if dev_compare_gt_2d:
        # Build a PoseEstimationResult using GT 2D keypoints from labels
        gt_pose_result = _build_pose_result_from_gt_2d((H, W), gt_objects)

    return _run_add_on_image(
        image,
        gt_objects,
        model_path=model_path,
        aligned=aligned,
        conf=conf,
        dev_compare_gt_2d=dev_compare_gt_2d,
        gt_pose_result=gt_pose_result,
    )


def evaluate_add_for_dataset_split(
    data_root: str,
    split: str,
    model_path: str,
    aligned: bool = True,
    conf: float = 0.5,
    dev_compare_gt_2d: bool = False,
    base_synth_root: str = "./synthetic_data-v2/synthetic_data-v2",
    split_order_path: str = "./split_order_synthetic.txt",
) -> None:
    """Run ADD evaluation for all images in a given YOLO pose dataset split.

    Iterates over all label files in labels/{split} and calls
    evaluate_add_for_dataset_entry for each corresponding index.
    """
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of 'train', 'val', 'test'")

    labels_dir = os.path.join(data_root, "labels", split)
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    indices: List[int] = []
    for name in os.listdir(labels_dir):
        if not name.endswith(".txt"):
            continue
        stem = os.path.splitext(name)[0]
        try:
            idx = int(stem)
        except ValueError:
            continue
        indices.append(idx)

    if not indices:
        raise RuntimeError(f"No label files found in {labels_dir}")

    indices.sort()
    print(f"Running ADD evaluation for split '{split}' over {len(indices)} samples...")

    all_add_pred: List[float] = []
    all_add_gt: List[float] = []

    for idx in indices:
        print("\n" + "=" * 80)
        print(f"Sample index {idx}")
        try:
            add_pred, add_gt = evaluate_add_for_dataset_entry(
                data_root=data_root,
                split=split,
                index=idx,
                model_path=model_path,
                aligned=aligned,
                conf=conf,
                dev_compare_gt_2d=dev_compare_gt_2d,
                base_synth_root=base_synth_root,
                split_order_path=split_order_path,
            )
            all_add_pred.extend(add_pred)
            all_add_gt.extend(add_gt)
        except Exception as e:  # pragma: no cover - dev convenience
            print(f"[WARN] Skipping index {idx} due to error: {e}")

    if all_add_pred:
        mean_pred = float(np.mean(all_add_pred))
        std_pred = float(np.std(all_add_pred))
        print("\n" + "=" * 80)
        print(f"Global summary for split '{split}' (predicted 2D keypoints):")
        print(f"  ADD mean = {mean_pred:.6f}")
        print(f"  ADD std  = {std_pred:.6f}")

    if all_add_gt:
        mean_gt = float(np.mean(all_add_gt))
        std_gt = float(np.std(all_add_gt))
        print("\n" + "=" * 80)
        print(f"Global summary for split '{split}' (GT 2D keypoints):")
        print(f"  ADD mean = {mean_gt:.6f}")
        print(f"  ADD std  = {std_gt:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate depth+pose pipeline with ADD metric either on a raw "
            "synthetic scene folder (rgb.jpg + pose.jsonl) or on a YOLO "
            "pose dataset sample using its labels as 2D GT."
        )
    )

    # Mode 1: directly on a synthetic scene folder
    parser.add_argument(
        "--scene-dir",
        type=str,
        help="Path to a synthetic_data-v2 scene folder (with rgb.jpg and pose.jsonl)",
    )

    # Mode 2: on YOLO dataset entry (uses labels + mapping back to pose.jsonl)
    parser.add_argument(
        "--data-root",
        type=str,
        help=(
            "Root of YOLO pose dataset (e.g. ./datasets/synthetic_data_yolo11_pose_new) "
            "with images/{split} and labels/{split} subfolders."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        help="Dataset split to evaluate when using --data-root.",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Image index within the given split (e.g. 1 for 1.jpg / 1.txt).",
    )

    parser.add_argument("--model-path", type=str, required=True, help="Path to YOLO pose weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    parser.add_argument("--no-align", action="store_true", help="Do not align predicted and GT 3D points before computing ADD")
    parser.add_argument(
        "--dev-compare-gt2d",
        action="store_true",
        help=(
            "Dev mode (dataset only): also compute ADD using GT 2D keypoints "
            "from YOLO labels and print a comparison to predicted 2D keypoints."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aligned = not args.no_align

    if args.scene_dir and args.data_root:
        raise RuntimeError("Please specify either --scene-dir or --data-root/--split/--index, not both.")

    if args.scene_dir:
        evaluate_add_for_scene(
            scene_dir=args.scene_dir,
            model_path=args.model_path,
            aligned=aligned,
            conf=args.conf,
        )
    else:
        if not args.data_root or args.split is None:
            raise RuntimeError(
                "When using dataset mode, you must provide --data-root and --split "
                "(and optionally --index; if omitted, the whole split is evaluated)."
            )

        if args.index is None:
            evaluate_add_for_dataset_split(
                data_root=args.data_root,
                split=args.split,
                model_path=args.model_path,
                aligned=aligned,
                conf=args.conf,
                dev_compare_gt_2d=args.dev_compare_gt2d,
            )
        else:
            evaluate_add_for_dataset_entry(
                data_root=args.data_root,
                split=args.split,
                index=args.index,
                model_path=args.model_path,
                aligned=aligned,
                conf=args.conf,
                dev_compare_gt_2d=args.dev_compare_gt2d,
            )


if __name__ == "__main__":
    main()
