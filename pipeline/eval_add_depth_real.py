import os
import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

from models import PoseEstimationResult, Keypoint, PoseDepth3D
from yolo_estimator import YoloPoseEstimator
from depth_estimator import DepthAnything3Estimator


# Root of this repository (parent of the "pipeline" folder)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Hard-coded manual corner annotations provided by the user.
# Expected line format (comma-separated):
#   cornerName, x_px, y_px, imageFilename, imageWidth_px, imageHeight_px
# Occluded corners (name ending with "o", e.g. "Ao", "Eo") are ignored.
ANNOTATIONS_TEXT = """
cornerB,636,227,08-wache-1.jpg,1140,2023
cornerAo,912,180,08-wache-1.jpg,1140,2023
cornerC,642,1224,08-wache-1.jpg,1140,2023
cornerDo,921,1233,08-wache-1.jpg,1140,2023
cornerEo,1002,60,08-wache-1.jpg,1140,2023
cornerFo,1130,24,08-wache-1.jpg,1140,2023
cornerHo,1025,1493,08-wache-1.jpg,1140,2023
cornerGo,1139,1511,08-wache-1.jpg,1140,2023
cornerE,723,130,20240219_134332_00_00_30_567.jpg,1920,1080
cornerF,1142,28,20240219_134332_00_00_30_567.jpg,1920,1080
cornerG,1160,723,20240219_134332_00_00_30_567.jpg,1920,1080
cornerHo,732,667,20240219_134332_00_00_30_567.jpg,1920,1080
cornerA,1796,334,20240219_134332_00_00_30_567.jpg,1920,1080
cornerBo,1608,336,20240219_134332_00_00_30_567.jpg,1920,1080
cornerCo,1608,624,20240219_134332_00_00_30_567.jpg,1920,1080
cornerD,1812,628,20240219_134332_00_00_30_567.jpg,1920,1080
cornerA,2344,768,20240503_123845.jpg,4624,2084
cornerB,2671,859,20240503_123845.jpg,4624,2084
cornerC,2697,1376,20240503_123845.jpg,4624,2084
cornerD,2338,1324,20240503_123845.jpg,4624,2084
cornerF,1326,1220,20240503_123845.jpg,4624,2084
cornerEo,1606,1199,20240503_123845.jpg,4624,2084
cornerH,1580,1604,20240503_123845.jpg,4624,2084
cornerG,1295,1581,20240503_123845.jpg,4624,2084
cornerE,3669,1377,20240503_123845.jpg,4624,2084
cornerF,4058,1314,20240503_123845.jpg,4624,2084
cornerH,3724,1845,20240503_123845.jpg,4624,2084
cornerG,4138,1830,20240503_123845.jpg,4624,2084
cornerAo,4617,1474,20240503_123845.jpg,4624,2084
cornerDo,4616,1789,20240503_123845.jpg,4624,2084
cornerBo,4539,1474,20240503_123845.jpg,4624,2084
cornerCo,4557,1791,20240503_123845.jpg,4624,2084
cornerE,3722,1865,20240503_123845.jpg,4624,2084
cornerF,4142,1851,20240503_123845.jpg,4624,2084
cornerHo,3746,2076,20240503_123845.jpg,4624,2084
cornerGo,4170,2078,20240503_123845.jpg,4624,2084
cornerE,4166,1853,20240503_123845.jpg,4624,2084
cornerFo,4599,1819,20240503_123845.jpg,4624,2084
cornerHo,4215,2076,20240503_123845.jpg,4624,2084
cornerGo,4612,2072,20240503_123845.jpg,4624,2084
cornerBo,4058,1865,20240503_123845.jpg,4624,2084
cornerAo,4125,1867,20240503_123845.jpg,4624,2084
cornerDo,4087,2076,20240503_123845.jpg,4624,2084
cornerCo,4143,2078,20240503_123845.jpg,4624,2084
cornerBo,4557,1849,20240503_123845.jpg,4624,2084
cornerAo,4614,1849,20240503_123845.jpg,4624,2084
cornerDo,4568,2050,20240503_123845.jpg,4624,2084
cornerCo,4619,2050,20240503_123845.jpg,4624,2084
cornerEo,651,332,20240601_152109.jpg,4000,1800
cornerF,544,214,20240601_152109.jpg,4000,1800
cornerG,672,1559,20240601_152109.jpg,4000,1800
cornerHo,743,1417,20240601_152109.jpg,4000,1800
cornerBo,2654,495,20240601_152109.jpg,4000,1800
cornerA,2806,445,20240601_152109.jpg,4000,1800
cornerD,2787,1166,20240601_152109.jpg,4000,1800
cornerCo,2640,1132,20240601_152109.jpg,4000,1800
cornerA,3132,693,20241009_150318.jpg,4000,2252
cornerB,3147,702,20241009_150318.jpg,4000,2252
cornerC,3150,1267,20241009_150318.jpg,4000,2252
cornerD,3130,1293,20241009_150318.jpg,4000,2252
cornerEo,320,722,20241009_150318.jpg,4000,2252
cornerHo,322,1273,20241009_150318.jpg,4000,2252
cornerF,201,704,20241009_150318.jpg,4000,2252
cornerG,205,1341,20241009_150318.jpg,4000,2252
cornerF,102,881,20241009_150318.jpg,4000,2252
cornerG,105,1254,20241009_150318.jpg,4000,2252
cornerEo,156,902,20241009_150318.jpg,4000,2252
cornerHo,154,1230,20241009_150318.jpg,4000,2252
cornerAo,178,881,20241009_150318.jpg,4000,2252
cornerBo,196,890,20241009_150318.jpg,4000,2252
cornerDo,177,1255,20241009_150318.jpg,4000,2252
cornerCo,194,1243,20241009_150318.jpg,4000,2252
cornerEo,3,1131,20241009_150318.jpg,4000,2252
cornerFo,21,1131,20241009_150318.jpg,4000,2252
cornerGo,21,1181,20241009_150318.jpg,4000,2252
cornerHo,1,1182,20241009_150318.jpg,4000,2252
cornerBo,76,1131,20241009_150318.jpg,4000,2252
cornerAo,97,1127,20241009_150318.jpg,4000,2252
cornerCo,78,1180,20241009_150318.jpg,4000,2252
cornerDo,98,1186,20241009_150318.jpg,4000,2252
cornerA,1473,933,20241009_213713.jpg,4000,2252
cornerB,2407,843,20241009_213713.jpg,4000,2252
cornerC,2454,2060,20241009_213713.jpg,4000,2252
cornerD,1416,2022,20241009_213713.jpg,4000,2252
cornerE,2812,1423,20241009_213713.jpg,4000,2252
cornerH,2841,1801,20241009_213713.jpg,4000,2252
cornerA,114,1513,20241009_213713.jpg,4000,2252
cornerB,317,1513,20241009_213713.jpg,4000,2252
cornerCo,286,1703,20241009_213713.jpg,4000,2252
cornerDo,102,1704,20241009_213713.jpg,4000,2252
cornerE,405,1538,20241009_213713.jpg,4000,2252
cornerHo,373,1694,20241009_213713.jpg,4000,2252
cornerFo,265,1541,20241009_213713.jpg,4000,2252
cornerGo,252,1686,20241009_213713.jpg,4000,2252
cornerFo,2359,1409,20241009_213713.jpg,4000,2252
cornerGo,2392,1857,20241009_213713.jpg,4000,2252
cornerE,1385,575,20241010_083526.jpg,4000,2252
cornerF,2119,566,20241010_083526.jpg,4000,2252
cornerG,2109,1408,20241010_083526.jpg,4000,2252
cornerH,1383,1413,20241010_083526.jpg,4000,2252
cornerA,2150,981,20241010_083526.jpg,4000,2252
cornerD,2150,1280,20241010_083526.jpg,4000,2252
cornerBo,1708,976,20241010_083526.jpg,4000,2252
cornerCo,1711,1302,20241010_083526.jpg,4000,2252
cornerA,1362,823,20241223_180148.jpg,4000,2252
cornerB,1663,777,20241223_180148.jpg,4000,2252
cornerC,1659,1353,20241223_180148.jpg,4000,2252
cornerD,1351,1345,20241223_180148.jpg,4000,2252
cornerE,2710,966,20241223_180148.jpg,4000,2252
cornerFo,2523,981,20241223_180148.jpg,4000,2252
cornerH,2717,1294,20241223_180148.jpg,4000,2252
cornerGo,2519,1282,20241223_180148.jpg,4000,2252
cornerA,1754,816,20241223_180432.jpg,4000,2252
cornerB,2082,739,20241223_180432.jpg,4000,2252
cornerD,1740,1360,20241223_180432.jpg,4000,2252
cornerC,2066,1362,20241223_180432.jpg,4000,2252
cornerE,3232,1020,20241223_180432.jpg,4000,2252
cornerFo,3057,1038,20241223_180432.jpg,4000,2252
cornerGo,3055,1370,20241223_180432.jpg,4000,2252
cornerH,3234,1400,20241223_180432.jpg,4000,2252
cornerE,1559,1523,20241226_133012.jpg,4000,2252
cornerF,2211,1431,20241226_133012.jpg,4000,2252
cornerG,2217,2229,20241226_133012.jpg,4000,2252
cornerHo,1582,2248,20241226_133012.jpg,4000,2252
cornerA,927,922,20241226_133012.jpg,4000,2252
cornerB,452,989,20241226_133012.jpg,4000,2252
cornerC,491,1574,20241226_133012.jpg,4000,2252
cornerDo,963,1538,20241226_133012.jpg,4000,2252
"""

# Folder with real-world images (trudi_ds data root)
TRUDI_DATA_DIR = os.path.join(_REPO_ROOT, "trudi_ds", "data")

# Default YOLO pose weights used in this project
YOLO_MODEL_PATH = os.path.join(_REPO_ROOT, "yolo11n-pose.pt")


def _parse_annotation_line(line: str) -> Tuple[str, float, float, str, int, int] | None:
    """Parse a single annotation line.

    Returns (corner_name, x, y, image_filename, width, height) or None if invalid.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # Strip optional markdown table pipes, e.g. "| cornerA, ... |"
    if line.startswith("|") and line.endswith("|"):
        line = line[1:-1].strip()
        if not line:
            return None

    # Skip separator rows like "| --- |"
    if set(line) <= {"-", " ", "|"}:
        return None

    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 6:
        return None

    name = parts[0]
    try:
        x = float(parts[1])
        y = float(parts[2])
        img_name = parts[3]
        w = int(float(parts[4]))
        h = int(float(parts[5]))
    except ValueError:
        return None

    # Ignore occluded corners (e.g. Ao, Bo, Eo, Fo, Go, Ho)
    if name.endswith("o"):
        return None

    return name, x, y, img_name, w, h


def load_manual_keypoints() -> Dict[str, List[Tuple[float, float]]]:
    """Load 2D GT keypoints from the manual annotation txt.

    Returns a dict mapping image filename -> list of (x, y) keypoints in pixels.
    Occluded corners (name ending with 'o') are discarded.
    """
    per_image: Dict[str, List[Tuple[float, float]]] = {}

    for raw_line in ANNOTATIONS_TEXT.strip().splitlines():
        parsed = _parse_annotation_line(raw_line)
        if parsed is None:
            continue
        _name, x, y, img_name, _w, _h = parsed
        per_image.setdefault(img_name, []).append((x, y))

    return per_image


def _compute_3d_for_single_point(
    x: float,
    y: float,
    image_shape: Tuple[int, int],
    depth_result,
) -> Tuple[float, float, float]:
    """Compute 3D camera-space coordinates for a single 2D point using DA3.

    This wraps PoseDepth3D for convenience.
    """
    pose_res = PoseEstimationResult(imageShape=image_shape, keypoints=[Keypoint(id=0, x=x, y=y)], boundingBoxes=[])
    frame = PoseDepth3D(pose_res, depth_result)
    kps3d = frame.compute_camera_coordinates()
    kp3d = kps3d[0]
    if kp3d.X is None or kp3d.Y is None or kp3d.Z is None:
        raise RuntimeError("3D coordinates not available for keypoint.")
    return float(kp3d.X), float(kp3d.Y), float(kp3d.Z)


def evaluate_real_images() -> None:
    """Run pose+depth on real images and compare to manual 2D keypoints.

    For each image listed in ANNOTATION_TXT:
      - Run YOLO pose to get predicted 2D keypoints.
      - Run Depth Anything 3 to get depth + intrinsics.
      - For every GT keypoint (x_gt, y_gt):
          * Find the nearest predicted keypoint (x_pred, y_pred) in 2D.
          * Accumulate 2D distance d2 = ||(x_pred, y_pred) - (x_gt, y_gt)||.
          * Compute 3D camera-space coordinates for both points using the same
            depth map and intrinsics, and accumulate the 3D distance d3.

    Prints per-image mean 2D and 3D distances, plus global statistics.
    """
    gt_by_image = load_manual_keypoints()
    if not gt_by_image:
        raise RuntimeError("No valid GT keypoints loaded from annotation file.")

    if not os.path.isfile(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")

    pose_estimator = YoloPoseEstimator(model_path=YOLO_MODEL_PATH, conf=0.5)
    depth_estimator = DepthAnything3Estimator(output_dir=None)

    global_d2_list: List[float] = []
    global_d3_list: List[float] = []

    for img_name, gt_points in gt_by_image.items():
        img_path = os.path.join(TRUDI_DATA_DIR, img_name)
        if not os.path.isfile(img_path):
            print(f"[WARN] Image not found for annotations: {img_path}, skipping.")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Failed to read image: {img_path}, skipping.")
            continue

        H, W = image.shape[:2]

        # Depth prediction (once per image)
        try:
            depth_result = depth_estimator.predict(image)
        except Exception as e:
            print(f"[WARN] Depth estimation failed for {img_name}: {e}")
            continue

        if depth_result.intrinsics is None:
            print(f"[WARN] No intrinsics from depth estimator for {img_name}, skipping 3D metrics.")

        # Pose prediction (once per image)
        try:
            pose_res = pose_estimator.predict(image)
        except Exception as e:
            print(f"[WARN] Pose estimation failed for {img_name}: {e}")
            continue

        if not pose_res.keypoints:
            print(f"[INFO] No predicted keypoints for {img_name}, skipping.")
            continue

        # Precompute 3D coordinates for all predicted keypoints (if possible)
        pred_kps = pose_res.keypoints
        pred_coords_2d = np.array([[kp.x, kp.y] for kp in pred_kps], dtype=np.float32)
        pred_kps3d: List[Tuple[float, float, float]] | None = None
        if depth_result.intrinsics is not None:
            try:
                frame_pred = PoseDepth3D(pose_res, depth_result)
                kps3d_full = frame_pred.compute_camera_coordinates()
                pred_kps3d = [
                    (float(k.X), float(k.Y), float(k.Z))
                    if (k.X is not None and k.Y is not None and k.Z is not None)
                    else None
                    for k in kps3d_full
                ]
            except Exception as e:
                print(f"[WARN] Failed to compute 3D for predicted keypoints in {img_name}: {e}")
                pred_kps3d = None

        img_d2_list: List[float] = []
        img_d3_list: List[float] = []

        for (x_gt, y_gt) in gt_points:
            gt_pt = np.array([x_gt, y_gt], dtype=np.float32)
            # Find nearest predicted keypoint in 2D
            diffs = pred_coords_2d - gt_pt[None, :]
            dists2 = np.sum(diffs ** 2, axis=1)
            best_idx = int(np.argmin(dists2))
            best_d2 = float(math.sqrt(float(dists2[best_idx])))
            img_d2_list.append(best_d2)
            global_d2_list.append(best_d2)

            # 3D distance using same depth map (if available)
            if depth_result.intrinsics is not None and pred_kps3d is not None:
                pred_3d = pred_kps3d[best_idx]
                if pred_3d is not None:
                    try:
                        Xg, Yg, Zg = _compute_3d_for_single_point(x_gt, y_gt, (H, W), depth_result)
                        Xp, Yp, Zp = pred_3d
                        d3 = math.sqrt((Xp - Xg) ** 2 + (Yp - Yg) ** 2 + (Zp - Zg) ** 2)
                        img_d3_list.append(float(d3))
                        global_d3_list.append(float(d3))
                    except Exception as e:
                        print(f"[WARN] 3D computation failed for GT point in {img_name}: {e}")

        if img_d2_list:
            mean_d2 = float(np.mean(img_d2_list))
            std_d2 = float(np.std(img_d2_list))
            if img_d3_list:
                mean_d3 = float(np.mean(img_d3_list))
                std_d3 = float(np.std(img_d3_list))
                print(
                    f"Image {img_name}: N_gt={len(img_d2_list)}, "
                    f"mean_2D={mean_d2:.3f} px (std={std_d2:.3f}), "
                    f"mean_3D={mean_d3:.6f} (std={std_d3:.6f})"
                )
            else:
                print(
                    f"Image {img_name}: N_gt={len(img_d2_list)}, "
                    f"mean_2D={mean_d2:.3f} px (std={std_d2:.3f}), 3D metrics unavailable"
                )
        else:
            print(f"[INFO] No valid GT keypoints used for {img_name}.")

    # Global summary
    if global_d2_list:
        g_mean_d2 = float(np.mean(global_d2_list))
        g_std_d2 = float(np.std(global_d2_list))
        print("\n=== Global 2D keypoint distance summary ===")
        print(f"N_total={len(global_d2_list)}, mean_2D={g_mean_d2:.3f} px, std_2D={g_std_d2:.3f} px")

    if global_d3_list:
        g_mean_d3 = float(np.mean(global_d3_list))
        g_std_d3 = float(np.std(global_d3_list))
        print("\n=== Global 3D (ADD-like) distance summary ===")
        print(f"N_total={len(global_d3_list)}, mean_3D={g_mean_d3:.6f}, std_3D={g_std_d3:.6f}")


if __name__ == "__main__":
    evaluate_real_images()
