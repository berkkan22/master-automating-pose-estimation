import os
import argparse

import cv2
import numpy as np

from models import PoseEstimationResult, PoseDepth3D
from yolo_estimator import YoloPoseEstimator
from depth_estimator import DepthAnything3Estimator


# Configuration
BASE_PATH = "/data/9katirci/master-automating-pose-estimation"
MODEL_PATH = f"{BASE_PATH}/runs/pose/pose_estimation_synthetic_data_new_img640_x_1000/weights/best.pt"
# IMAGE = f"{BASE_PATH}/datasets/trudi_ds_yolo11_instand_segmentation/test/images/20240503_124036.jpg"
IMAGE = f"{BASE_PATH}/datasets/synthetic_data_yolo11_pose_new/images/test/49.jpg"
OUT_DIR = f"{BASE_PATH}/temp_results"


# Distinct colors (BGR) for different containers/detections
CONTAINER_COLORS = [
    (0, 0, 255),    # red
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]


def _load_gt_keypoints_for_image(image_path: str, image_shape: tuple[int, int]) -> list[list[tuple[float, float]]]:
    """Load ground-truth keypoints from YOLO pose labels for a single image.

    Expects labels in Ultralytics YOLO pose format with visibility:
      class cx cy w h x1 y1 v1 ... x8 y8 v8

    The label file is resolved by replacing '/images/' with '/labels/' and
    changing the extension to .txt.

    Returns a list of containers, each container being a list of (x, y) pixels
    for its visible keypoints.
    """
    h, w = image_shape

    if "/images/" not in image_path:
        return []

    label_path = image_path.replace("/images/", "/labels/")
    label_path = os.path.splitext(label_path)[0] + ".txt"

    if not os.path.exists(label_path):
        return []

    containers: list[list[tuple[float, float]]] = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5 + 3 * 8:
                    continue
                # keypoints start at index 5
                kps: list[tuple[float, float]] = []
                for i in range(8):
                    try:
                        x_n = float(parts[5 + 3 * i])
                        y_n = float(parts[5 + 3 * i + 1])
                        v = float(parts[5 + 3 * i + 2])
                    except (ValueError, IndexError):
                        break
                    if v <= 0:
                        continue
                    x_px = x_n * w
                    y_px = y_n * h
                    kps.append((x_px, y_px))
                if kps:
                    containers.append(kps)
    except OSError:
        return []

    return containers


def draw_keypoints(image: np.ndarray, result: PoseEstimationResult) -> np.ndarray:
    """Return a copy of image with keypoints drawn."""
    vis = image.copy()
    num_boxes = len(result.boundingBoxes)
    num_kpts = len(result.keypoints)
    kpts_per_box = num_kpts // num_boxes if num_boxes > 0 else 0

    for idx, kp in enumerate(result.keypoints):
        if kpts_per_box > 0 and num_boxes > 0:
            container_idx = min(idx // kpts_per_box, num_boxes - 1)
        else:
            container_idx = 0
        color = CONTAINER_COLORS[container_idx % len(CONTAINER_COLORS)]
        cv2.circle(vis, (int(kp.x), int(kp.y)), radius=10, color=color, thickness=-1)
    return vis


def draw_bounding_boxes(image: np.ndarray, result: PoseEstimationResult) -> np.ndarray:
    """Return a copy of image with bounding boxes drawn."""
    vis = image.copy()
    for box in result.boundingBoxes:
        cv2.rectangle(vis, (box.x1, box.y1), (box.x2, box.y2), color=(255, 0, 0), thickness=6)
    return vis


def overlay_gt_keypoints(
    image: np.ndarray,
    gt_keypoints_per_container: list[list[tuple[float, float]]],
) -> np.ndarray:
    """Draw GT keypoints as squares in the same color as each container.

    - Uses CONTAINER_COLORS indexed by container index.
    - Draws a small square for each GT keypoint.
    """
    vis = image.copy()
    for container_idx, kps in enumerate(gt_keypoints_per_container):
        color = CONTAINER_COLORS[container_idx % len(CONTAINER_COLORS)]
        for (x, y) in kps:
            x_i, y_i = int(x), int(y)
            cv2.rectangle(vis, (x_i - 6, y_i - 6), (x_i + 6, y_i + 6), color, thickness=2)
    return vis

def draw_keypoints_with_3d_coords(
    image: np.ndarray,
    frame_3d: PoseDepth3D,
    gt_keypoints_per_container: list[list[tuple[float, float]]] | None = None,
) -> np.ndarray:
    """Return a copy of image with keypoints drawn and annotated with 3D coordinates."""
    vis = image.copy()

    # Draw the pinhole camera center (principal point) in red, if intrinsics are available.
    K = frame_3d.depth_result.intrinsics
    if K is not None and K.shape == (3, 3):
        depth_h, depth_w = frame_3d.depth_result.depth.shape[:2]
        img_h, img_w = image.shape[:2]

        cx_depth = K[0, 2]
        cy_depth = K[1, 2]

        # Map principal point from depth-map coordinates to original image coordinates.
        cx_img = int(cx_depth / depth_w * img_w)
        cy_img = int(cy_depth / depth_h * img_h)

        cv2.circle(vis, (cx_img, cy_img), radius=10, color=(0, 0, 0), thickness=-1)
        cv2.putText(vis, "principal point", (cx_img + 5, cy_img - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

    num_boxes = len(frame_3d.pose_result.boundingBoxes)
    num_kpts_3d = len(frame_3d.keypoints_3d)
    kpts_per_box = num_kpts_3d // num_boxes if num_boxes > 0 else 0

    for idx, kp3d in enumerate(frame_3d.keypoints_3d):
        if kpts_per_box > 0 and num_boxes > 0:
            container_idx = min(idx // kpts_per_box, num_boxes - 1)
        else:
            container_idx = 0
        color = CONTAINER_COLORS[container_idx % len(CONTAINER_COLORS)]

        x_img, y_img = int(kp3d.x_img), int(kp3d.y_img)
        cv2.circle(vis, (x_img, y_img), radius=10, color=color, thickness=-1)
        text = f"({kp3d.X:.2f}, {kp3d.Y:.2f}, {kp3d.Z:.2f})"
        cv2.putText(vis, text, (x_img + 5, y_img - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

    # Optionally overlay GT keypoints as colored squares
    if gt_keypoints_per_container:
        vis = overlay_gt_keypoints(vis, gt_keypoints_per_container)
    return vis


def run_pipeline(image_path: str, model_path: str, out_dir: str, dev: bool = False) -> PoseDepth3D:
    os.makedirs(out_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # ! 1. Pose estimation
    estimator = YoloPoseEstimator(model_path)
    result = estimator.predict(image)

    if not result.keypoints:
        print("No keypoints detected.")
        return result

    # Optionally load ground-truth keypoints (for dev visualization)
    gt_keypoints_per_container: list[list[tuple[float, float]]] = []
    if dev:
        gt_keypoints_per_container = _load_gt_keypoints_for_image(image_path, image.shape[:2])

    # Visualize and save
    keypoint_visualization = draw_keypoints(image, result)
    if dev and gt_keypoints_per_container:
        keypoint_visualization = overlay_gt_keypoints(keypoint_visualization, gt_keypoints_per_container)
    boxes_visualization = draw_bounding_boxes(image, result)

    keypoint_path = os.path.join(out_dir, "keypoints_visualization.jpg")
    boxes_path = os.path.join(out_dir, "bounding_boxes_visualization.jpg")

    cv2.imwrite(keypoint_path, keypoint_visualization)
    cv2.imwrite(boxes_path, boxes_visualization)

    print(f"Keypoints visualization saved to: {keypoint_path}")
    print(f"Bounding boxes visualization saved to: {boxes_path}")

    # ! 2. Depth estimation
    depth_estimator = DepthAnything3Estimator(output_dir=OUT_DIR)
    depth_result = depth_estimator.predict(image)
    depth_map = depth_result.depth
    print(f"Depth map shape: {depth_map.shape}, is_metric: {depth_result.isMetric}")

    # Combine pose keypoints with depth, then compute 3D camera and world coordinates.
    frame_3d = PoseDepth3D(result, depth_result)

    try:
        kps_cam = frame_3d.compute_camera_coordinates()
    except ValueError as e:
        print(f"\n[Warning] Could not compute camera-space 3D points: {e}")
        kps_cam = frame_3d.keypoints_3d

    try:
        kps_world = frame_3d.compute_world_coordinates()
    except ValueError as e:
        print(f"[Warning] Could not compute world-space coordinates: {e}")
        kps_world = kps_cam

    # print("\nKeypoints with depth, camera-space 3D and world-space 3D coordinates:")
    # for kp3d in kps_world:
    #     print(
    #         f"id={kp3d.id:2d} img=({kp3d.x_img:8.2f},{kp3d.y_img:8.2f}) "
    #         f"depth={kp3d.depth:.6f} cam=({kp3d.X},{kp3d.Y},{kp3d.Z}) "
    #         f"world=({kp3d.Xw},{kp3d.Yw},{kp3d.Zw})"
    #     )    
    
    
    print("\nPipeline completed successfully.")
    print(str(frame_3d.keypoints_3d[0]))
    
    d_keypoints_visualized = draw_keypoints_with_3d_coords(image, frame_3d, gt_keypoints_per_container if dev else None)
    depth_kp_vis_path = os.path.join(out_dir, "keypoints_with_3d_coords.jpg")
    cv2.imwrite(depth_kp_vis_path, d_keypoints_visualized)
    print(f"Keypoints with 3D coordinates visualization saved to: {depth_kp_vis_path}")
    
    return frame_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pose + depth pipeline.")
    parser.add_argument("--dev", action="store_true", help="Overlay ground-truth keypoints as colored squares.")
    args = parser.parse_args()

    run_pipeline(IMAGE, MODEL_PATH, OUT_DIR, dev=args.dev)


if __name__ == "__main__":
    main()
    