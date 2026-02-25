import os

import cv2
import numpy as np

from models import PoseEstimationResult, PoseDepth3D
from yolo_estimator import YoloPoseEstimator
from depth_estimator import DepthAnything3Estimator


# Configuration
BASE_PATH = "/data/9katirci/master-automating-pose-estimation"
MODEL_PATH = f"{BASE_PATH}/runs/pose/train2/weights/best.pt"
IMAGE = f"{BASE_PATH}/DJI_20230823160823_0063_D.jpg"
OUT_DIR = f"{BASE_PATH}/temp_results"


def draw_keypoints(image: np.ndarray, result: PoseEstimationResult) -> np.ndarray:
    """Return a copy of image with keypoints drawn."""
    vis = image.copy()
    for kp in result.keypoints:
        cv2.circle(vis, (int(kp.x), int(kp.y)), radius=5, color=(0, 255, 0), thickness=-1)
    return vis


def draw_bounding_boxes(image: np.ndarray, result: PoseEstimationResult) -> np.ndarray:
    """Return a copy of image with bounding boxes drawn."""
    vis = image.copy()
    for box in result.boundingBoxes:
        cv2.rectangle(vis, (box.x1, box.y1), (box.x2, box.y2), color=(255, 0, 0), thickness=2)
    return vis

# def draw_


def run_pipeline(image_path: str, model_path: str, out_dir: str) -> PoseEstimationResult:
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

    # Visualize and save
    keypoint_visualization = draw_keypoints(image, result)
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

    print("\nKeypoints with depth, camera-space 3D and world-space 3D coordinates:")
    for kp3d in kps_world:
        print(
            f"id={kp3d.id:2d} img=({kp3d.x_img:8.2f},{kp3d.y_img:8.2f}) "
            f"depth={kp3d.depth:.6f} cam=({kp3d.X},{kp3d.Y},{kp3d.Z}) "
            f"world=({kp3d.Xw},{kp3d.Yw},{kp3d.Zw})"
        )    
    
    
    print("\nPipeline completed successfully.")
    print(str(frame_3d.keypoints_3d[0]))
    
    return frame_3d


def main() -> None:
    run_pipeline(IMAGE, MODEL_PATH, OUT_DIR)


if __name__ == "__main__":
    main()
    