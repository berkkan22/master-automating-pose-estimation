from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np


class Keypoint:
    """
    Represents a detected keypoint with its ID and (x, y) coordinates.
    
    Attributes:
        id (int): The identifier for the keypoint
        x (float): The x-coordinate of the keypoint
        y (float): The y-coordinate of the keypoint
    """
    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y
        
    def __str__(self):
        return f"Keypoint(id={self.id}, x={self.x:.2f}, y={self.y:.2f})"
    
class BoundingBox:
    """
    Represents a detected bounding box defined by its top-left and bottom-right coordinates.
    
    Attributes:
        x1 (int): The x-coordinate of the top-left corner of the bounding box
        y1 (int): The y-coordinate of the top-left corner of the bounding box
        x2 (int): The x-coordinate of the bottom-right corner of the bounding box
        y2 (int): The y-coordinate of the bottom-right corner of the bounding box
    """
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        
    def __str__(self):
        return f"BoundingBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"

class PoseEstimationResult:
    """
    Encapsulates the results of pose estimation, including image dimensions, detected keypoints, and bounding boxes.

    Attributes:
        imageShape (Tuple[int, int]): The dimensions of the input image (height, width
        keypoints (List[Keypoint]): A list of detected keypoints
        boundingBoxes (List[BoundingBox]): A list of detected bounding boxes
    """
    def __init__(self, imageShape: Tuple[int, int], keypoints: List[Keypoint], boundingBoxes: List[BoundingBox]):
        self.imageShape = imageShape
        self.keypoints = keypoints
        self.boundingBoxes = boundingBoxes
        
    def __str__(self):
        return f"PoseEstimationResult(imageShape={self.imageShape}, keypoints=[{', '.join(str(kp) for kp in self.keypoints)}])"


class BasePoseEstimator(ABC):
    """Abstract base class for pose estimation models.

    Any concrete pose estimator (YOLO, MediaPipe, etc.) should subclass this
    and implement the abstract methods so the rest of the pipeline can rely on
    a consistent interface.
    """

    @abstractmethod
    def predict(self, image: np.ndarray) -> PoseEstimationResult:
        """Run pose estimation on a single image.

        Args:
            image: Input image as a NumPy array in HxWxC (BGR or RGB, as the
                   implementation expects).

        Returns:
            PoseEstimationResult: detected keypoints and bounding boxes
            for this image.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """A short name identifying this estimator (e.g. "yolo", "mediapipe")."""


class DepthEstimationResult:
    """Encapsulates the results of depth estimation for a single image.

    Attributes:
        imageShape (Tuple[int, int]): Dimensions of the input image (height, width).
        depth (np.ndarray): Depth map as a 2D array with shape (H, W).
        isMetric (Optional[bool]): Whether the depth values are in metric units (meters),
            if known from the underlying model.
        confidence (Optional[np.ndarray]): Optional per-pixel confidence map with
            the same spatial dimensions as ``depth``.
        intrinsics (Optional[np.ndarray]): Optional 3x3 camera intrinsics matrix K. 
        extrinsics (Optional[np.ndarray]): Optional camera extrinsics (e.g. 4x4). Position of the camera in world coordinates, if available from the underlying model.
    """

    def __init__(
        self,
        imageShape: Tuple[int, int],
        depth: np.ndarray,
        isMetric: Optional[bool] = None,
        confidence: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        extrinsics: Optional[np.ndarray] = None,
    ):
        self.imageShape = imageShape
        self.depth = depth
        self.isMetric = isMetric
        self.confidence = confidence
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def __str__(self) -> str:
        h, w = self.imageShape
        return (
            f"DepthEstimationResult(imageShape=({h}, {w}), "
            f"depth_shape={self.depth.shape}, isMetric={self.isMetric})"
        )


class BaseDepthEstimator(ABC):
    """Abstract base class for depth estimation models.

    Concrete depth estimators (Depth Anything 3, MiDaS, etc.) should subclass
    this and implement :meth:`predict` so the rest of the pipeline can rely on
    a consistent interface.
    """

    @abstractmethod
    def predict(self, image: np.ndarray) -> DepthEstimationResult:
        """Run depth estimation on a single image.

        Args:
            image: Input image as a NumPy array in HxWxC (BGR or RGB, as the
                   implementation expects).

        Returns:
            DepthEstimationResult: predicted depth map (and optionally
            confidence / metric information) for this image.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this depth estimator (e.g. "da3-large")."""


class KeypointDepth3D:
    """Stores one keypoint, its depth, and its 3D coordinates.

    Attributes:
        id: Keypoint id (e.g. from YOLO).
        x_img, y_img: 2D coordinates in the original image space.
        u_depth, v_depth: 2D coordinates in the depth-map space.
        depth: Depth value at (u_depth, v_depth).
        X, Y, Z: 3D coordinates in camera space (computed with intrinsics).
        Xw, Yw, Zw: 3D coordinates in world space (computed with extrinsics).
    """

    def __init__(
        self,
        id: int,
        x_img: float,
        y_img: float,
        u_depth: float,
        v_depth: float,
        depth: float,
    ) -> None:
        self.id = id
        self.x_img = float(x_img)
        self.y_img = float(y_img)
        self.u_depth = float(u_depth)
        self.v_depth = float(v_depth)
        self.depth = float(depth)

        self.X: Optional[float] = None
        self.Y: Optional[float] = None
        self.Z: Optional[float] = None

        self.Xw: Optional[float] = None
        self.Yw: Optional[float] = None
        self.Zw: Optional[float] = None

    def __str__(self) -> str:
        return (
            f"KeypointDepth3D(id={self.id}, x_img={self.x_img:.2f}, y_img={self.y_img:.2f}, "
            f"u_depth={self.u_depth:.2f}, v_depth={self.v_depth:.2f}, depth={self.depth:.4f}, "
            f"X={self.X}, Y={self.Y}, Z={self.Z}, Xw={self.Xw}, Yw={self.Yw}, Zw={self.Zw})"
        )


class PoseDepth3D:
    """Binds pose keypoints with a depth map and camera calibration, and computes 3D points.

    - Stores associations: each 2D keypoint (x, y) with its depth value.
    - compute_camera_coordinates(): fills X, Y, Z (camera space) for each keypoint using intrinsics.
    - compute_world_coordinates(): fills Xw, Yw, Zw (world space) using extrinsics.
    """

    def __init__(
        self,
        pose_result: PoseEstimationResult,
        depth_result: DepthEstimationResult,
    ) -> None:
        self.pose_result = pose_result
        self.depth_result = depth_result
        self.keypoints_3d: List[KeypointDepth3D] = []

    def _ensure_keypoint_depths(self) -> None:
        """Associate each 2D keypoint with a depth value from the depth map."""
        if self.keypoints_3d:
            return

        depth_map = self.depth_result.depth
        depth_h, depth_w = depth_map.shape[:2]
        img_h, img_w = self.pose_result.imageShape

        for kp in self.pose_result.keypoints:
            # Map image-space (x, y) to depth-map-space (u, v) by relative position.
            u = float(np.clip(kp.x / img_w * depth_w, 0, depth_w - 1))
            v = float(np.clip(kp.y / img_h * depth_h, 0, depth_h - 1))
            iu = int(u)
            iv = int(v)
            depth_val = float(depth_map[iv, iu])

            self.keypoints_3d.append(
                KeypointDepth3D(
                    id=kp.id,
                    x_img=kp.x,
                    y_img=kp.y,
                    u_depth=u,
                    v_depth=v,
                    depth=depth_val,
                )
            )

    def compute_camera_coordinates(self) -> List[KeypointDepth3D]:
        """Compute 3D camera-space coordinates (X, Y, Z) for each keypoint.

        Uses the intrinsics matrix K from the depth prediction, assuming
        (u_depth, v_depth) live in the same pixel space as the depth map.
        """
        self._ensure_keypoint_depths()

        K = self.depth_result.intrinsics
        if K is None:
            raise ValueError("No intrinsics available in DepthEstimationResult; cannot compute 3D.")
        if K.shape != (3, 3):
            raise ValueError(f"Expected intrinsics K with shape (3, 3), got {K.shape}")

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        for kp3d in self.keypoints_3d:
            Z = kp3d.depth
            X = (kp3d.u_depth - cx) / fx * Z
            Y = (kp3d.v_depth - cy) / fy * Z
            kp3d.X = float(X)
            kp3d.Y = float(Y)
            kp3d.Z = float(Z)

        return self.keypoints_3d

    def compute_world_coordinates(self) -> List[KeypointDepth3D]:
        """Compute world-space coordinates for each keypoint using extrinsics.

        Assumes extrinsics are a world-to-camera (w2c) matrix (3x4 or 4x4)
        as used in DA3; world coordinates are obtained via its inverse.
        """
        if not self.keypoints_3d or self.keypoints_3d[0].X is None:
            self.compute_camera_coordinates()

        T_w2c = self.depth_result.extrinsics
        if T_w2c is None:
            raise ValueError("No extrinsics available in DepthEstimationResult; cannot compute world coordinates.")

        T_w2c = np.asarray(T_w2c, dtype=float)
        if T_w2c.shape == (3, 4):
            T_tmp = np.eye(4, dtype=float)
            T_tmp[:3, :4] = T_w2c
            T_w2c = T_tmp
        elif T_w2c.shape != (4, 4):
            raise ValueError(f"Expected extrinsics with shape (3,4) or (4,4), got {T_w2c.shape}")

        T_c2w = np.linalg.inv(T_w2c)

        for kp3d in self.keypoints_3d:
            if kp3d.X is None:
                continue
            cam_pt = np.array([kp3d.X, kp3d.Y, kp3d.Z, 1.0], dtype=float)
            world_pt = T_c2w @ cam_pt
            kp3d.Xw = float(world_pt[0])
            kp3d.Yw = float(world_pt[1])
            kp3d.Zw = float(world_pt[2])

        return self.keypoints_3d

