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
    """

    def __init__(
        self,
        imageShape: Tuple[int, int],
        depth: np.ndarray,
        isMetric: Optional[bool] = None,
        confidence: Optional[np.ndarray] = None,
    ):
        self.imageShape = imageShape
        self.depth = depth
        self.isMetric = isMetric
        self.confidence = confidence

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

