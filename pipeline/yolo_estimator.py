from typing import List

import numpy as np
from ultralytics import YOLO

from models import BasePoseEstimator, PoseEstimationResult, Keypoint, BoundingBox


class YoloPoseEstimator(BasePoseEstimator):
    """Concrete pose estimator using a YOLO pose model.

    This wraps the ultralytics YOLO model and converts its outputs into
    PoseEstimationResult so the rest of the pipeline can stay model-agnostic.
    """

    def __init__(self, model_path: str, conf: float = 0.5, verbose: bool = False):
        self._model = YOLO(model_path)
        self._conf = conf
        self._verbose = verbose

    @property
    def name(self) -> str:
        return "yolo"

    def predict(self, image: np.ndarray) -> PoseEstimationResult:
        """
        Run YOLO pose on a single image and return a PoseEstimationResult.
        """
        H, W = image.shape[:2]
        result = self._model([image], verbose=self._verbose, conf=self._conf)[0]

        keypoints: List[Keypoint] = []
        boxes: List[BoundingBox] = []

        if result.keypoints is not None:
            keypoints_raw = result.keypoints.xy.cpu().numpy()  # (num_dets, num_kpts, 2)
            for det_idx, det_keypoints in enumerate(keypoints_raw):
                for keypoint_index, (x, y) in enumerate(det_keypoints):
                    keypoints.append(Keypoint(id=keypoint_index, x=float(x), y=float(y)))

        if result.boxes is not None:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))

        return PoseEstimationResult(imageShape=(H, W), keypoints=keypoints, boundingBoxes=boxes)
