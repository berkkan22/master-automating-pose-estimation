from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3

from models import BaseDepthEstimator, DepthEstimationResult


class DepthAnything3Estimator(BaseDepthEstimator):
    """Depth estimator wrapping Depth Anything 3.

    This class adapts the Depth Anything 3 API to the pipeline's
    :class:`BaseDepthEstimator` interface, so it can be used interchangeably
    with other depth backends.
    """

    def __init__(
        self,
        model_name: str = "da3-large",
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ) -> None:
        """Create a Depth Anything 3 estimator.

        Args:
            model_name: Model preset name, e.g. "da3-large", "da3-giant",
                "da3metric-large", etc.
            device: Optional device string ("cuda", "cpu", ...). If None,
                the device is chosen automatically.
            process_res: Processing resolution passed to the DA3 API.
            process_res_method: Resize method for DA3 preprocessing.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self._device_str = device
        self._device = torch.device(device)
        self._output_dir = output_dir
        self._process_res = process_res
        self._process_res_method = process_res_method

        # Initialize underlying Depth Anything 3 model
        self._model = DepthAnything3(model_name=model_name)
        self._model.to(self._device)
        # Ensure the API knows which device to use
        self._model.device = self._device

    @property
    def name(self) -> str:
        return f"depth-anything3-{self._model.model_name}"

    def predict(self, image: np.ndarray) -> DepthEstimationResult:
        """Run Depth Anything 3 on a single image.

        Args:
            image: Input image as HxWxC NumPy array (BGR or RGB). The DA3
                input processor can handle typical OpenCV/PIL formats.

        Returns:
            DepthEstimationResult with the predicted depth map.
        """
        if image is None:
            raise ValueError("image must be a NumPy array, got None")

        if image.ndim != 3:
            raise ValueError(f"Expected image with 3 dimensions (H, W, C), got shape {image.shape}")

        H, W = image.shape[:2]

        # Depth Anything 3 API expects a list of images
        prediction = self._model.inference(
            image=[image],
            process_res=self._process_res,
            process_res_method=self._process_res_method,
            export_dir=self._output_dir,
            export_format="glb",
            infer_gs=False,
            use_ray_pose=False,
        )

        depth_batch = prediction.depth  # shape (N, H, W)
        if depth_batch.ndim != 3 or depth_batch.shape[0] == 0:
            raise RuntimeError(f"Unexpected depth shape from model: {depth_batch.shape}")

        depth_map = depth_batch[0].astype(np.float32, copy=False)

        # Optional confidence map
        confidence_map = None
        if getattr(prediction, "conf", None) is not None:
            conf_batch = prediction.conf
            if conf_batch is not None and conf_batch.ndim == 3 and conf_batch.shape[0] > 0:
                confidence_map = conf_batch[0]

        is_metric = None
        if hasattr(prediction, "is_metric"):
            try:
                is_metric = bool(prediction.is_metric)
            except Exception:
                is_metric = None

        return DepthEstimationResult(
            imageShape=(H, W),
            depth=depth_map,
            isMetric=is_metric,
            confidence=confidence_map,
        )
