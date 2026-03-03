# Pose + Depth 3D Pipeline

This folder contains a small pipeline that combines YOLO pose estimation with Depth Anything 3 depth estimation to get 3D keypoints from a single RGB image.

The high‑level idea:

![Pipeline overview](Pipeline_overview.svg)

For more details:

![Detailed pipeline](Pipeline_detailed_overview.svg)

## What the Pipeline Does

- Runs a YOLO pose model on an input image (2D keypoints + bounding boxes).
- Runs Depth Anything 3 on the same image to get a dense depth map (and camera intrinsics/extrinsics when available).
- Samples depth at each keypoint and projects it to 3D camera coordinates; optionally converts to world coordinates.
- Saves visualizations to `pipeline/output` and `temp_results` (keypoints, bounding boxes, keypoints with 3D coordinates).

Key files:
- `pipeline.py` – main script and `run_pipeline` function.
- `yolo_estimator.py` – YOLO pose wrapper.
- `depth_estimator.py` – Depth Anything 3 wrapper.
- `models.py` – shared data structures (keypoints, depth results, 3D projection).

## Requirements

- OS: Linux (tested on Ubuntu‑like)
- Python: 3.11.2 recommended
- GPU: CUDA‑capable GPU strongly recommended for YOLO and Depth Anything 3
- Disk/Network: ability to download models from Hugging Face and Ultralytics if not cached

## Quick Setup (Recommended)

From the repo root (`master-automating-pose-estimation`):

```bash
# 1) Create and activate a virtual env
python3 -m venv .venv
source .venv/bin/activate

# 2) Upgrade pip
pip install --upgrade pip

# 3) Install main dependencies
pip install -r requirements.txt

# 4) Clone and install Depth Anything 3 (used by this pipeline)
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 depth-anything-3
cd depth-anything-3
pip install -e .

# 5) (Optional) Clone and install Depth-Anything-V2
git clone https://github.com/DepthAnything/Depth-Anything-V2 Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
# Download the checkpoints listed here and put them under the checkpoints directory.


# 6) (Optional) Clone and install Depth Pro (ml-depth-pro)
git clone https://github.com/apple/ml-depth-pro ml-depth-pro
cd ml-depth-pro
pip install -e .
```

If you use CUDA, ensure the correct PyTorch with GPU support is installed (see pytorch.org for the exact command for your system).

## Minimal Configuration

The default demo in `pipeline.py` assumes:

- A trained YOLO pose model at:
	- `runs/pose/train2/weights/best.pt`
- A test image at:
	- `datasets/trudi_ds_yolo11_instand_segmentation/test/images/20240503_124036.jpg`

You can change these at the top of `pipeline.py`:

```python
BASE_PATH = "/data/9katirci/master-automating-pose-estimation"
MODEL_PATH = f"{BASE_PATH}/runs/pose/train2/weights/best.pt"
IMAGE = f"{BASE_PATH}/datasets/trudi_ds_yolo11_instand_segmentation/test/images/20240503_124036.jpg"
OUT_DIR = f"{BASE_PATH}/temp_results"
```

Adjust `BASE_PATH`, `MODEL_PATH`, and `IMAGE` to match your environment and data.

## How to Run the Pipeline

From the repo root, with the virtual env activated:

```bash
cd pipeline
python pipeline.py
```

Outputs (by default):

- `temp_results/keypoints_visualization.jpg` – 2D keypoints overlay
- `temp_results/bounding_boxes_visualization.jpg` – bounding boxes overlay
- `temp_results/keypoints_with_3d_coords.jpg` – 2D keypoints annotated with 3D coordinates
- Depth‑Anything‑3 export files (GLB etc.) in `temp_results` (or the `output_dir` you configure).

To integrate the pipeline into your own code, import and call `run_pipeline`:

```python
from pipeline import run_pipeline

frame_3d = run_pipeline(image_path, model_path, out_dir)
```

`frame_3d` is a `PoseDepth3D` instance that holds all 3D keypoints (camera/world coordinates when available).

## Extending the Pipeline with Your Own Models

You can plug in your own pose or depth models by subclassing the base interfaces in `models.py`.

### Custom Pose Model

Create a class that extends `BasePoseEstimator` and implement `predict` and the `name` property:

```python
from models import BasePoseEstimator, PoseEstimationResult, Keypoint, BoundingBox

class MyPoseEstimator(BasePoseEstimator):
	@property
	def name(self) -> str:
		return "my-pose"

	def predict(self, image) -> PoseEstimationResult:
		# 1) Run your pose model on `image`
		# 2) Convert outputs to lists of Keypoint and BoundingBox
		return PoseEstimationResult(imageShape=image.shape[:2],
								   keypoints=[...],
								   boundingBoxes=[...])
```

Then use it in `pipeline.py` instead of `YoloPoseEstimator`.

### Custom Depth Model

Similarly, extend `BaseDepthEstimator` and implement `predict` and the `name` property:

```python
from models import BaseDepthEstimator, DepthEstimationResult

class MyDepthEstimator(BaseDepthEstimator):
	@property
	def name(self) -> str:
		return "my-depth"

	def predict(self, image) -> DepthEstimationResult:
		# Run your depth model and produce a depth map `depth`
		return DepthEstimationResult(imageShape=image.shape[:2],
									 depth=depth,
									 isMetric=None,
									 intrinsics=None,
									 extrinsics=None)
```

If your depth model can provide camera intrinsics/extrinsics, fill those fields as well. The rest of the pipeline will automatically compute 3D keypoints using your implementation.

