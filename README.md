# Pose Estimation Pipeline for Loading Units using Monocular RGB Images from Oblique Views

This repository contains the implementation of a Master Thesis by Berkkan Katırcı, which develops an automated pose estimation pipeline for loading units (containers, pallets) using monocular RGB images captured from oblique (aerial) views.

## Overview

The goal of this project is to automate the detection and 3D pose estimation of loading units in logistics environments. The pipeline combines:

- **YOLO-based Deep Learning Models** for 2D object detection, segmentation, and keypoint estimation
- **Depth Estimation** using state-of-the-art models (Depth Anything V2/V3, Depth Pro)
- **3D Reconstruction** by combining 2D keypoints with depth information to obtain 3D coordinates

The system is designed to work with:
- Real-world datasets (TruDI dataset - drone imagery of loading units)
- Synthetic datasets (computer-generated scenes for training)

### Key Features

- End-to-end pipeline from image input to 3D pose estimation
- Support for multiple YOLO architectures (YOLO11, custom keypoint detection)
- Multiple depth estimation backends (Depth Anything V2, Depth Anything V3, Depth Pro)
- Comprehensive training, validation, and inference scripts
- Modular architecture allowing easy extension with custom models

## Pipeline Architecture

The main pipeline combines pose estimation and depth estimation to produce 3D keypoints. See the [`pipeline/`](./pipeline/) folder for detailed documentation and implementation.

**High-level workflow:**

1. **2D Pose Estimation**: YOLO model detects loading units and estimates 2D keypoints
2. **Depth Estimation**: Depth model produces a dense depth map from the same image
3. **3D Projection**: Keypoints are projected to 3D using depth values and camera intrinsics
4. **Visualization**: Results are saved with 2D/3D overlays for analysis

For detailed pipeline documentation, architecture diagrams, and usage instructions, see [pipeline/README.md](./pipeline/README.md).

## Repository Structure

```
master-automating-pose-estimation/
├── pipeline/                    # Main 3D pose estimation pipeline
│   ├── pipeline.py             # Main pipeline script
│   ├── yolo_estimator.py       # YOLO pose wrapper
│   ├── depth_estimator.py      # Depth estimation wrapper
│   └── README.md               # Detailed pipeline documentation
├── scripts/                     # Data preparation and conversion scripts
├── depth_estimation/            # Depth estimation experiments
├── results/                     # Output directory for predictions
├── runs/                        # YOLO training outputs
└── datasets/                    # Dataset storage (not in repo)
```

## Installation

### Requirements

- Python 3.11+ (3.11.2 recommended)
- CUDA-capable GPU (strongly recommended)
- Linux/Windows with WSL (pipeline tested on Ubuntu-like systems)

### Setup

1. **Clone the repository**:
```bash
git clone <repo-url>
cd master-automating-pose-estimation
```

2. **Create and activate virtual environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Install Depth Estimation Models** (choose based on your needs):

```bash
# Depth Anything V3 (used by main pipeline)
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 depth-anything-3
cd depth-anything-3
pip install -e .
cd ..

# Depth Anything V2 (optional, for comparison)
git clone https://github.com/DepthAnything/Depth-Anything-V2 Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
cd ..

# Depth Pro (optional, metric depth estimation)
git clone https://github.com/apple/ml-depth-pro ml-depth-pro
cd ml-depth-pro
pip install -e .
cd ..
```

## Scripts Overview

### Training Scripts

#### `train_pose_yolo.py`
Trains a YOLO11 pose estimation model to detect loading unit keypoints.

**Configuration**:
- Model: YOLO11n-pose (from scratch or pretrained)
- Dataset: Configured via YAML file (e.g., `synthetic_data-v2_keypoints.yaml`)
- Training parameters: 700 epochs, image size 1280px

**Usage**:
```bash
python train_pose_yolo.py
```

**Note**: Edit the script to change dataset YAML or training parameters before running.

---

#### `train_segmentation_yolo.py`
Trains a YOLO11 segmentation model to segment loading units (instance segmentation).

**Configuration**:
- Model: YOLO11n-seg
- Dataset: Configured via YAML file (e.g., `trudi_ds_yolo11_instand_segmentation.yaml` or `synthetic_data-v2_keypoints.yaml`)
- Training parameters: 700 epochs, image size 1280px

**Usage**:
```bash
python train_segmentation_yolo.py
```

---

### Prediction/Inference Scripts

#### `predict_pose.py`
Runs pose prediction on a test dataset and evaluates performance with detailed metrics.

**What it does**:
- Loads a trained YOLO pose model
- Runs inference on test images
- Matches predictions to ground truth using IoU
- Computes keypoint metrics (PCK - Percentage of Correct Keypoints)
- Saves visualizations (GT vs. predictions side-by-side)
- Generates summary metrics (JSON and TXT)

**Configuration**:
- Model path: `runs/pose/train/weights/best.pt`
- Test dataset: `datasets/synthetic_data-v2-coco-v2/images/test`
- Output: `results/pose_predictions_train/`

**Usage**:
```bash
python predict_pose.py
```

**Outputs**:
- `<image>_gt.jpg`: Ground truth keypoints visualization
- `<image>_pred.jpg`: Predicted keypoints visualization
- `<image>_gt_vs_pred.jpg`: Side-by-side comparison
- `summary_metrics.json`: Aggregate metrics (TP, FP, FN, PCK, etc.)
- `per_image_summary.txt`: Per-image detection counts

---

#### `predict_segmentation.py`
Runs segmentation prediction and evaluates IoU metrics against ground truth masks.

**What it does**:
- Loads a trained YOLO segmentation model
- Runs inference on test images
- Computes IoU between predicted and ground truth masks
- Saves visualizations with masks overlaid

**Usage**:
```bash
python predict_segmentation.py
```

---

#### `predict_segmentation_yolo.py`
Batch prediction script for testing multiple YOLO segmentation models at various confidence thresholds.

**What it does**:
- Tests multiple training runs (train, train2, train3)
- Sweeps confidence thresholds from 0.30 to 1.00 in 0.05 increments
- Saves predictions for comparison

**Configuration**:
- Models: `runs/segment/{train,train2,train3}/weights/best.pt`
- Confidence range: 0.30 to 1.00 (step 0.05)
- Output: `same_images_predictions/`

**Usage**:
```bash
python predict_segmentation_yolo.py
```

---

### Depth + Pose Scripts

#### `predict_pose_with_depth.py`
Combines YOLO pose estimation with Depth Anything V2 to estimate 3D keypoint coordinates.

**What it does**:
- Runs YOLO pose model to get 2D keypoints
- Runs Depth Anything V2 for depth estimation
- Uses a reference object (first container) with known dimensions to scale depth to meters
- Samples depth at each keypoint location
- Saves TXT files with 2D and 3D coordinates for each keypoint
- Visualizes keypoints with 3D annotations

**Configuration**:
- Pose model: `runs/pose/train/weights/best.pt`
- Depth model: Depth Anything V2 (checkpoint: `depth_anything_v2_vits.pth`)
- Reference height: 2.6m (container height)
- Camera intrinsics: Configurable focal length and sensor dimensions

**Usage**:
```bash
python predict_pose_with_depth.py \
    --pose-model runs/pose/train/weights/best.pt \
    --da-checkpoint depth_anything_v2_vits.pth \
    --image-dir datasets/test_images \
    --out-dir results/pose_depth
```

**Outputs**:
- `<image>_keypoints_3d.txt`: Per-keypoint 2D and 3D coordinates
- `<image>_visualization.jpg`: Image with keypoints labeled with 3D coordinates

---

#### `pose_depth_all_keypoints.py`
Uses Depth Pro (metric depth estimation) to compute accurate 3D coordinates for all detected keypoints.

**What it does**:
- Runs YOLO pose model to detect keypoints
- Runs Depth Pro to get metric depth map (in meters)
- Uses pinhole camera model with focal length to back-project keypoints to 3D
- Saves all keypoints with 2D+3D coordinates to TXT files
- Creates visualizations with 3D coordinates annotated

**Configuration**:
- Pose model: `runs/pose/train/weights/best.pt`
- Depth model: Depth Pro (automatically downloaded from Hugging Face)
- Output: `results/pose_depth_all/`

**Usage**:
```bash
python pose_depth_all_keypoints.py \
    --image path/to/image.jpg \
    --pose-model runs/pose/train/weights/best.pt \
    --out-dir results/pose_depth_all
```

**Outputs**:
- `<image>_keypoints_3d.txt`: All detected keypoints with 3D coordinates
- `<image>_visualization.jpg`: Visualization with 3D labels

---

#### `pose_depth_from_points.py`
Similar to `pose_depth_all_keypoints.py`, but allows manual specification of points for depth sampling.

**Usage**:
```bash
python pose_depth_from_points.py --image <image_path> --points-file <points.txt>
```

---

### Data Preparation Scripts (in `scripts/`)

#### `convert_pose_to_coco.py`
Converts synthetic pose dataset (pose.jsonl format) to COCO format with train/val/test splits.

**Usage**:
```bash
python scripts/convert_pose_to_coco.py \
    --src-root synthetic_data-v2/synthetic_data-v2 \
    --out-root datasets/synth_pose_coco \
    --seed 42 --train 0.8 --val 0.1 --test 0.1
```

---

#### `convert_synthetic_data-v2_to_coco.py`
Converts synthetic segmentation dataset to COCO format.

**Usage**:
```bash
python scripts/convert_synthetic_data-v2_to_coco.py \
    --src-root synthetic_data-v2 \
    --out-root datasets/synthetic_coco
```

---

#### `convert_trudi_ds_to_yolo_segmentation.py`
Converts TruDI dataset annotations to YOLO segmentation format.

**Usage**:
```bash
python scripts/convert_trudi_ds_to_yolo_segmentation.py
```

---

#### `split_trudi_ds_converted_seg.py`
Splits TruDI dataset into train/val/test sets with proper directory structure.

**Usage**:
```bash
python scripts/split_trudi_ds_converted_seg.py
```

---

#### `visualize_pose_points.py`
Utility script to visualize pose keypoint annotations on images.

**Usage**:
```bash
python scripts/visualize_pose_points.py --image <path> --labels <path>
```

---

## Datasets

This project uses two main datasets:

1. **TruDI Dataset**: Real-world drone images of loading units in logistics environments
2. **Synthetic Dataset V2**: Computer-generated scenes with perfect ground truth annotations

Dataset YAML configuration files:
- `trudi_ds_yolo11_instand_segmentation.yaml` - TruDI segmentation config
- `synthetic_data-v2_keypoints.yaml` - Synthetic pose estimation config

## Results

Training and prediction outputs are saved to:
- `runs/` - YOLO training outputs (models, metrics, visualizations)
- `results/` - Prediction outputs organized by script/experiment
- `temp_results/` - Temporary pipeline outputs

## Citation

If you use this code or pipeline in your research, please cite:

```
Berkkan Katirci (2026)
"Pose Estimation Pipeline for Loading Units using Monocular RGB Images from Oblique Views"
Master Thesis
```

```
@mastersthesis{katirci2026pose,
author = {Berkkan Katirci},
title = {{Pose Estimation Pipeline for Loading Units using Monocular RGB Images from Oblique Views}},
school = {University of Hamburg},
year = {2026},
type = {Master's thesis},
address = {Hamburg, Germany},
url = {https://github.com/berkkan22/master-automating-pose-estimation}
}
```

## SSH Access to GitHub

Generate a key, add it to GitHub, test, then use it to clone/pull/push.

```bash
# 1) Generate key
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2) Start agent and add key
eval "$(ssh-agent -s)"

# 3) Show public key (copy this)
cat ~/.ssh/id_ed25519.pub
```

On GitHub: **Settings → SSH and GPG keys → New SSH key**, paste the copied key, save.

```bash
# 4) Test
ssh -T git@github.com

# 5) Clone private repo via SSH
git clone git@github.com:owner/repo.git

# 6) Normal workflow
git pull
git push
```

**References**
- GitHub Docs: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
- GitHub Docs: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account