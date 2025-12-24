# Master Automating Pose Estimation


## File overview
- `train_segmentation_yolo.py`: Trains YOLOv11 instance segmentation.
- `trudi_ds_yolo11_instand_segmentation.yaml`: YOLO dataset config (paths + class names).
- `README.md`: This file; brief index of what each file does.
- `/scripts/convert_trudi_ds_to_yolo_segmentation.py` converts the trudi_ds data to the coco format that yolo understands
- `scripts/split_trudi_ds_converted_seg.py` splits the dataset into training and validation dataset
- `predict_segmentation_yolo.py` predicts a set of given images and saves the image
- `predict_segmentation.py` predicts the test images in the dataset
- `val_segmentation_yolo.py` validates the model and saves it
- `convert_synthetic_data-v2_to_coco.py` converts the synthetic_data-v2 into coco format (keypoint or segmentation?)


## training
5 classes
  0: container
  1: freight_car
  2: semi_trailer
  3: tank_container
  4: trailer

3 classes
  0: 
  1: 
  2: 


- `segment/train` (first training)
  - 600 epochs
  - 1280 image size
  - training with 5 classes
  - dataset: trudi_ds
  - model: yolo11n-seg.yaml

- `segment/train2` (second training)
  - 1000 epochs, but canceld after 609 because no change in the last 100 epochs
  - 1280 image size
  - training with 5 classes
  - dataset: trudi_ds
  - model: yolo11n-seg.yaml

- `segment/train3` 
  - 700 epochs, (could not do 1000 epochs because cuda memory error )
  - 1280 image size
  - training with 3 classes
  - dataset: trudi_ds
  - model: yolo11n-seg.yaml

- `pose/train` synthetic data
  - 700 epochs, (but canceled at 422 because no improvement in the last 100 runs)
  - 1280 image size
  - training with 1 class
  - dataset: synthetic_data-v2
  - model: yolo11n-pose.pt

- `pose/train2` synthetic data
  - 700 epochs, (but canceled at 639 because no improvement in the last 100 runs)
  - 1280 image size
  - training with 1 class
  - dataset: synthetic_data-v2
  - model: yolo11n-pose.yaml

The validation results are in the `./results/` folder as well as in `./runs/segment/val*`
