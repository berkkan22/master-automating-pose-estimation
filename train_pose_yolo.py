from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML

# load a pretrained model (recommended for training)
# model = YOLO("yolo11n-seg.pt")


#model.TASK("segment")  # set the task to segmentation

# Train the model
# results = model.train(
#     data="trudi_ds_yolo11_instand_segmentation.yaml", epochs=700, imgsz=1280, task="segment")
results = model.train(
    data="synthetic_data-v2_keypoints.yaml", epochs=700, imgsz=1280, task="pose")
