from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-seg.pt")  # load an official model
model = YOLO("runs/segment/train/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps  # a list contains map50-95(B) of each category
metrics.seg.map  # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps  # a list contains map50-95(M) of each category
with open("validation_metrics_train.txt", "w") as f:
  f.write("Detection (boxes):\n")
  f.write(f"mAP50-95: {metrics.box.map}\n")
  f.write(f"mAP50: {metrics.box.map50}\n")
  f.write(f"mAP75: {metrics.box.map75}\n")
  f.write(f"Per-class mAP50-95: {', '.join(map(str, metrics.box.maps))}\n\n")

  f.write("Segmentation (masks):\n")
  f.write(f"mAP50-95: {metrics.seg.map}\n")
  f.write(f"mAP50: {metrics.seg.map50}\n")
  f.write(f"mAP75: {metrics.seg.map75}\n")
  f.write(f"Per-class mAP50-95: {', '.join(map(str, metrics.seg.maps))}\n")