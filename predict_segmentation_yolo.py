from ultralytics import YOLO

#! use on local machine
# BASE_PAHT = "./result_segmentation"

#! use on remote server
BASE_PAHT = "./runs"

model = YOLO(f"{BASE_PAHT}/segment/train/weights/best.pt")


# Predict with the model
# predict on an image
results = model(
    ["./trudi_ds/data/DJI_0327_frame_7268.jpg"],
    conf=0.80  # confidence threshold, adjust as needed
)

# Access the results
for result in results:
    # xy = result.masks.xy  # mask in polygon format
    # xyn = result.masks.xyn  # normalized
    # masks = result.masks.data  # mask in matrix format (num_objects x H x W)
    result.save(
        filename="test2.jpg")
