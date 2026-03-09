from ultralytics import YOLO
import os

#! use on local machine
# BASE_PAHT = "./result_segmentation"

#! use on remote server
BASE_PAHT = "./runs"

OUTPUT_DIR = "./syntetic_resutls"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(f"{BASE_PAHT}/segment/train2/weights/best.pt")


# Predict with the model
# predict on an image
image_paths = [
    "./synthetic_data-v2/synthetic_data-v2/v1_default_270/rgb.jpg",
    "./synthetic_data-v2/synthetic_data-v2/v1_human_140/rgb.jpg",
    "./synthetic_data-v2/synthetic_data-v2/v1_drone_31/rgb.jpg",
    "./synthetic_data-v2/synthetic_data-v2/v1_drone_115/rgb.jpg",
    "./synthetic_data-v2/synthetic_data-v2/v1_default_15/rgb.jpg"
]

results = model(
    image_paths,
    conf=0.50  # confidence threshold, adjust as needed
)

# Access the results
for img_path, result in zip(image_paths, results):
    # xy = result.masks.xy  # mask in polygon format
    # xyn = result.masks.xyn  # normalized
    # masks = result.masks.data  # mask in matrix format (num_objects x H x W)
    folder_name = os.path.basename(os.path.dirname(img_path))
    save_path = os.path.join(OUTPUT_DIR, f"{folder_name}.jpg")
    result.save(filename=save_path)
