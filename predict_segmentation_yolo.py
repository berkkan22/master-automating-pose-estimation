from ultralytics import YOLO
import os

#! use on local machine
# BASE_PAHT = "./result_segmentation"

#! use on remote server
BASE_PAHT = "./runs"

OUTPUT_DIR = "./same_images_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Predict with the model
# predict on an image

image_paths = [
    "./datasets/trudi_ds_yolo11_instand_segmentation/test/images/DJI_0478_frame_3476.jpg",
]

train_runs = ["train", "train2", "train3"]
conf_values = [i / 100 for i in range(30, 101, 5)]  # 0.50, 0.55, ..., 1.00

for train_name in train_runs:
    model_path = f"{BASE_PAHT}/segment/{train_name}/weights/best.pt"
    model = YOLO(model_path)

    for conf in conf_values:
        results = model(
            image_paths,
            conf=conf,
        )

        # Access the results
        for img_path, result in zip(image_paths, results):
            # xy = result.masks.xy  # mask in polygon format
            # xyn = result.masks.xyn  # normalized
            # masks = result.masks.data  # mask in matrix format (num_objects x H x W)
            img_name = os.path.basename(img_path)
            conf_str = f"{conf:.2f}"
            save_name = f"{train_name}_conf{conf_str}_{img_name}"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            result.save(filename=save_path)
