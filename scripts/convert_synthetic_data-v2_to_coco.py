import json
import os
import cv2
import shutil

# Define paths
base_path = "./synthetic_data-v2/synthetic_data-v2"

# Iterate through each folder in the base folder
folders = [os.path.join(base_path, folder) for folder in os.listdir(
    base_path) if os.path.isdir(os.path.join(base_path, folder))]

# Create train, val, and test folders inside images and labels
base_output_path = os.path.join("./datasets/synthetic_data-v2-coco-v2")
images_output_path = os.path.join(base_output_path, "images")
labels_output_path = os.path.join(base_output_path, "labels")

train_images_path = os.path.join(images_output_path, "train")
val_images_path = os.path.join(images_output_path, "val")
test_images_path = os.path.join(images_output_path, "test")
train_labels_path = os.path.join(labels_output_path, "train")
val_labels_path = os.path.join(labels_output_path, "val")
test_labels_path = os.path.join(labels_output_path, "test")

for path in [train_images_path, val_images_path, test_images_path, train_labels_path, val_labels_path, test_labels_path]:
    os.makedirs(path, exist_ok=True)

# Split folders into train, val, and test
val_count = 10
test_count = 10
val_folders = folders[:val_count]
test_folders = folders[val_count:val_count + test_count]
train_folders = folders[val_count + test_count:]


def process_folders(folders, images_path, labels_path):
    for idx, folder in enumerate(folders):
        pose_file = os.path.join(folder, "pose.jsonl")
        segmentation_file = os.path.join(folder, "segmentation.txt")
        image_file = os.path.join(folder, "rgb.jpg")
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotation_file = os.path.join(labels_path, f"{idx}.txt")

        print(f"Processing folder {idx}: {folder}")

        # Save the image to the output folder
        output_image_path = os.path.join(images_path, f"{idx}.jpg")
        cv2.imwrite(output_image_path, cv2.cvtColor(
            image, cv2.COLOR_RGB2BGR))

        # Parse segmentation file for bounding boxes
        bounding_boxes = []
        if os.path.exists(segmentation_file):
            with open(segmentation_file, "r") as f:
                for line in f:
                    data = line.strip().split(",")
                    if len(data) == 10:
                        # Normalize by image width
                        x_min = max(
                            0, min(map(int, data[:8:2])))
                        # Normalize by image height
                        y_min = max(
                            0, min(map(int, data[1:8:2])))
                        # Normalize by image width
                        x_max = max(
                            1, max(map(int, data[:8:2])))
                        # Normalize by image height
                        y_max = max(
                            1, max(map(int, data[1:8:2])))
                        label = data[-1]

                        bbox_center_x = (
                            x_min + (x_max - x_min) / 2) / image.shape[1]
                        bbox_center_y = (
                            y_min + (y_max - y_min) / 2) / image.shape[0]
                        bbox_width = (x_max - x_min) / image.shape[1]
                        bbox_height = (y_max - y_min) / image.shape[0]
                        bbox_area = (x_max - x_min) * (y_max - y_min)

                        bounding_boxes.append({
                            "bbox": [bbox_center_x, bbox_center_y, bbox_width, bbox_height],
                            "area": bbox_area,
                            "label": label
                        })

        # Parse pose.jsonl for keypoints
        if os.path.exists(pose_file):
            with open(pose_file, "r") as f:
                for idx, line in enumerate(f):
                    data = json.loads(line)
                    keypoints = []
                    for corner in data["corners"]:
                        if corner["image_position"] is not None:
                            # Normalize by image width
                            normalized_x = corner["image_position"][1] / \
                                image.shape[1]
                            # Normalize by image height
                            normalized_y = corner["image_position"][0] / \
                                image.shape[0]
                            # Visible keypoint
                            keypoints.extend([normalized_x, normalized_y, 2])
                        else:
                            keypoints.extend([0, 0, 0])  # Not visible keypoint

                    # Match bounding box with keypoints
                    bbox_data = bounding_boxes[idx]
                    bbox_str = " ".join(map(str, bbox_data["bbox"]))
                    keypoints_str = " ".join(map(str, keypoints))
                    with open(annotation_file, "a") as f:
                        f.write("0 " + bbox_str + " " + keypoints_str + "\n")


# Process train, val, and test folders
process_folders(train_folders, train_images_path, train_labels_path)
process_folders(val_folders, val_images_path, val_labels_path)
process_folders(test_folders, test_images_path, test_labels_path)
