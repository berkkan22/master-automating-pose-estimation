import os
import json
import shutil

# Classes to include and their indices
CLASSES = {
    "container": 0,
    "freight_car": 1,
    "semi_trailer": 2,
    "tank_container": 3,
    "trailer": 4
}


def normalize_point(x, y, img_w, img_h):
    return x / img_w, y / img_h


def convert_annotation(json_path, output_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    image_name = os.path.splitext(os.path.basename(data["imagePath"]))[0]
    txt_path = os.path.join(output_dir, image_name + ".txt")

    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in CLASSES:
            continue
        class_idx = CLASSES[label]
        points = shape.get("points", [])
        norm_points = []
        for x, y in points:
            nx, ny = normalize_point(x, y, img_w, img_h)
            norm_points.extend([f"{nx:.6f}", f"{ny:.6f}"])
        line = f"{class_idx} " + " ".join(norm_points)
        lines.append(line)

    if lines:
        with open(txt_path, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(lines))


if __name__ == "__main__":
    # Change these paths as needed
    input_dir = "./trudi_ds/data"  # Directory with JSON files
    output_dir = "./yolo_labels"  # Directory to save YOLO txt files
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.endswith(".json"):
            json_path = os.path.join(input_dir, fname)
            convert_annotation(json_path, output_dir)
            # Copy corresponding image
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            image_path = data.get("imagePath")
            if image_path:
                src_img_path = os.path.join(input_dir, image_path)
                dst_img_path = os.path.join(
                    output_dir, os.path.basename(image_path))
                if os.path.exists(src_img_path):
                    shutil.copy2(src_img_path, dst_img_path)
