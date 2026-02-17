import os
import shutil
import random

# Set your source folder containing images and labels
source_folder = './yolo_labels'
output_folder = './datasets/trudi_ds_yolo11_instand_segmentation'
# Persist split order at repository root to stay consistent across systems
split_txt = os.path.join('.', 'split_order.txt')

# Ratios
train_ratio = 0.8  # 80% train
val_ratio = 0.1    # 10% val
test_ratio = 0.1   # 10% test

# Deterministic seed for initial generation
SEED = 33

image_extensions = ['.jpg']


def ensure_split_dirs():
  for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_folder, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, split, 'labels'), exist_ok=True)


def list_images():
  if not os.path.isdir(source_folder):
    raise FileNotFoundError(f"Source folder not found: {source_folder}")
  return sorted([f for f in os.listdir(source_folder) if os.path.splitext(f)[1].lower() in image_extensions])


def read_split_txt():
  if not os.path.exists(split_txt):
    return None
  with open(split_txt, 'r', encoding='utf-8') as f:
    lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.startswith('#')]
  return lines if lines else None


def write_split_txt(files):
  os.makedirs(output_folder, exist_ok=True)
  with open(split_txt, 'w', encoding='utf-8') as f:
    f.write(f"# split order generated with seed {SEED}\n")
    for fn in files:
      f.write(fn + '\n')
  print(f"Wrote split file: {split_txt}")


def compute_slices(order):
  total = len(order)
  train_end = int(total * train_ratio)
  val_end = train_end + int(total * val_ratio)
  return order[:train_end], order[train_end:val_end], order[val_end:]


def copy_files(image_list, split):
  for img_file in image_list:
    img_src = os.path.join(source_folder, img_file)
    img_dst = os.path.join(output_folder, split, 'images', img_file)
    shutil.copy2(img_src, img_dst)

    label_file = os.path.splitext(img_file)[0] + '.txt'
    label_src = os.path.join(source_folder, label_file)
    label_dst = os.path.join(output_folder, split, 'labels', label_file)
    if os.path.exists(label_src):
      shutil.copy2(label_src, label_dst)


def main():
  ensure_split_dirs()

  # 1) Try to read existing split txt
  order = read_split_txt()

  # 2) If none, generate deterministically from current source
  if order is None:
    candidates = list_images()
    if not candidates:
      print(f"No images found in {source_folder} with extensions {image_extensions}.")
      return
    random.seed(SEED)
    order = candidates[:]
    random.shuffle(order)
    write_split_txt(order)
  else:
    # Ensure files exist in source, ignore those that are missing
    candidates = set(list_images())
    original_len = len(order)
    order = [fn for fn in order if fn in candidates]
    if len(order) < original_len:
      print(f"Warning: {original_len - len(order)} files from split_order.txt are missing in source and will be skipped.")

  # 3) Compute split slices and copy
  train_images, val_images, test_images = compute_slices(order)
  copy_files(train_images, 'train')
  copy_files(val_images, 'val')
  copy_files(test_images, 'test')

  print(f"Done. Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")


if __name__ == '__main__':
  main()