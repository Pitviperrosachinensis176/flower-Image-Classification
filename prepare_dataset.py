import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8, seed=42):

    random.seed(seed)

    if not os.path.exists(source_dir):
        print(f"[ERROR] Source directory not found: {source_dir}")
        return

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    classes = [d for d in os.listdir(source_dir)
               if os.path.isdir(os.path.join(source_dir, d))]

    print(f"[INFO] Classes found: {classes}")

    total_train = 0
    total_val = 0

    for class_name in classes:
        class_source = os.path.join(source_dir, class_name)
        class_train = os.path.join(train_dir, class_name)
        class_val = os.path.join(val_dir, class_name)

        os.makedirs(class_train, exist_ok=True)
        os.makedirs(class_val, exist_ok=True)

        images = [f for f in os.listdir(class_source)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        for img in train_images:
            shutil.copy2(
                os.path.join(class_source, img),
                os.path.join(class_train, img)
            )
        for img in val_images:
            shutil.copy2(
                os.path.join(class_source, img),
                os.path.join(class_val, img)
            )

        print(f"  {class_name:12s} → train: {len(train_images)}, val: {len(val_images)}")
        total_train += len(train_images)
        total_val += len(val_images)

    print(f"\n[SUCCESS] Dataset splitting completed.")
    print(f"Total training images: {total_train}")
    print(f"Total validation images: {total_val}")


def main():
    parser = argparse.ArgumentParser(description='Prepare and split the flower dataset')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to the source dataset directory (containing class subdirectories)')
    parser.add_argument('--train', type=str, default='dataset/train',
                        help='Output path for the training set')
    parser.add_argument('--val', type=str, default='dataset/validation',
                        help='Output path for the validation set')
    parser.add_argument('--split', type=float, default=0.8,
                        help='Ratio of training data (default: 0.8)')
    args = parser.parse_args()

    split_dataset(args.source, args.train, args.val, args.split)


if __name__ == "__main__":
    main()
