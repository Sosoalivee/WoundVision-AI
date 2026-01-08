import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split  # pip install scikit-learn

# 1) WHERE YOUR ORIGINAL CLASS FOLDERS ARE (Abrasions, Bruises, ...)
SOURCE_ROOT = Path(r"E:\downloads\wound_dataset\Wound_dataset")


# 2) PROJECT ROOT (THIS FILE SITS HERE)
PROJECT_ROOT = Path(__file__).resolve().parent

# 3) EXISTING TARGET STRUCTURE
DATASET_ROOT = PROJECT_ROOT / "dataset"
IMAGES_ROOT = DATASET_ROOT / "images"   # has train/val/test
LABELS_ROOT = DATASET_ROOT / "labels"   # optional for simple labels

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

def ensure_split_dirs(class_name: str):
    for split in ["train", "val", "test"]:
        (IMAGES_ROOT / split / class_name).mkdir(parents=True, exist_ok=True)

def split_and_copy_class(class_dir: Path, class_id: int):
    class_name = class_dir.name
    images = sorted(class_dir.glob("*.jpg"))

    if not images:
        print(f"No images in {class_dir}, skipping.")
        return

    train_imgs, temp_imgs = train_test_split(
        images, train_size=TRAIN_RATIO, shuffle=True, random_state=42
    )
    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_imgs, test_imgs = train_test_split(
        temp_imgs, train_size=val_size, shuffle=True, random_state=42
    )

    ensure_split_dirs(class_name)

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs,
    }

    for split_name, split_files in splits.items():
        split_dir = IMAGES_ROOT / split_name / class_name
        label_file = LABELS_ROOT / split_name / "labels.txt"
        label_file.parent.mkdir(parents=True, exist_ok=True)

        with open(label_file, "a") as lf:
            for src in split_files:
                dst = split_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                rel = dst.relative_to(PROJECT_ROOT).as_posix()
                lf.write(f"{rel} {class_id}\n")

    print(
        f"{class_name}: "
        f"{len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test"
    )

def main():
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"SOURCE_ROOT does not exist: {SOURCE_ROOT}")

    for sub in ["train", "val", "test"]:
        if not (IMAGES_ROOT / sub).exists():
            raise FileNotFoundError(f"Missing folder: {IMAGES_ROOT / sub}")

    # class_name -> integer id (sorted for stability)
    class_dirs = [d for d in SOURCE_ROOT.iterdir() if d.is_dir()]
    for class_id, class_dir in enumerate(sorted(class_dirs, key=lambda p: p.name)):
        split_and_copy_class(class_dir, class_id)

    print("Done filling dataset/images and labels.")

if __name__ == "__main__":
    main()
