from pathlib import Path
import shutil

src_dir = Path(r"E:\downloads\wound_dataset\Wound_dataset\Normal_skin\normal")
dst_dir = Path(r"E:\downloads\wound_dataset\Wound_dataset\Normal_skin")

for img in src_dir.iterdir():
    if img.is_file():
        shutil.move(str(img), dst_dir / img.name)

print("Moved images to", dst_dir)
