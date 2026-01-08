import zipfile
from pathlib import Path

zip_path = Path(r"E:\normal.zip")
dest_dir = Path(r"E:\downloads\wound_dataset\Wound_dataset\Normal_skin")

dest_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(dest_dir)

print("Extracted normal images to", dest_dir)
