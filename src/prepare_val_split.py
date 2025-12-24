from pathlib import Path
import os
import shutil

# Gets the folder where THIS script is saved
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to Tiny ImageNet validation folder (note: directory name is "tiny-imagenet-200")
VAL_DIR = BASE_DIR / "data" / "tiny-imagenet-200" / "val"
VAL_IMAGES_DIR = VAL_DIR / "images"

# Tiny ImageNet の val_annotations.txt は
#   <image_name>\t<label>\t<x>\t<y>\t<w>\t<h>
# という形式なので、最初の2列だけ使う
with open(VAL_DIR / "val_annotations.txt", "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        img, label = parts[0], parts[1]

        # ラベルディレクトリを val/ 以下に作成 (一般的な変換: val/<class>/<image>.JPEG)
        label_dir = VAL_DIR / label
        os.makedirs(label_dir, exist_ok=True)

        src = VAL_IMAGES_DIR / img
        dst = label_dir / img

        # 同じファイルが既にある場合は上書きしなくても良いなら try/except にしてもOK
        shutil.copy(src, dst)


