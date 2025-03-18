import os
import base64
from PIL import Image
from io import BytesIO
from tqdm import tqdm

tsv_path = "extracted_images/imgs.tsv"
output_dir = "images"
fail_path = "fail.txt"
fail_list = []
os.makedirs(output_dir, exist_ok=True)
with open(tsv_path, "r") as tsv_file:
    data = [line.strip().split("\t") for line in tsv_file.readlines()]

for i in tqdm(range(len(data))):
    imgid, img_base64 = data[i]
    try:
        img_path = os.path.join(output_dir, f"{imgid}.png")
        if not os.path.exists(img_path):
            im = Image.open(BytesIO(base64.b64decode(img_base64))).convert("RGB")
            im.save(img_path)
    except:
        fail_list.append(imgid)

with open(fail_path, "w") as f:
    f.write("\n".join(fail_list))
        