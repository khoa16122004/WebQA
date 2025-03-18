import os
import base64
from PIL import Image
from io import BytesIO
from tqdm import tqdm

tsv_path = "extracted_images/imgs.tsv"
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
with open(tsv_path, "r") as tsv_file:
    data = [line.strip().split("\t") for line in tsv_file.readlines()]

for i in tqdm(range(len(data))):
    imgid, img_base64 = data[i]
    im = Image.open(BytesIO(base64.b64decode(img_base64))).convert("RGB")
    im.save(os.path.join(output_dir, f"{imgid}.png"))
    