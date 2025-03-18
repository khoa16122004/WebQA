import os
import base64
from PIL import Image
from io import BytesIO
from tqdm import tqdm

tsv_path = "extracted_images/imgs.tsv"
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

with open(tsv_path, "r") as tsv_file:
    imgids = [line.strip().split("\t")[0] for line in tsv_file]
    img_base64s = [line.strip().split("\t")[1] for line in tsv_file]

for i in tqdm(range(len(imgids))):
    imgid = imgids[i]
    img_base64 = img_base64s[i]
    im = Image.open(BytesIO(base64.b64decode(img_base64)))
    im.save(os.path.join(output_dir, f"{imgid}.jpg"))
    