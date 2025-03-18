import os
from tqdm import tqdm
import json

imgs_dir = "images"

annotation_path = "extracted_images/WebQA_data/WebQA_train_val.json"
with open(annotation_path, "r") as f:
    data = json.load(f)

with open("samples.txt", 'r') as f:
    samples = [line.strip() for line in f.readlines()]

for guid in samples:
    sample = data[guid]
    img_posFacts = [img['image_id'] for img in sample['img_posFacts']]
    img_negFacts = [img['image_id'] for img in sample['img_negFacts']]

    img_ids = img_posFacts + img_negFacts
    for id_ in img_ids:
        path = os.path.join(imgs_dir, id_)
        if not os.path.exists(path):
            print(f"Missing image: {id_}")

