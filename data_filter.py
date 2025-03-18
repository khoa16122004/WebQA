import os
from tqdm import tqdm
import json

imgs_dir = "images"
img_name_list = [name.split(".")[0] for name in os.listdir(imgs_dir)]
img_name_dict = {name: True for name in img_name_list}

annotation_path = "extracted_images/WebQA_data/WebQA_train_val.json"
with open(annotation_path, "r") as f:
    data = json.load(f)

Success = []
for key, item in tqdm(data.items(), desc="Processing annotations"):
    img_posFacts = item['img_posFacts']
    img_negFacts = item['img_negFacts']

    if not all(img in img_name_list for img in img_posFacts + img_negFacts):
        Success.append(key)

output_path = "missing_images.txt"
with open(output_path, "w") as f:
    f.write("\n".join(Success))

print(f"Saved missing image keys to {output_path}")
