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
    img_posFacts = [img['image_id'] for img in item['img_posFacts']]
    img_negFacts = [img['image_id'] for img in item['img_negFacts']]
    img_ids = img_posFacts + img_negFacts
    print(img_ids)
    for img_id in img_ids:
        
        try:
            img_name_dict[str(img_id)]
            Success.append(key)
            # print("append")
        except:
            # print("not append")
            continue
        
        input()

output_path = "samples.txt"
with open(output_path, "w") as f:
    f.write("\n".join(Success))

print(f"Saved missing image keys to {output_path}")
