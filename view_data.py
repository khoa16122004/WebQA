import os
import json
import random

def view_data(sample):
    img_posFacts = sample['img_posFacts']
    img_negFacts = sample['img_negFacts']
    txt_posFacts = sample['txt_posFacts']
    txt_negFacts = sample['txt_negFacts']
    
    print("---------------Image Postive Document--------------------")
    for item in img_posFacts:
        print(f"{item['image_id']} - {item['caption']}")
    
    print("---------------Image Negative Document--------------------")
    for item in img_negFacts:
        print(f"{item['image_id']} - {item['caption']}")
        
    print("---------------Txt Postive Document--------------------")
    for item in txt_posFacts:
        print(f"{item['snippet_id']} - {item['fact']}")
    
    print("---------------Txt Negative Document--------------------")
    for item in txt_negFacts:
        print(f"{item['snippet_id']} - {item['fact']}")
        
def main():
    path = "extracted_images/WebQA_data/WebQA_train_val.json"
    with open(path, "r") as f:
        data = json.load(f)
        keys = data.keys()

    while True:
        print("----------------------------------- Done -----------------------------")
        selection = int(input("Choose: "))
        if selection == -1:
            print("Done")
        
        else:
            sample_id = random.randint(0, len(data))
            sample_key = keys[sample_id]
            view_data(data[sample_key])
            
if __name__ == "__main__":
    main()
    
    