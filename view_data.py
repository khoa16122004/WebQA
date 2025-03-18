import os
import json
import random

def view_data(sample):
    
    print("Question: ", sample["Q"])
    print("Answer: ", sample['A'])
    
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
    sample_path = "samples.txt"
    with open(sample_path) as g:
        sample_guids = [line.strip() for line in g.readlines()]
        
    with open(path, "r") as f:
        data = json.load(f)
        keys = list(data.keys())

    while True:
        print("----------------------------------- Done -----------------------------")
        selection = int(input("Choose: "))
        if selection == -1:
            print("Done")
        
        else:
            sampel_guid = random.choice(sample_guids)
            view_data(data[sampel_guid])
            
if __name__ == "__main__":
    main()
    
    