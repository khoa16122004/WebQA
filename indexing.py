import faiss
import numpy as np
import clip
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
import json
from torchvision import transforms
import cv2 as cv
import faiss.contrib.torch_utils

class VectorDB:
    def __init__(self, txt_sample_path=None, 
                 image_sample_path=None,
                 txt_index_path=None ,
                 image_index_path=None,
                 model_name='ViT-B/32'):
        self.model, self.preprocess = clip.load(model_name, device="cuda")

        if txt_sample_path:
            self.txt_df = pd.read_csv(txt_sample_path)
        
        if image_sample_path:
            self.image_df = pd.read_csv(image_sample_path)
        if image_index_path:
            res = faiss.StandardGpuResources()

            self.image_index = faiss.read_index(image_index_path)
            # self.image_index = faiss.index_cpu_to_gpu(res,
            #                                           "0",
            #                                           self.image_index)
            print("Image vectors db len: ", self.image_index.ntotal)
        if txt_index_path:
            self.text_index = faiss.read_index(txt_index_path)
            print("Text vectors db len: ", self.text_index.ntotal)

    def faiss_add(self, features):
        feature_dim = features.shape[1]
        index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(),
                                     feature_dim)
        # print(features.shape)
        if not index.is_trained:
            raise RuntimeError("Faiss is not trained")
            
        else:
            print("Store vector proccessing")
            
            # for feature_vector in tqdm(features):
                # index.add(feature_vector)
            # print(type(features))
            index.add(features)    
            return index

        
    def search(self, question_query, k=3):
        question_encode = clip.tokenize(question_query).cuda()
        query_feature = self.model.encode_text(question_encode).to(torch.float32).cpu()
        
        D_img, I_img = self.image_index.search(query_feature, k)
        # D_txt, I_txt = self.text_index.search(query_feature, k)
        I_img = I_img[0].tolist()

        result_img = self.image_df[self.image_df['index'].isin(I_img)]
        # result_txt = self.txt_df.iloc[I_txt]
        
        # return result_img, result_txt
        return result_img
    

    
    def indexing_image(self, img_dir, output_path, batch_size=128):
        # pil reading
        # when retreieval: img_paths[index] -> img_path
        imgs = []
        imgs_name = sorted(os.listdir(img_dir))
        img_paths = [os.path.join(img_dir, img_name) for img_name in imgs_name]
        print("Len Image Database: ", len(img_paths))

        print("Preproccess image: ")
        for img_path in img_paths:
            img = Image.open(img_path).convert("RGB")

            imgs.append(self.preprocess(img))
        imgs = torch.stack(imgs).cuda()
        print(imgs.shape)
        
        print("Extract features of image:  ")

        # CLIP preprocess
        img_features = []
        with torch.no_grad():
            for i in range(0, imgs.shape[0], batch_size):
                if i + batch_size < imgs.shape[0]:
                    imgs_batch = imgs[i : i + batch_size]
                else:
                    imgs_batch = imgs[i:]
                    
                imgs_features = self.model.encode_image(imgs_batch)
                img_features.append(imgs_features)
        img_features = torch.cat(img_features, dim=0)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True).to(torch.float32).cpu()
        
        # add index
        index = self.faiss_add(img_features)
        
        # save index        
        faiss.write_index(index, output_path)
        image_df = pd.DataFrame({'index': range(len(imgs_name)),
                                      "image_id": imgs_name})
        image_df.to_csv("image_df.csv", index=False)
        
    def indexing_text(self, samples_path, annotation_path, output_path, batch_size=128):
        with open(samples_path) as f, open(annotation_path) as g:
            samples = [line.strip() for line in f.readlines()][:5]
            annotations = json.load(g)
            
            txt_features = []
            txt_snippet_ids = []
            for guid in samples:
                sample = annotations[guid]
                txt_posFacts = [txt['fact'] for txt in sample['txt_posFacts']]
                txt_negFacts = [txt['fact'] for txt in sample['txt_negFacts']]
                
                # txt_snippet_ids
                id_posFacts = [txt['snippet_id'] for txt in sample['txt_posFacts']]
                id_negFacts = [txt['snippet_id'] for txt in sample['txt_negFacts']]
                txt_snippet_ids += id_posFacts + id_negFacts 
                
                
                txts = txt_posFacts + txt_negFacts
                tokenized_text = clip.tokenize(txts).cuda()
                
                for i in range(0, tokenized_text.shape[0], batch_size):
                    if i + batch_size < tokenized_text.shape[0]:
                        tokenized_text_batch = tokenized_text[i : i + batch_size]
                    else:
                        tokenized_text_batch = tokenized_text[i:]
                    text_features = self.model.encode_text(tokenized_text_batch)
                    txt_features.append(text_features)
            txt_features = torch.cat(txt_features).cpu()
            txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True).to(torch.float32).cpu()

            # add index
            index = self.faiss_add(txt_features)
            
            # save index        
            faiss.write_index(index, output_path)
            txt_df = pd.DataFrame({'index': range(len(txt_snippet_ids)),
                                          "snniped_id": txt_snippet_ids})
            txt_df.to_csv("txt_df.csv", index=False)
                
if __name__ == "__main__":
    db = VectorDB()
    db.indexing_image(img_dir='images',
                      output_path="img_vector_db.faiss")
    
    # db.indexing_image(samples_path='samples.txt',
    #                   annotation_path="extracted_images/WebQA_data/WebQA_train_val.json",
    #                   output_path=""
    #                   )
    
    # db = VectorDB(txt_sample_path=None,
    #               image_sample_path="image_df.csv",
    #               txt_index_path=None,
    #               image_index_path="img_vector_db.faiss")
    
    # while True:
    #     question = "bear boltle"
    #     # result_img, result_txt = db.search(question)
    #     result_img = db.search(question)
    #     print(result_img)
    #     input("Next")
    #     break
    