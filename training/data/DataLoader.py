import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import json
from PIL import Image

class NSDDataset(Dataset):
    def __init__(self, processed_data_path = './data/processed_data', load_into_memory=True, with_images=False):
        dataset_json = json.load(open(f"{processed_data_path}/dataset.json", "r"))
        self.load_into_memory = load_into_memory
        self.with_images = with_images
        self.process
        
        annotations = dataset_json["annotations"]
        self.data_size = len(annotations)
        if not load_into_memory:
            self.annotations = annotations
            return
        
        self.dataset = []
        for annotation in annotations:
            beta = np.load(annotation["beta"])
            image_num = annotation["image"]
            image_info = dataset_json["images"][image_num]
            captions = image_info["captions"]
            embeddings = []
            for capt_dict in captions:
                embedding = np.load(capt_dict["embd"])
                embeddings.append(embedding)
            if not self.with_images:
                self.dataset.append((beta, embeddings))
            else:
                image_path = image_info["im_path"]
                self.dataset.append((beta, embeddings, Image.open(image_path)))
    
    def __getitem__(self, idx):
        if self.load_into_memory:
            return self.dataset[idx]
        
        beta = np.load(self.annotations[idx]["beta"])
        image_num = self.annotations[idx]["image"]
        image_info = self.dataset_json["images"][image_num]
        captions = image_info["captions"]
        embeddings = []
        for capt_dict in captions:
            embedding = np.load(capt_dict["embd"])
            embeddings.append(embedding)
        if not self.with_images:
            return beta, embeddings
        else:
            image_path = image_info["im_path"]
            return beta, embeddings, Image.open(image_path)
        
    
    def __len__(self):
        return self.data_size