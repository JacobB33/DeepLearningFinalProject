import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import json
from PIL import Image
import random

def get_train_dataset(processed_data_path='./data/processed_data', load_into_memory=True,
                      with_images=False):
    return NSDDataset(processed_data_path, load_into_memory, with_images)


def get_train_test_dataset(train_percentage, processed_data_path='./data/processed_data', load_into_memory=True,
                           with_images=False):
    if train_percentage == 1.0:
        return NSDDataset(processed_data_path, load_into_memory, with_images), None

    total_samples = len(json.load(open(f"{processed_data_path}/dataset.json", "r"))['annotations'])
    train_start = 0
    train_end = int(train_percentage * total_samples)
    test_end = -1
    return NSDDataset(processed_data_path, load_into_memory, with_images, train_start, train_end), \
        NSDDataset(processed_data_path, load_into_memory, with_images, train_end, test_end)


class NSDDataset(Dataset):
    def __init__(self, processed_data_path='./data/processed_data', load_into_memory=True, with_images=False):
        assert load_into_memory
        assert not with_images
        dataset_json = json.load(open(f"{processed_data_path}/dataset.json", "r"))
        self.load_into_memory = load_into_memory
        self.with_images = with_images
        self.processed_data_path = processed_data_path
        # self.process

        annotations = dataset_json["annotations"]
        annotations = annotations[0: 100]
        

        self.data_size = len(annotations)

        self.dataset = []
        self.image_embeddings = {}
        for annotation in annotations:
            beta = np.load(os.path.join(processed_data_path, annotation["beta"])).astype(np.float32)
            image_num = str(annotation["img"])

            if image_num not in self.image_embeddings:
                image_info = dataset_json["images"][image_num]
                captions = image_info["captions"]
                embeddings = []
                for capt_dict in captions:
                    embedding = np.load(os.path.join(processed_data_path, capt_dict["embd"]))
                    embeddings.append(torch.from_numpy(embedding))
                if not self.with_images:
                    # print(embeddings[0].shape)
                    self.image_embeddings[image_num] = embeddings
                    # if not len(self.image_embeddings[image_num]) == 5:
                    #     print(captions)
                    #     exit()
                    # print(self.image_embeddings[image_num].shape)

                # else:
                #     self.image_embeddings[image_num] = (
                #         embeddings, Image.open(os.path.join(processed_data_path, image_info['im_path'])))

            self.dataset.append((torch.from_numpy(beta), image_num))

    def __getitem__(self, idx):
        if self.load_into_memory:
            beta, image_num = self.dataset[idx]
            image_data = self.image_embeddings[image_num]
            return beta, image_data[0]
        assert False

    def __len__(self):
        return self.data_size


class NSDDatasetNonMemory(Dataset):
    def __init__(self, processed_data_path='./data/processed_data', load_into_memory=True, with_images=False, start=-1,
                 stop=-1):
        dataset_json = json.load(open(f"{processed_data_path}/dataset.json", "r"))
        self.load_into_memory = load_into_memory
        self.with_images = with_images
        self.processed_data_path = processed_data_path
        # self.process
        self.dataset_json = dataset_json
        if start == -1:
            annotations = dataset_json["annotations"]
        else:
            annotations = dataset_json["annotations"][start: len(dataset_json['annotations']) if stop == -1 else stop]
        self.data_size = len(annotations)
        if not load_into_memory:
            self.annotations = annotations
            return

        self.dataset = []
        for annotation in annotations:
            beta = np.load(os.path.join(processed_data_path, annotation["beta"]))
            image_num = str(annotation["img"])
            image_info = dataset_json["images"][image_num]
            captions = image_info["captions"]
            embeddings = []
            for capt_dict in captions:
                embedding = np.load(os.path.join(processed_data_path, capt_dict["embd"]))
                embeddings.append(embedding)
            if not self.with_images:
                self.dataset.append((beta, embeddings))
            else:
                image_path = image_info["im_path"]
                self.dataset.append((beta, embeddings, Image.open(os.path.join(processed_data_path, image_path))))

    def __getitem__(self, idx):
        if self.load_into_memory:
            return self.dataset[idx]

        beta = np.load(os.path.join(self.processed_data_path, self.annotations[idx]["beta"]))
        image_num = self.annotations[idx]["image"]
        image_info = self.dataset_json["images"][image_num]
        captions = image_info["captions"]
        embeddings = []
        for capt_dict in captions:
            embedding = np.load(os.path.join(self.processed_data_path, capt_dict["embd"]))
            embeddings.append(embedding)
        if not self.with_images:
            return beta, embeddings
        else:
            image_path = image_info["im_path"]
            return beta, embeddings, Image.open(os.path.join(self.processed_data_path, image_path))

    def __len__(self):
        return self.data_size


if __name__ == "__main__":
    test = NSDDataset(processed_data_path="/home/jacob/projects/DeepLearningFinalProject/data/processed_data")
    print('here')
