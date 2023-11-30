import numpy as np
import os
import json
import fsspec
import torch
from training.configs import ModelConfig, Snapshot
from training.networks.encoder import BrainScanEmbedder

data_path = './data/processed_data'
data_json = './data/processed_data/dataset.json'
data_json = json.load(open(data_json, 'r'))
def load_bs_embedder():
    try:
        snapshot = fsspec.open('/home/jacob/projects/DeepLearningFinalProject/zcheckpoints/attention-1.pt')
        with snapshot as f:
            snapshot_data = torch.load(f, map_location="cpu")
    except FileNotFoundError:
        print("Snapshot not found")
        raise FileNotFoundError

    snapshot = Snapshot(**snapshot_data)
    embedder = BrainScanEmbedder(ModelConfig([8, 16, 32, 48, 77], -1, 8))
    embedder.load_state_dict(snapshot.model_state)
    return embedder.to('cuda')

embedder = load_bs_embedder()

annotations = data_json['annotations']
root_annotation = annotations[0]
root_image_id = str(root_annotation['img'])
root_image_info = data_json['images'][root_image_id]
root_caption_numpy = np.load(os.path.join(data_path, root_image_info['captions'][0]['embd']))
root_caption_string = root_image_info['captions'][0]['cap']
root_beta = torch.from_numpy(np.load(os.path.join(data_path, root_annotation['beta'])).astype(np.float32)).to('cuda').unsqueeze(0)
root_embed = embedder(root_beta).squeeze(0).detach().cpu().numpy()
print(root_caption_numpy.shape)
print(root_caption_string)
for i in range(1, 3):
    new_annotation = annotations[i]
    new_image_id = str(new_annotation['img'])
    new_image_info = data_json['images'][new_image_id]
    new_caption_numpy = np.load(os.path.join(data_path, new_image_info['captions'][0]['embd']))
    new_caption_string = new_image_info['captions'][0]['cap']
    new_beta = torch.from_numpy(np.load(os.path.join(data_path, new_annotation['beta'])).astype(np.float32)).to('cuda').unsqueeze(0)
    new_embed = embedder(new_beta).squeeze(0).detach().cpu().numpy()
    print(new_caption_numpy.shape)
    print(new_caption_string)
    print(f"dist = {np.linalg.norm(root_caption_numpy-new_caption_numpy)}")
    print(f'embed dist from eachotehr = {np.linalg.norm(root_embed-new_embed)}')
    print(f'new from root caption = {np.linalg.norm(root_caption_numpy-new_embed)}')
    print(f'min = {new_embed.min()}')
    print(f'max = {new_embed.max()}')
    