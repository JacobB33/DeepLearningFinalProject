from typing import List
from data.nsd_access import NSDAccess
import scipy.io
import pandas as pd
import numpy as np
import torch
from StableDiffusionFork.ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder
import os
import json
import tqdm

@torch.no_grad()
def get_text_embeddings(
    captions: List[str], model: FrozenOpenCLIPEmbedder
) -> List[np.ndarray]:
    """This method gets the text embeddings using the frozen clip embedder. They are returned as
    a list in the order that they were given in the captions list

    Args:
        captions (List[str]): This is the captions to embedd
        model (FrozenOpenCLIPEmbedder): This is the model to embed them with. I got it from the embedder for textToImage

    Returns:
        List[np.ndarray]: This returns a list of embeddings such that the ith element is the embedding for the ith caption
    """
    results = model(captions).cpu().numpy()
    return [results[i] for i in range(len(results))]


def setup_dirs(processed_data_path: str) -> None:
    """Creates the directories for the processed data. Currently creates
    data/processed_data/embeddings and data/processed_data/betas if they do not exist

    Args:
        processed_data_path (str): Path to the processed data root directory
    """
    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)
    if not os.path.exists(os.path.join(processed_data_path, "embeddings")):
        os.mkdir(os.path.join(processed_data_path, "embeddings"))
    if not os.path.exists(os.path.join(processed_data_path, "betas")):
        os.mkdir(os.path.join(processed_data_path, "betas"))
        
    for i in range(1, 38):
        if not os.path.exists(os.path.join(processed_data_path, "betas", f"ses_{i}")):
            os.mkdir(os.path.join(processed_data_path, "betas", f"ses_{i}"))



def main(subject, processed_data_path="data/processed_data/"):
    setup_dirs(processed_data_path=processed_data_path)
    embedding_model = FrozenOpenCLIPEmbedder(freeze=True, layer="penultimate").to(
        "cuda"
    )

    nsd = NSDAccess("data/nsd/")

    subject_behaviors = pd.DataFrame()
    for session_num in range(1, 38):
        # print(len(subject_behaviors))
        subject_behaviors = pd.concat(
            [subject_behaviors, nsd.read_behavior(subject, session_num)]
        )
    
    # This is the data structure that we will be saving to a file. See the readme for more information
    dataset = {"annotations": [], "images": {}}

    # It is 1 indexed I think. It has a value of 73000, even though we have the max index of stim_discriptions is 72999
    # These can be used with nsd.read_image_coco_info to get the image information
    stimulus = subject_behaviors["73KID"] - 1
    stimulus = stimulus.to_numpy()

    
    for session_num in tqdm.tqdm(range(1, 38), colour='blue'):
        betas = nsd.read_betas("subj01", session_index=session_num, data_type="betas_fithrf_GLMdenoise_RR", data_format='func1pt8mm')
        mask, atlas_dict = nsd.read_atlas_results('subj01', 'streams', data_format='func1pt8mm')
        ventral_mask = mask.transpose([2, 1, 0]) == atlas_dict['ventral']
        ventral_betas = betas[:, ventral_mask]
        trial_stimulus = stimulus[(session_num-1)*750:(session_num)*750]
        coco_annotations = nsd.read_image_coco_info(trial_stimulus)
        
        for trial_number in tqdm.tqdm(range(len(betas)), colour='green'):
            
            # numpy array of length 7604 which is the voxels of brain activity
            ventral_beta = ventral_betas[trial_number]
            assert len(ventral_beta) == 7604
            
            
            image_id = int(trial_stimulus[trial_number])
            
            if not image_id in dataset["images"]:
                # we have not embedded this image yet
                
                coco_info = coco_annotations[trial_number]
                captions = [info["caption"] for info in coco_info]
                embeddings = get_text_embeddings(captions, embedding_model)
                image_to_embedding_list = []
                for k in range(len(embeddings)):
                    save_path = os.path.join(
                        processed_data_path,
                        "embeddings",
                        f"img{image_id}_caption_{k}.npy",
                    )
                    np.save(save_path, embeddings[k])
                    image_to_embedding_list.append(
                        {"cap": captions[k], "embd": save_path}
                    )
                    
                dataset["images"][image_id] = image_to_embedding_list

            beta_path = os.path.join(
                processed_data_path, f"betas/ses_{session_num}", f"trial_{trial_number}.npy"
            )
            np.save(beta_path, ventral_beta)
            dataset["annotations"].append(
                {"session_number": session_num, "trial": trial_number, "img": image_id, "beta": beta_path}
            )

    
        json.dump(dataset, open(f"{processed_data_path}/dataset.json", "w"))


if __name__ == "__main__":
    # nsd = NSDAccess("data/nsd/")
    # print(nsd.read_betas('subj01', 1, data_type='betas_fithrf_GLMdenoise_RR', data_format='func1pt8mm').shape)
    
    # print(nsd.read_betas('subj01', 1, [1], data_type='betas_fithrf_GLMdenoise_RR', data_format='func1pt8mm').shape)
    main("subj01")
