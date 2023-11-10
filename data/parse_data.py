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


def main(subject, processed_data_path="data/processed_data/"):
    setup_dirs(processed_data_path=processed_data_path)
    embedding_model = FrozenOpenCLIPEmbedder(freeze=True, layer="penultimate").to(
        "cuda"
    )

    nsd = NSDAccess("data/nsd/")

    subject_behaviors = pd.DataFrame()
    for i in range(1, 38):
        subject_behaviors = pd.concat(
            [subject_behaviors, nsd.read_behavior(subject, i)]
        )

    # This is the data structure that we will be saving to a file. See the readme for more information
    dataset = {"annotations": [], "images": {}}

    # It is 1 indexed I think. It has a value of 73000, even though we have the max index of stim_discriptions is 72999
    # These can be used with nsd.read_image_coco_info to get the image information
    stimulus = subject_behaviors["73KID"] - 1

    trial_number = 0
    for i in tqdm.tqdm(range(1, 38)):
        betas = nsd.read_betas("subj01", session_index=i)
        for index in range(len(betas)):
            beta = betas[
                index
            ]  # numpy array of length 750 which is the voxels of brain activity
            image_id = int(stimulus[trial_number])
            if not image_id in dataset["images"]:
                coco_info = nsd.read_image_coco_info([image_id])
                captions = [info["caption"] for info in coco_info]
                embeddings = get_text_embeddings(captions, embedding_model)
                image_to_embedding_list = []
                for i in range(len(embeddings)):
                    save_path = os.path.join(
                        processed_data_path,
                        "embeddings",
                        f"img{image_id}_caption_{i}.npy",
                    )
                    np.save(save_path, embeddings[i])
                    image_to_embedding_list.append(
                        {"caption": captions[i], "embedding_path": save_path}
                    )
                dataset["images"][image_id] = image_to_embedding_list

            beta_path = os.path.join(
                processed_data_path, "betas", f"trial_{trial_number}.npy"
            )
            np.save(beta_path, beta)
            dataset["annotations"].append(
                {"trial": trial_number, "image": image_id, "beta_path": beta_path}
            )
            trial_number += 1

    json.dump(dataset, open(f"{processed_data_path}/dataset.json", "w"))


if __name__ == "__main__":
    main("subj01")
