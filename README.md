# Diffusion From Brain Scans
To clone, first, git clone and  then run these commands in the repo:

```bash
git submodule init
git submodule update
conda env create -f environment.yml
pip install -e .
```

Then download the stable diffusion weights from [here](https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt) and put them into the checkpoint folder. The processed data from the repo can be downloaded with instructions from the processed data folder.

The model checkpoints are available [here](https://drive.google.com/file/d/158UAf4_wozh7vF4RxJTEdQt6XQ2gq8IP/view?usp=sharing).
Once you have these downloaded, you can attempt to run an inference. The way to do that is to run the following command:

```bash
python ./StableDiffusionFork/scripts/brain2img.py \
    --n_samples 1 \
    --ckpt /home/jacob/projects/DeepLearningFinalProject/StableDiffusionFork/checkpoints/512-base-ema.ckpt \
    --config ./StableDiffusionFork/configs/stable-diffusion/v2-inference.yaml \
    --idx DATASET_IDX_TO_USE
```
Make sure the data is unzipped inside of the data folder to run the above commands.
## Code included in this repo from:
- https://github.com/tknapen/nsd_access
