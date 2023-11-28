# Instructions to process the data
This is the instructions for downloading the data from the NSD or natural scene dataset. This is from this paper: [Allen, St-Yves, Wu, Breedlove, Prince, Dowdle, Nau, Caron, Pestilli, Charest, Hutchinson, Naselaris*, & Kay*. A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nature Neuroscience (2021).](https://doi.org/10.1038/s41593-021-00962-x)

To process the data run the following command:
```bash
python process_data.py
```
This assumes that the folder structure of the nsd dataset is located at`.`/data/nsd``
The data is processed into a dictionary object that is of the following form (paths are relative from the processed data root):

```python
{
    "annotations": List[
        {
            "session_number": int, # Session number
            "trial": int, # Trial number
            "img": int, # Image Number
            "beta": str, # Path to the beta file
        }
    ] # List of annotations, from trial 0 to the end

    "images": {
        "image_num": { # The image number (integer)
            "im_path": str, # Path to the image file
            "captions": [
                    {
                        "cap": str, # A caption for the image
                        "embd": str, # Path to the embedding file 
                    }
                ]
        }
    }
}

```
The processed image caption embeddings are stored in the folder ``./data/processed_data/embeddings`` and the betas of the ventral lobe (a numpy array of length 7604) is stored in the folder ``./data/processed_data/betas/session_i/trial_j``. The json described above is saved at ``./data/processed_data/annotations.json``. The image files are saved to ``./data/processed_data/images``. The processed data is not included in the repository due to its size.

## Pre Processed Data
You can find a current version of the processed data at [the following location](https://drive.google.com/file/d/1lLWr0C8mjgnUq0lFIUmEKUqjsh7jzI6D/view?usp=sharing). Use the following command to download it:
```bash
sudo apt install gdown
cd data
gdown https://drive.google.com/uc?id=1lLWr0C8mjgnUq0lFIUmEKUqjsh7jzI6D
```
Then unzip it:
```bash
sudo apt install unzip
unzip processed_data.zip
```

# Instructions to download the raw data

For this project, we need to have the aws cli installed. To do this run:
```bash
sudo apt install awscli
```

Then, we need to configure the aws cli with credentials. You can use any aws credentials (might need to create an account). For this, run:
```bash
aws configure
```

## Run these commands to download the data
Be sure to remove the --dryrun flag to actually download all of the data. You will get a lot of access denied which is fine (some of the files are not yet released). This only downloads the data relevant to subject 1 and for the experiments we need. I think that there is extra data included but I don't know how to remove it.
**Be sure to run this inside the data folder.**

### Download the nsd_data folder
```bash
aws s3 sync --dryrun s3://natural-scenes-dataset/nsddata ./nsd/nsddata --exclude "*func1mm*" --exclude "*subj02*" --exclude "*subj03*" --exclude "*subj04*" --exclude "*subj05*" --exclude "*subj06*" --exclude "*subj07*" --exclude "*subj08*"
```
This is 6gb

### Command for priors folder:
```bash
aws s3 sync --dryrun s3://natural-scenes-dataset/nsddata_betas ./nsd/nsddata_betas --exclude "*func1mm*" --exclude "*subj02*" --exclude "*subj03*" --exclude "*subj04*" --exclude "*subj05*" --exclude "*subj06*" --exclude "*subj07*" --exclude "*subj08*"
```
This is 500 gb

### Command for stimuli folder:
```bash
aws s3 sync --dryrun s3://natural-scenes-dataset/nsddata_stimuli ./nsd/nsddata_stimuli --exclude "*func1mm*" --exclude "*subj02*" --exclude "*subj03*" --exclude "*subj04*" --exclude "*subj05*" --exclude "*subj06*" --exclude "*subj07*" --exclude "*subj08*"
```
This is 60 gb
