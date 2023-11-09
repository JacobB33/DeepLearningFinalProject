# Instructions to download the data

For this project, we need to have the aws cli installed. In order to do this run:
```bash
sudo apt install awscli
```

Then, we need to configure the aws cli with credentials. You can use any aws credentials (might need to create an account). For this, run:
```bash
aws configure
```

## Run these commands to download the data
Be sure to remove the --dryrun flag to actually download all of the data.
**Be sure to run this inside the data folder.**

### Download the nsd_data folder
```bash
aws s3 sync --dryrun s3://natural-scenes-dataset/nsddata ./nsd/nsddata --exclude "*func1mm*" --exclude "*subj02*" --exclude "*subj03*" --exclude "*subj04*" --exclude "*subj05*" --exclude "*subj06*" --exclude "*subj07*" --exclude "*subj08*"
```
This is 6gb

### command for priors folder:
```bash
aws s3 sync --dryrun s3://natural-scenes-dataset/nsddata_betas ./nsd/nsddata_betas --exclude "*func1mm*" --exclude "*subj02*" --exclude "*subj03*" --exclude "*subj04*" --exclude "*subj05*" --exclude "*subj06*" --exclude "*subj07*" --exclude "*subj08*"
```
This is 500 gb

### Command for stimuli folder:
```bash
aws s3 sync --dryrun s3://natural-scenes-dataset/nsddata_stimuli ./nsd/nsddata_stimuli --exclude "*func1mm*" --exclude "*subj02*" --exclude "*subj03*" --exclude "*subj04*" --exclude "*subj05*" --exclude "*subj06*" --exclude "*subj07*" --exclude "*subj08*"
```
This is 60 gb
