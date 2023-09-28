# Aerial-Semantic-Segmentation

This package contains a UNET CNN model slightly modified from [This Biomedical Image Processing Model](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) to semantically segment multiple claasses. 

The package also contains images and masks from the [Aerial Drone Image Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset/code) from Kaggle. 

# How to Use
## Package Dependencies
```
torch
torchvision
tqdm
pandas
os
PIL
numpy
matlplotlib
```
## First Clone
Open ```train.py``` and replace the ```rootfolder``` with the path the directory you cloned this repo in.
```py
# Setup Paths
aerialPaths = pathDirectory(rootFolder="S:\Aerial-Semantic-Segmentation")
aerialPaths.summarizeDataset()
aerialPaths.showMap()
```
## Training the Model
To train the model, set the ```LOAD_MODEL``` flag to ```False``` 

## Saving the Model
To Save the model checkpoints for loading set the ```SAVE_MODEL``` flag to ```True```, this will trigger the script to save a checkpoint_x.pth file in ```Aerial-Semantic-Segmentation/logs/checkpoints/```

## Loading Saved Models
To Load a saved model for evaluation, set the ```LOAD_MODEL``` flag to ```True``` and input a run number for the model you wish to load from  ```Aerial-Semantic-Segmentation/logs/checkpoints/```
```py
# Load Checkpoint
checkpoint = loadModel(modelIndex=1)
```
