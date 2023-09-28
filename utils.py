import torch
import os
from dataset import aerialSemantics, BatchPatchLoader
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor

class pathDirectory():
    # Define Relative Paths from Root Folder
    def __init__(self, rootFolder="Aerial-Semantic-Segmentation\\"):

        self.mappingPath = os.path.join(rootFolder, "class_dictionary.csv")
        self.mappingDataframe = pd.read_csv(self.mappingPath, header=0)
        self.num_classes = len(self.mappingDataframe)

        self.train_imgs  = os.path.join(rootFolder, "data\\train\\train_img")
        self.train_mask  = os.path.join(rootFolder, "data\\train\\train_mask")
        self.train_segs  = os.path.join(rootFolder, "data\\train\\train_seg")
        self.val_imgs    = os.path.join(rootFolder, "data\\val\\val_img")
        self.val_mask    = os.path.join(rootFolder, "data\\val\\val_mask")
        self.val_segs    = os.path.join(rootFolder, "data\\val\\val_seg")
        self.test_imgs   = os.path.join(rootFolder, "data\\test\\test_img")
        self.test_mask   = os.path.join(rootFolder, "data\\test\\test_mask")
        self.test_segs   = os.path.join(rootFolder, "data\\test\\test_seg")        
        self.SAV_FLDR    = os.path.join(rootFolder, "data\\saved_imgs")
        self.LOG_FLDR    = os.path.join(rootFolder, "logs\\")

        checkpointIndex = len(os.listdir(os.path.join(rootFolder, "logs\\checkpoints")))
        self.checkpoint_filename = os.path.join(rootFolder, f"logs\\checkpoints\\checkpoint_{checkpointIndex+1}.pth")

    def showMap(self):
        print("\n ----------------------- Class Map -----------------------\n")
        print(self.mappingDataframe)
        print("------------------------------------------------------------\n\n")
        return 
    
    def imageChecker(self, folder="train", index=0):
        if folder == "train":
            image_dir = os.listdir(self.train_imgs)
            mask_dir = os.listdir(self.train_mask)
            segs_dir = os.listdir(self.train_segs)
            
            image = Image.open(os.path.join(self.train_imgs, image_dir[index])).convert("RGB")
            mask = Image.open(os.path.join(self.train_mask, mask_dir[index])).convert("L")
            segs = Image.open(os.path.join(self.train_segs, segs_dir[index])).convert("RGB")

        elif folder == "validation":
            image_dir = os.listdir(self.val_imgs)
            mask_dir = os.listdir(self.val_mask)
            segs_dir = os.listdir(self.val_segs)
            
            image = Image.open(os.path.join(self.val_imgs, image_dir[index])).convert("RGB")
            mask = Image.open(os.path.join(self.val_mask, mask_dir[index])).convert("L")
            segs = Image.open(os.path.join(self.val_segs, segs_dir[index])).convert("RGB")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Mask Image")
        
        axes[2].imshow(segs)
        axes[2].set_title("Segmentation Image")

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        return 
    
    def summarizeDataset(self):
        image_dir = os.listdir(self.train_imgs)
        image_path = os.path.join(self.train_imgs, image_dir[0])
        image = Image.open(image_path)

        print("\n ----------------------- Dataset Summary -----------------------\n")
        print(f"Image Size:                    {image.size[0]} x {image.size[1]}")
        print(f"Number of Classes to Identify: {self.num_classes}")
        print(f"Length of Training Set:        {len(os.listdir(self.train_imgs))}")
        print(f"Length of Validation Set:      {len(os.listdir(self.val_imgs))}")
        print(f"Length of Test Set:            {len(os.listdir(self.test_imgs))}")
        print("--------------------------------------------------------------------")
        print("\n\n")
        return

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    patch_size,
    batch_size,
    num_batches=None,  # Number of batches per epoch
    num_workers=4,
    pin_memory=True,
):
    train_ds = aerialSemantics(
        image_dir=train_dir,
        mask_dir=train_maskdir,
    )

    train_loader = BatchPatchLoader(
        dataset=train_ds,
        batch_size=batch_size,
        num_batches=num_batches,
        patch_size=patch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_ds = aerialSemantics(
        image_dir=val_dir,
        mask_dir=val_maskdir,
    )

    val_loader = BatchPatchLoader(
        dataset=val_ds,
        batch_size=batch_size,
        num_batches=num_batches,
        patch_size=patch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
  
def calculate_iou(loader, model, num_classes, device="cuda"):
    iou_sum = 0
    total_batches = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, predicted_labels = torch.max(outputs, 1)

            iou_batch = 0
            for class_idx in range(1, num_classes):  # Skip background class (usually index 0)
                intersection = torch.logical_and(predicted_labels == class_idx, y == class_idx).sum().item()
                union = torch.logical_or(predicted_labels == class_idx, y == class_idx).sum().item()
                iou = intersection / (union + 1e-10)  # Add a small epsilon to avoid division by zero
                iou_batch += iou

            iou_batch /= (num_classes - 1)  # Calculate average IoU over all classes except background
            iou_sum += iou_batch
            total_batches += 1
    model.train()
    mean_iou = iou_sum / total_batches
    return mean_iou

def save_predictions_as_imgs(
        loader, model, classMap: pd.DataFrame, folder="saved_imgs/", device="cuda", num=10,
        ):
    model.eval()
    cols = classMap.columns
    for batch_idx, (x_batch, y_batch) in enumerate(loader):
        if batch_idx<num:
            x_batch = x_batch.to(device=device)
            with torch.no_grad():
                preds_batch = torch.sigmoid(model(x_batch))

            # Save the first image from the batch
            pred_image = preds_batch[0]
            num_classes, img_height, img_width = pred_image.shape
            predArray = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            class_predictions = torch.argmax(pred_image, dim=0).cpu().numpy()

            def colorassign(c):
                mask = class_predictions == c+1
                predArray[mask, 0] = classMap.iloc[c][cols[1]]
                predArray[mask, 1] = classMap.iloc[c][cols[2]]
                predArray[mask, 2] = classMap.iloc[c][cols[3]]

            with ThreadPoolExecutor() as executor:
                executor.map(colorassign, range(num_classes))

            # Save the image with a filename containing the batch index
            predFile = f"predictions\\batch_{batch_idx+1}_prediction.png"
            predPath = os.path.join(folder, predFile)
            predImg = Image.fromarray(predArray)
            predImg.save(predPath)

            # Save the image with a filename containing the batch index
            inputFile = f"inputs\\batch_{batch_idx+1}_input.png"
            inputPath = os.path.join(folder, inputFile)
            # Convert the input image data to uint8
            x = (x_batch[0]).cpu().numpy().astype(np.uint8)
            inputImg = Image.fromarray(np.transpose(x, (1, 2, 0))).convert("RGB")  # Assuming image format (C, H, W)
            inputImg.save(inputPath)
            print(f"Saved Prediction: {batch_idx+1}")
        
def showPreds(folder="saved_imgs/", num=3):
    predsFolder = os.path.join(folder, "predictions/")
    inputsFolder = os.path.join(folder, "inputs/")

    preds = os.listdir(predsFolder)
    inputs = os.listdir(inputsFolder)

    rand_idx = random.sample(range(0, len(preds)), num)
    fig, axs = plt.subplots(num, 2, figsize=(num, 4))

    for i in range(num):
        pred_idx = rand_idx[i]
        input_idx = rand_idx[i]

        pred_path = os.path.join(predsFolder, preds[pred_idx])
        input_path = os.path.join(inputsFolder, inputs[input_idx])

        pred_img = Image.open(pred_path)
        input_img = Image.open(input_path)

        axs[i, 1].imshow(pred_img)
        axs[i, 1].set_title(f"Prediction {pred_idx}")

        axs[i, 0].imshow(input_img)
        axs[i, 0].set_title(f"Input {input_idx}")

        axs[i, 0].axis("off")
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

def dataLogger(
        logFolder,
        trainLog, 
        hyperparameters
        ):
    print("\n Logging Data")

    trainingFolder = os.path.join(logFolder, "Training")
    current = len(os.listdir(trainingFolder))+1

    # Create Training Log and Hyperparameters Files
    trainingFile = open(os.path.join(trainingFolder, f"TrainingLog_0{current}.csv"), "x")
    hyperparametersFile = open(os.path.join(logFolder, "Hyperparameters.csv"), "a")

    # Save the training log to a CSV file
    trainLog.to_csv(trainingFile, index=False, mode="w", header=True, lineterminator='\n')
    hyperparameters.to_csv(hyperparametersFile, index=False, mode="a", header=False, lineterminator='\n')
    
    # Close Files
    trainingFile.close()
    hyperparametersFile.close()

    print(f"Log Complete for Run {current}")
    return

def loadModel(modelIndex=1):
    checkpointPath = f"logs\\checkpoints\\checkpoint_{modelIndex}.pth"
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpointPath)
    else:
        checkpoint = torch.load(checkpointPath, map_location=torch.device('cpu'))
    return checkpoint