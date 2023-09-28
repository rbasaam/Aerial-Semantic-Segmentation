import torch
from tqdm import tqdm
import torch.nn as nn 
import torch.optim as optim
from model import UNET
import pandas as pd
from utils import(
    pathDirectory,
    get_loaders,
    calculate_iou,
    save_predictions_as_imgs,
    dataLogger,
    loadModel,
    showPreds,
)

# Setup Paths
aerialPaths = pathDirectory(rootFolder="S:\Aerial-Semantic-Segmentation")
aerialPaths.summarizeDataset()
aerialPaths.showMap()

# Flags
LOAD_MODEL    = False       # Load Trained Model
SAVE_MODEL    = True        # Save Trained Model
LOG_FILES     = True        # Log Training and Validation Losses and Accuracies

# Device and Data Loading
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY    = True
NUM_WORKERS   = 8

# Hyperparameters
LEARNING_RATE = 1e-8        # Learning Rate Step Size
PATCH_SIZE    = 512         # PATCH_SIZE x PATCH_SIZE images
BATCH_SIZE    = 128         # Number of Images per Batch
NUM_BATCHES   = 100         # Number of batches to train on
NUM_EPOCHS    = 5           # Number of times to loop over the batches

# Training Function
def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader, total=NUM_BATCHES)
    for batch_id, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.long().to(device=device)
        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Update TQDM Loop
        loop.set_postfix(loss=loss.item())
    return loss.item()

# Main Function
def main():

    model = UNET(in_channels=3, out_channels=aerialPaths.num_classes).to(device=DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    train_loader, val_loader = get_loaders(
        train_dir=aerialPaths.train_imgs,
        train_maskdir=aerialPaths.train_mask,
        val_dir=aerialPaths.val_imgs,
        val_maskdir=aerialPaths.val_mask,
        patch_size=PATCH_SIZE,
        batch_size=BATCH_SIZE,
        num_batches=NUM_BATCHES,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
   # Lists to Store Losses and Accuracies
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Load Trained and Saved Model
    if LOAD_MODEL: 
        print("\nAttempting to Load Model:\n")
        # Initialize Model
        model = UNET(in_channels=3, out_channels=aerialPaths.num_classes).to(device=DEVICE)
        # Load Checkpoint
        checkpoint = loadModel(modelIndex=1)
        # Load Checkpoint into Model
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Set Model to Evaluation Mode
        model.eval()
        print("Load Successful")
        # Show Predictions
        showPreds(folder=aerialPaths.SAV_FLDR, num=5)
    
    #Train Model
    elif LOAD_MODEL==False:
        print("Entering Training Phase\n")
        for epoch in range(NUM_EPOCHS): 
            # Perform Training on Training Set
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} Training")
            train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, device=DEVICE)
            # Perform Validation on Validation Set
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} Validation")
            val_loss = train_fn(val_loader, model, optimizer, loss_fn, scaler, device=DEVICE)
            # Append Losses and Accuracies to Lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(calculate_iou(train_loader, model, num_classes=aerialPaths.num_classes, device=DEVICE)) 
            val_accuracies.append(calculate_iou(val_loader, model, num_classes=aerialPaths.num_classes, device=DEVICE))
        
        # Save Model
        if SAVE_MODEL:
            print("\nAttempting to Save Model")
            # Create Checkpoint Dictionary
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            # Save Checkpoint
            torch.save(checkpoint, aerialPaths.checkpoint_filename)
            print(f"Saved {aerialPaths.checkpoint_filename} Successfuly\n")

            # Print Some Examples to Folder
            print("Saving Predictions to Folder")
            save_predictions_as_imgs(
                val_loader, model, classMap=aerialPaths.mappingDataframe, folder=aerialPaths.SAV_FLDR, device=DEVICE, num=20,
            )

        # Log Training, Validation Losses and Accuracies and Hyperparameters
        if LOG_FILES:
            print("Creating Log Files")
            # Hyperparameters Dictionary
            hyperparameters = {
                    "LEARNING_RATE": LEARNING_RATE,
                    "BATCH_SIZE": BATCH_SIZE,
                    "PATCH_SIZE": PATCH_SIZE,
                    "NUM_BATCHES": NUM_BATCHES,
                    "NUM_EPOCHS": NUM_EPOCHS,
                }
            hyperparameters = pd.DataFrame(hyperparameters, index=[0])           
            # Create Dataframe of Training Log
            trainLog = pd.DataFrame(
                {
                    "Epoch": range(1, NUM_EPOCHS + 1),
                    "TrainLoss": train_losses,
                    "ValidationLoss": val_losses,
                    "TrainAccuracy": train_accuracies,
                    "ValidationAccuracy": val_accuracies,
                }
            )
            # Create Log Files
            dataLogger(
                logFolder=aerialPaths.LOG_FLDR,
                trainLog=trainLog, 
                hyperparameters=hyperparameters
                )

if __name__ == "__main__":
    main()
