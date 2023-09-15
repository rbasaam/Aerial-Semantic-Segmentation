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
    plot_loss_acc,
    showPreds,
)

# Home or Lab
aerialPaths = pathDirectory(workstation="Lab")
aerialPaths.summarizeDataset(printFlag=False)

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY    = True
NUM_WORKERS   = 8
LEARNING_RATE = 1e-5
BATCH_SIZE    = 64
PATCH_SIZE    = 1024  # 2048 Originally
NUM_BATCHES   = 20
NUM_EPOCHS    = 400
LOAD_MODEL    = False
SAVE_MODEL    = False
SAVE_PREDS    = False

hyperparameters = {
    "LEARNING_RATE": LEARNING_RATE,
    "BATCH_SIZE": BATCH_SIZE,
    "PATCH_SIZE": PATCH_SIZE,
    "NUM_BATCHES": NUM_BATCHES,
    "NUM_EPOCHS": NUM_EPOCHS,
}


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
   
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Load Trained and Saved Model
    if LOAD_MODEL: 
        print(f"\nAttempting to Load Model: {aerialPaths.checkpoint_filename}\n")
        model = UNET(in_channels=3, out_channels=aerialPaths.num_classes).to(device=DEVICE)
        if torch.cuda.is_available():
            checkpoint = torch.load(aerialPaths.checkpoint_filename)
        else:
            checkpoint = torch.load(aerialPaths.checkpoint_filename, map_location=torch.device(device=DEVICE))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        trainLog = checkpoint["TrainLog"]
        train_losses = checkpoint["TrainLoss"]
        val_losses = checkpoint["ValidationLoss"]
        train_accuracies = checkpoint["TrainAccuracy"]
        val_accuracies = checkpoint["ValidationAccuracy"]
        model.eval()
        print("Load Successful")
        plot_loss_acc(train_losses, val_losses, train_accuracies, val_accuracies, NUM_EPOCHS)
        showPreds(folder=aerialPaths.SAV_FLDR, num=5)
    
    #Train Model
    elif LOAD_MODEL==False:
        print("Entering Training Phase\n")
        for epoch in range(NUM_EPOCHS): 
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} Training")
            train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, device=DEVICE)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} Validation")
            val_loss = train_fn(val_loader, model, optimizer, loss_fn, scaler, device=DEVICE)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(calculate_iou(train_loader, model, num_classes=aerialPaths.num_classes, device=DEVICE)) 
            val_accuracies.append(calculate_iou(val_loader, model, num_classes=aerialPaths.num_classes, device=DEVICE))
        trainLog = pd.DataFrame(
            {
            "Epoch": range(1, NUM_EPOCHS + 1),
            "TrainLoss": train_losses,
            "ValidationLoss": val_losses,
            "TrainAccuracy": train_accuracies,
            "ValidationAccuracy": val_accuracies,
            }
        )
        dataLogger(
            logFolder=aerialPaths.LOG_FLDR,
            train_losses=train_losses,
            val_losses=val_losses, 
            train_accuracies=train_accuracies, 
            val_accuracies=val_accuracies, 
            hyperparameters=hyperparameters
            )
        if SAVE_MODEL:
            # Save Model
            print("\nAttempting to Save Model")
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "TrainLoss": train_losses, 
                "ValidationLoss": val_losses,
                "TrainAccuracy": train_accuracies,
                "ValidationAccuracy": val_accuracies,
                "TrainLog": trainLog,

            }
            torch.save(checkpoint, aerialPaths.checkpoint_filename)
            print("Save Successful\n")
        if SAVE_PREDS:
            # Print Some Examples to Folder
            print("Saving Predictions to Folder")
            save_predictions_as_imgs(
                val_loader, model, classMap=aerialPaths.mappingDataframe, folder=aerialPaths.SAV_FLDR, device=DEVICE, num=20,
            )

if __name__ == "__main__":
    main()
