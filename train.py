import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import os

# Assuming your get_dataloaders is in data.py
from data import get_dataloaders

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    # Progress bar for the training batches
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
    
    for i, (X, y) in pbar:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        correct += (y_pred.argmax(dim=1) == y).sum().item()
        total += X.size(0)
        
        # Update progress bar every step so you don't get bored
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Evaluating", leave=False):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)

            running_loss += loss.item() * X.size(0)
            correct += (y_pred.argmax(dim=1) == y).sum().item()
            total += X.size(0)

    return running_loss / total, correct / total

if __name__ == '__main__':
    # --- Configuration ---
    EPOCHS = 100
    BATCH_SIZE = 8 # Stop using 13. Just stop.
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VAL_INTERVAL = 1 # How often to run evaluation. 1 = every epoch.
    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Data & Model ---
    train_loader, test_loader = get_dataloaders('./', batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Linear(2048, 94)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(log_dir='runs/resnet50_experiment')

    best_acc = 0.0

    # --- Main Loop ---
    for epoch in range(EPOCHS):
        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch, EPOCHS)
        
        # Logging Train metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        # Evaluation
        if (epoch + 1) % VAL_INTERVAL == 0:
            test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
            
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)

            # Checkpoint the "best" model so you don't lose everything when Windows decides to update
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")

            print(f"Epoch [{epoch+1:03d}/{EPOCHS}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} "
                  f"{' (New Best!)' if test_acc == best_acc else ''}")

    writer.close()
    print("Training complete. Try not to break the deployment.")