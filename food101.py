import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import Food101
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from torch.optim import Adam
from torch import nn

from torch.amp import autocast



SAMPLE_SHAPE = (224, 224)
NO_EPOCHS = 100
BATCH_SIZE = 32
NO_CLASSES = 101
SAMPLES_PER_TRAIN = 750
SAMPLES_PER_TEST = 250
LEARNING_RATE = 1e-3
NUM_WORKERS = 8

MEAN, STD = [0.544981, 0.44349139, 0.34361495], [0.27094177, 0.27345256, 0.27805459]

train_transform = transforms.Compose([
    
    transforms.Resize(size=SAMPLE_SHAPE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomRotation(degrees=45),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD) # stats from scritps/compute_stats.py

])

test_transform = transforms.Compose([
    transforms.Resize(size=SAMPLE_SHAPE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

train_data = Food101(
    root='./',
    split='train',
    transform=train_transform
)

test_data = Food101(
    root='./',
    split='test',
    transform=test_transform
)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

if __name__ == '__main__' :
    
    len_train_loader = SAMPLES_PER_TRAIN * NO_CLASSES // BATCH_SIZE
    len_test_loader = SAMPLES_PER_TEST * NO_CLASSES // BATCH_SIZE

    model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)

    # print(model.fc.in_features, model.fc.out_features) (2048, 1000)

    # ### Freeze backbone

    # for param in model.parameters() :
    #     param.requires_grad = False

    model.fc = nn.Linear(in_features=2048, out_features=NO_CLASSES)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(params=model.parameters(), lr = LEARNING_RATE)

    writer = SummaryWriter()

    for epoch in range(NO_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_loader, total=len_train_loader, leave=True, desc=f"Epoch {epoch+1}/{NO_EPOCHS}") as pbar:
            for X, y in pbar:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()

                with autocast(device_type='cuda', dtype=torch.float16) :
                    y_pred = model(X)
                    loss = loss_fn(y_pred, y)
                
                loss.backward()
                optimizer.step()

                # Stats
                running_loss += loss.item()
                _, predicted = y_pred.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                pbar.set_postfix({
                    "loss": f"{running_loss / total:.4f}",
                    "acc": f"{100. * correct / total:.2f}%"
                })
            
        writer.add_scalar('Loss / train', running_loss / total, epoch)
        writer.add_scalar('Acc / train', correct / total, epoch)

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad() :
            with tqdm(test_loader, total=len_test_loader, leave=True, desc=f"Epoch {epoch+1}/{NO_EPOCHS}") as pbar:
                for X, y in pbar:
                    X, y = X.to(device), y.to(device)

                    y_pred = model(X)
                    loss = loss_fn(y_pred, y)

                    # Stats
                    running_loss += loss.item()
                    _, predicted = y_pred.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()

                    pbar.set_postfix({
                        "loss": f"{running_loss / total:.4f}",
                        "acc": f"{100. * correct / total:.2f}%"
                    })

        writer.add_scalar('Loss / test', running_loss / total, epoch)
        writer.add_scalar('Acc / test', correct / total, epoch)
    

    writer.flush()
    writer.close()


