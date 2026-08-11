
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from transfer_model import TransferResNet
import os

# 1. TRANSFORMS

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = (0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = (0.4914, 0.4822, 0.4465),
        std = (0.2470, 0.2435, 0.2616)
    )
])

# 2. DATASETS

full_train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=False,
    transform=test_transform
)

# 3. TRAIN / VALIDATION SPLIT

train_size = 45000
val_size = 5000

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size]
)

# 4. DATALOADERS

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# 5. DATASET INFORMATION

print("Training:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# 6. DEVICE


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

# 7. MODEL

model = TransferResNet().to(device)

# 8. LOSS FUNCTION

criterion = nn.CrossEntropyLoss()

# 9. OPTIMIZER

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# 10. TRAINING SETTINGS

epochs = 15

best_val_accuracy = 0.0

os.makedirs(
    "saved_models",
    exist_ok=True
)

# 11. TRAINING LOOP

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Clear previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()


    
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    # Calculate validation accuracy
    val_accuracy = (
        100 * val_correct / val_total
    )
# best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "saved_models/best_model.pth"
        )

        best_message = " ← BEST MODEL SAVED"

    else:

        best_message = ""


    average_loss = (
        running_loss / len(train_loader)
    )

    print(
        f"epoch {epoch + 1:02d} | "
        f"loss = {average_loss:.4f} | "
        f"val accuracy = {val_accuracy:.2f}%"
        f"{best_message}"
    )

# 12. LOAD BEST MODEL

print("\nTraining complete.")

model.load_state_dict(
    torch.load(
        "saved_models/best_model.pth",
        map_location=device
    )
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

# 13. FINAL TEST EVALUATION

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


test_accuracy = (
    100 * correct / total
)

# 14. FINAL RESULTS


print(
    f"\nFinal test accuracy: "
    f"{test_accuracy:.2f}%"
)

print(
    "\nBest model saved at:"
    " saved_models/best_model.pth"
)

