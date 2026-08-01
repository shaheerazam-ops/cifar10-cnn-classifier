import torch 
import torch.nn as nn 
from torchvision import transforms , datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import torch.optim as optim
from model import CNN
import os

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root = "data",
    train = True,
    download = True,
    transform=transform 
)

test_dataset = datasets.CIFAR10(
    root="data",
    train = False ,
    download = True,
    transform = transform 
)

train_loader = DataLoader(
    train_dataset,
    batch_size = 64,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 64,
    shuffle = False
)

print(len(train_dataset))
print(len(test_dataset))

images , labels = next (iter(train_loader))
print(images.shape)
print(labels.shape)

classes = train_dataset.classes

print("categories:",classes)
"""
plt.imshow(images[0].permute(1,2,0))
plt.title(classes[labels[0]])
plt.axis("off")
plt.show()
"""
"""
from model import CNN

model = CNN()

print(model)

outputs = model(images)

print(outputs.shape)
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5 

for epoch in range (epochs):

    model.train()

    running_loss= 0 

    for images , labels in train_loader:
         images = images.to(device)
         labels = labels.to(device)

         outputs = model(images)

         loss = criterion(outputs,labels)

         optimizer.zero_grad()

         loss.backward()

         optimizer.step()

         running_loss += loss.item()

    print(
        f"epoch {epoch+1} | loss = {running_loss/len(train_loader):.4f}"
    )     


model.eval()
correct = 0 
total = 0 

with torch.no_grad():

    for images , labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


accuracy = 100 * correct/total

print(f"\n test accuracy : {accuracy:.2f}%")

os.makedirs("saved_models",exist_ok=True)

torch.save(model.state_dict(),"saved_models/cifar10_cnn.pth")

print("model saved succesfully!")

