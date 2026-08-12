import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import CNN

from sklearn.metrics import confusion_matrix, classification_report

# 1. Device

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# 2. Transform

test_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )

])


# 3. Test Dataset


test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=test_transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# 4. Load Model

model = CNN().to(device)

model.load_state_dict(
    torch.load(
        "saved_models/best_model.pth",
        map_location=device
    )
)

model.eval()


# 5. Make Predictions


all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

# 6. Confusion Matrix

cm = confusion_matrix(
    all_labels,
    all_predictions
)


print("\nConfusion Matrix:")
print(cm)

# 7. Classification Report

classes = test_dataset.classes


print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes
    )
)

# 8 finding cat /dog msitakes 
import matplotlib.pyplot as plt 

cat_class = classes.index("cat")
dog_class = classes.index("dog")

cat_dog_images = []
cat_dog_confidences = []

with torch.no_grad():
    for images , labels in test_loader :
        images_device = images.to(device)

        outputs = model(images_device)

        probabilities = torch.softmax(outputs,dim = 1)

        predictions = torch.argmax(outputs , dim = 1)

        for i in range(len(labels)):
            actual = labels[i].item()
            predicted = predictions[i].item()

            # actually cat but predicted dog 

            if actual == cat_class and predicted == dog_class:

                cat_dog_images.append(images[i])
                cat_dog_confidences.append(
                    probabilities[i][dog_class].item()
                )

# 9 display mistkaes 

print(f"\nCat = Dog mistakes : {len(cat_dog_images)}")

num_images = min(12 , len(cat_dog_images))

plt.figure(figsize= (12, 8))

for i in range(num_images):
    image = cat_dog_images[i]

    image = image.permute( 1 , 2 , 0) #( converting pic size)

    mean = torch.tensor(
        (0.4914 , 0.4822 , 0.4465)
    )

    std = torch.tensor(
        (0.2470 , 0.2435 , 0.2616)
    )

    image = image * std + mean 
    image = torch.clamp(image, 0 , 1)

    plt.subplot(3 ,4 , i + 1)

    plt.imshow(image)

    plt.title(
        f"cat = Dog\n"
        f"{cat_dog_confidences[i] * 100 : .1f}%"
    )

    plt.axis("off")

    

plt.tight_layout() 

plt.show()




average_confidence = sum(cat_dog_confidences) / len(cat_dog_confidences)
print(
     f"Average Cat -> Dog confidence: "
     f"{average_confidence * 100:.2f}%"
     )