import torch
import torchvision
import torchvision.transforms as transforms
import sys
from PIL import Image
from model import CNN


# -------------------------
# 1. Device
# -------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# 2. Categories
# -------------------------

categories = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# -------------------------
# 3 check image argument
# -------------------------
if len(sys.argv)< 2 :
    print("usage python predict.py <image_path>")
    sys.exit()

image_path = sys.argv[1]

# -------------------------
# 4 preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# -------------------------
# 5. load image
# -------------------------
image = Image.open(image_path).convert("RGB")


# -------------------------
# 6. preprocess image
# -------------------------

image = transform(image)

image = image.unsqueeze(0).to(device)

# -------------------------
# 7. load model
# -------------------------

model = CNN().to(device)
model.load_state_dict(
    torch.load(
        "saved_models/cifar10_cnn.pth",
        map_location=device
    )
)


# -------------------------
# 8. Evaluation mode
# -------------------------

model.eval()




# -------------------------
# 9. Prediction
# -------------------------

with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(output, 1)

    confidence , predicted = torch.max(probabilities, 1)

    


# -------------------------
# 9. Display result
# -------------------------

prediction = categories[predicted.item()]

confidence = confidence.item() * 100

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2f}%")