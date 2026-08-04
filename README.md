# CIFAR-10 CNN Image Classifier

A PyTorch-based Convolutional Neural Network (CNN) for classifying images into the 10 categories of the CIFAR-10 dataset.

This project was built as a hands-on deep learning project to understand the complete image classification workflow — from dataset loading and CNN architecture design to model training, evaluation, saving, and inference on external images.

## Project Status

**Current version:** v1 — Baseline CNN

The current model achieves approximately **68% test accuracy after 5 training epochs**. The model and training pipeline will be iteratively improved in later versions.

---

## Features

* CIFAR-10 dataset loading with PyTorch/Torchvision
* Batched training using `DataLoader`
* Custom CNN architecture built with PyTorch
* ReLU activation functions
* Max pooling for spatial downsampling
* Cross-entropy loss
* Adam optimizer
* GPU/CPU device support
* Model evaluation using test accuracy
* Model checkpoint saving/loading
* Inference on individual CIFAR-10 images
* External image inference
* Softmax-based prediction confidence

---

## CIFAR-10 Classes

The model classifies images into 10 categories:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

---

## Model Architecture

The current CNN consists of two convolutional blocks followed by fully connected classification layers.

```text
Input
3 × 32 × 32
      │
      ▼
Conv2D
3 → 32 channels
3 × 3 kernel
padding = 1
      │
      ▼
ReLU
      │
      ▼
MaxPool
2 × 2
      │
      ▼
Conv2D
32 → 64 channels
3 × 3 kernel
padding = 1
      │
      ▼
ReLU
      │
      ▼
MaxPool
2 × 2
      │
      ▼
Flatten
      │
      ▼
Linear
4096 → 128
      │
      ▼
ReLU
      │
      ▼
Linear
128 → 10
      │
      ▼
Class Logits
```

### PyTorch Architecture

```python
CNN(
    (features): Sequential(
        (0): Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2)
        (3): Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        (4): ReLU()
        (5): MaxPool2d(kernel_size=2, stride=2)
    )

    (classifier): Sequential(
        (0): Flatten()
        (1): Linear(4096, 128)
        (2): ReLU()
        (3): Linear(128, 10)
    )
)
```

---

## Training

The model is trained using:

* **Optimizer:** Adam
* **Learning rate:** `0.001`
* **Loss:** CrossEntropyLoss
* **Batch size:** `64`
* **Epochs:** `5`
* **Input size:** `3 × 32 × 32`

The current preprocessing pipeline uses `ToTensor()`.

No additional normalization is currently applied.

### Training Pipeline

```text
CIFAR-10
   ↓
ToTensor()
   ↓
DataLoader
   ↓
CNN
   ↓
Logits
   ↓
CrossEntropyLoss
   ↓
Backpropagation
   ↓
Adam Optimizer
   ↓
Updated Weights
```

---

## Results

### Baseline Performance

| Metric          |   Result |
| --------------- | -------: |
| Training epochs |        5 |
| Batch size      |       64 |
| Test accuracy   |     ~68% |
| Dataset         | CIFAR-10 |

The exact test accuracy can vary slightly between training runs because the training data is shuffled and the model weights are randomly initialized.

---

## Inference

The project includes an inference script that loads the trained model and predicts the class of an image.

### CIFAR-10 Test Image

Example:

```text
Actual: cat
Predicted: cat
```

The model can also make predictions on external images.

Example:

```text
Prediction: ship
Confidence: 91.68%
```

The external image is resized to `32 × 32` and converted to a PyTorch tensor before being passed to the model.

### Inference Pipeline

```text
External Image
      ↓
Resize → 32 × 32
      ↓
ToTensor()
      ↓
Add Batch Dimension
      ↓
1 × 3 × 32 × 32
      ↓
Trained CNN
      ↓
10 Logits
      ↓
Softmax
      ↓
Prediction + Confidence
```

---

## Project Structure

```text
cifar10-cnn-classifier/
│
├── train.py                 # Dataset loading, training and evaluation
├── model.py                 # CNN architecture
├── predict.py               # Model inference
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored files and directories
├── README.md                # Project documentation
│
└── saved_models/
    └── cifar10_cnn.pth      # Trained model weights
```

The CIFAR-10 dataset is not included in the repository and is downloaded automatically by Torchvision.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/cifar10-cnn-classifier.git
cd cifar10-cnn-classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run:

```bash
python train.py
```

The CIFAR-10 dataset will be downloaded automatically if it is not already present.

After training, the model weights are saved to:

```text
saved_models/cifar10_cnn.pth
```

---

## Run Inference

To classify an external image:

```bash
python predict.py image.jpg
```

Example:

```text
Prediction: ship
Confidence: 91.68%
```

---

## What I Learned

This project was developed to gain practical experience with:

* PyTorch tensors
* Neural network modules
* CNN architecture
* Convolutional layers
* Feature maps
* ReLU activation
* Max pooling
* Flattening
* Fully connected layers
* DataLoader and batching
* Training loops
* Forward propagation
* Loss calculation
* Backpropagation
* Gradient descent
* Adam optimization
* `model.train()` vs `model.eval()`
* `torch.no_grad()`
* Logits vs probabilities
* Softmax
* Model serialization with `.pth`
* Image preprocessing
* Model inference

---

## Future Improvements

This repository is intentionally being developed incrementally.

Planned improvements include:

* [ ] Data augmentation
* [ ] Improved CNN architecture
* [ ] Hyperparameter tuning
* [ ] Learning-rate experiments
* [ ] Longer training
* [ ] Validation split
* [ ] Confusion matrix
* [ ] Per-class accuracy
* [ ] Training/validation loss curves
* [ ] Training/validation accuracy curves
* [ ] Improved external image handling
* [ ] Experiment tracking
* [ ] Model performance comparison

---

## Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Pillow
* Git & GitHub

---

## License

This project is intended as a learning and portfolio project.
