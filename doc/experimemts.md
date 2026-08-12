# CIFAR-10 CNN Image Classifier

A PyTorch-based image classification project built to explore convolutional neural networks, model evaluation, error analysis, data augmentation, transfer learning, and fine-tuning on the CIFAR-10 dataset.

The project started with a custom CNN trained from scratch and progressively moved toward a pretrained ResNet18 architecture. The goal was not simply to achieve the highest accuracy, but to understand **why different architectures and training strategies perform differently**.

---

## Results

| Experiment | Model                          | Validation Accuracy |              Test Accuracy |
| ---------- | ------------------------------ | ------------------: | -------------------------: |
| 1          | Custom CNN                     |              85.36% |                 **85.77%** |
| 2          | ResNet18 — Frozen              |              81.36% |                 **80.80%** |
| 3          | ResNet18 — Partial Fine-Tuning |          **94.38%** | *Pending final evaluation* |

### Best Result So Far

The strongest result came from partially fine-tuning a pretrained ResNet18.

At epoch 7:

```text
Training Loss:      0.0222
Validation Accuracy: 94.38%
```

The final test accuracy is intentionally left pending because the Colab GPU session ended before the complete 15-epoch experiment and final test evaluation could be completed.

---

## Project Goals

This project was designed to answer several practical machine-learning questions:

* How well can a CNN trained from scratch classify CIFAR-10?
* Which classes are the most difficult for the model?
* What types of mistakes does the model make?
* Does data augmentation improve generalization?
* Does a pretrained ImageNet model transfer effectively to CIFAR-10?
* What happens when the pretrained feature extractor is partially fine-tuned?
* How much does fine-tuning improve performance compared with a frozen feature extractor?

---

## Dataset

The project uses the **CIFAR-10** dataset.

CIFAR-10 contains:

* 60,000 RGB images
* 32 × 32 pixel resolution
* 10 classes
* 50,000 training images
* 10,000 test images

### Classes

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

The original 50,000 training images were divided into:

```text
Training:   45,000
Validation:  5,000
Test:       10,000
```

The dataset itself is not included in this repository.

---

## Model 1 — Custom CNN

The first model was designed and trained from scratch using PyTorch.

Architecture:

```text
Input
  ↓
Conv2D (32)
  ↓
BatchNorm
  ↓
ReLU
  ↓
Conv2D (32)
  ↓
BatchNorm
  ↓
ReLU
  ↓
MaxPool
  ↓
Conv2D (64)
  ↓
BatchNorm
  ↓
ReLU
  ↓
Conv2D (64)
  ↓
BatchNorm
  ↓
ReLU
  ↓
MaxPool
  ↓
Conv2D (128)
  ↓
BatchNorm
  ↓
ReLU
  ↓
Conv2D (128)
  ↓
BatchNorm
  ↓
ReLU
  ↓
MaxPool
  ↓
Fully Connected
  ↓
10 Classes
```

### Result

```text
Best Validation Accuracy: 85.36%
Test Accuracy:            85.77%
```

This established the baseline for the remaining experiments.

---

## Model 2 — Transfer Learning with ResNet18

The second experiment used a pretrained **ResNet18** model.

The original ImageNet classifier was replaced with a 10-class CIFAR-10 classifier.

The convolutional feature extractor was initially frozen while the new classifier was trained.

### Result

```text
Best Validation Accuracy: 81.36%
Test Accuracy:            80.80%
```

Interestingly, the frozen pretrained model performed **worse than the custom CNN**.

This demonstrated an important limitation of transfer learning:

> A pretrained feature extractor is not automatically optimal for a new dataset when its learned high-level representations cannot adapt to the target domain.

---

## Model 3 — Partial Fine-Tuning

The third experiment investigated whether allowing the deeper ResNet features to adapt to CIFAR-10 would improve performance.

The earlier ResNet layers remained frozen while `layer4` and the new classification head were allowed to learn.

```text
ResNet18

layer1  → Frozen
layer2  → Frozen
layer3  → Frozen
layer4  → Trainable
FC      → Trainable
```

A smaller learning rate was used to avoid aggressively modifying the pretrained weights.

### Validation Progress

```text
Epoch 01 → 93.14%
Epoch 02 → 93.96%
Epoch 03 → 93.08%
Epoch 04 → 94.26%
Epoch 05 → 94.28%
Epoch 06 → 93.42%
Epoch 07 → 94.38%  ← Best validation accuracy
```

### Current Result

```text
Best Validation Accuracy: 94.38%
Final Test Accuracy:      Pending
```

The experiment was interrupted when the available Colab GPU compute was exhausted before the complete training run could finish.

---

## Error Analysis

Rather than looking only at overall accuracy, the project also examined individual classification errors.

The original CNN produced the following classification performance:

| Class      | Precision | Recall |   F1 |
| ---------- | --------: | -----: | ---: |
| Airplane   |      0.84 |   0.90 | 0.87 |
| Automobile |      0.91 |   0.95 | 0.93 |
| Bird       |      0.79 |   0.83 | 0.81 |
| Cat        |      0.81 |   0.66 | 0.73 |
| Deer       |      0.83 |   0.87 | 0.85 |
| Dog        |      0.75 |   0.84 | 0.79 |
| Frog       |      0.89 |   0.90 | 0.90 |
| Horse      |      0.92 |   0.86 | 0.89 |
| Ship       |      0.93 |   0.91 | 0.92 |
| Truck      |      0.95 |   0.86 | 0.90 |

### Main Weakness: Cat vs Dog

The confusion matrix showed that cats were one of the most difficult classes.

The model made:

```text
138
```

cat → dog mistakes.

The average confidence of these incorrect predictions was approximately:

```text
69.16%
```

This was investigated by visualizing the misclassified images.

The CIFAR-10 images are extremely small (32 × 32), and many examples contain limited visual information. Fine-grained distinctions such as cat vs. dog can therefore be difficult for a CNN to learn.

---

## Data Augmentation Experiment

An additional experiment investigated whether color jitter would improve generalization.

The result was:

```text
Best Validation Accuracy: 83.96%
```

compared with:

```text
Previous Validation Accuracy: 85.36%
```

The experiment therefore **did not improve the model**.

Instead of keeping an augmentation simply because it is commonly used, it was removed based on the experimental result and the characteristics of the dataset.

This reinforced an important ML principle:

> Data augmentation should be selected based on the data and validated experimentally rather than added blindly.

---

## Key Findings

### 1. A stronger architecture does not automatically produce a better result

The frozen ResNet18 achieved only:

```text
80.80% test accuracy
```

while the simpler custom CNN achieved:

```text
85.77%
```

The pretrained model needed adaptation to the target dataset.

### 2. Fine-tuning made a major difference

Allowing the deeper ResNet features to adapt produced:

```text
94.38% validation accuracy
```

at epoch 7.

This was a substantial improvement over both previous approaches.

### 3. Error analysis matters

Overall accuracy alone hid the fact that:

* Cat classification was relatively weak.
* Dog classification was also challenging.
* Cat → dog was a significant source of error.

### 4. Experiments should be evidence-driven

The color-jitter experiment decreased validation performance, so it was not retained.

Rather than continuously adding techniques, each modification was evaluated against a baseline.

---

## Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Pillow
* Google Colab
* CUDA / NVIDIA Tesla T4
* Git / GitHub

---

## Project Structure

```text
cifar10-cnn-classifier/
│
├── model.py
├── train.py
│
├── transfer_model.py
├── transfer_train.py
│
├── evaluate.py
├── requirements.txt
├── README.md
└── .gitignore
```

The CIFAR-10 dataset and model checkpoints are intentionally excluded from the repository.

---

## Installation

```bash
git clone https://github.com/shaheerazam-ops/cifar10-cnn-classifier.git

cd cifar10-cnn-classifier

pip install -r requirements.txt
```

---

## Requirements

```text
torch
torchvision
matplotlib
pillow
```

---

## Running the Custom CNN

```bash
python train.py
```

The training script:

1. Loads CIFAR-10
2. Creates training and validation splits
3. Trains the CNN
4. Tracks validation accuracy
5. Saves the best checkpoint
6. Evaluates the final model on the test set

---

## Running Transfer Learning

```bash
python transfer_train.py
```

The transfer-learning experiment resizes CIFAR-10 images for ResNet18 and uses ImageNet pretrained weights.

For GPU training, Google Colab with CUDA was used because the local machine did not have a dedicated GPU.

---

## Reproducibility

The dataset is intentionally excluded from GitHub because of its size.

To reproduce the experiments:

1. Download the CIFAR-10 Python dataset.
2. Extract it locally.
3. Install the dependencies.
4. Run the training scripts.
5. Use the evaluation scripts to generate classification metrics and confusion matrices.

---

## Limitations

* CIFAR-10 images are only 32 × 32 pixels.
* The local development machine did not have a dedicated GPU.
* The final fine-tuned ResNet18 test accuracy is pending because the available Colab GPU session ended during training.
* No web interface or production deployment is included.
* The project focuses on model development and experimentation rather than production serving.

---

## Future Improvements

Potential next steps include:

* Complete final evaluation of the fine-tuned ResNet18.
* Compare full fine-tuning against partial fine-tuning.
* Perform systematic hyperparameter tuning.
* Add learning-rate scheduling.
* Evaluate additional architectures.
* Add experiment tracking.
* Build a lightweight inference interface with Streamlit or Gradio.
* Export the trained model for inference.

---

## What I Learned

This project was primarily a practical exploration of the ML development workflow:

```text
Dataset
   ↓
Baseline CNN
   ↓
Training
   ↓
Validation
   ↓
Test Evaluation
   ↓
Error Analysis
   ↓
Experimentation
   ↓
Transfer Learning
   ↓
Fine-Tuning
   ↓
Model Comparison
```

The biggest takeaway was that improving a machine-learning model is not simply about choosing a larger architecture. Understanding the data, identifying failure cases, forming hypotheses, running controlled experiments, and comparing results are equally important.

---

## Status

**Completed:** Core CNN training, evaluation, error analysis, augmentation experiment, transfer learning, and partial fine-tuning.

**In progress:** Final test evaluation of the best fine-tuned ResNet18 checkpoint.

---

## Author

**Shaheer Azam Khan**

BS Software Engineering — University of Karachi

Interested in **AI/ML Engineering, Computer Vision, and applied deep learning**.
