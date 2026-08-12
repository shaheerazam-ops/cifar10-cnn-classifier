# CIFAR-10 Image Classifier — PyTorch

A deep-learning image classification project built with **PyTorch**, focused on CNN architecture design, model evaluation, error analysis, data augmentation, transfer learning, and fine-tuning.

The project compares a custom CNN against a pretrained ResNet18 to understand how architecture and fine-tuning affect performance on CIFAR-10.

## Results

| Experiment        | Model             | Validation |       Test |
| ----------------- | ----------------- | ---------: | ---------: |
| Baseline          | Custom CNN        |     85.36% | **85.77%** |
| Transfer Learning | ResNet18 (Frozen) |     81.36% | **80.80%** |
| Fine-Tuning       | ResNet18 + Layer4 | **94.38%** |    Pending |

> **Best result:** 94.38% validation accuracy using a partially fine-tuned ResNet18.

The fine-tuned experiment was interrupted by the available Google Colab GPU session before the final test evaluation could be completed.

## What I Experimented With

* Built a CNN from scratch using convolution, BatchNorm, ReLU, and max-pooling layers.
* Created training, validation, and test pipelines.
* Performed error analysis using classification metrics and confusion matrices.
* Tested data augmentation and removed color jitter after it reduced validation performance.
* Compared a frozen pretrained ResNet18 against a custom CNN.
* Fine-tuned the deeper ResNet18 layers to adapt ImageNet features to CIFAR-10.

## Key Finding

The frozen ResNet18 performed worse than the custom CNN (**80.80% vs 85.77% test accuracy**).

After allowing the deeper `layer4` block to fine-tune, validation accuracy increased dramatically to **94.38%**.

This demonstrated that pretrained features may require adaptation when transferred to a different dataset.

## Dataset

**CIFAR-10**

* 60,000 RGB images
* 32×32 resolution
* 10 classes
* 45,000 training
* 5,000 validation
* 10,000 test

## Tech Stack

`Python` · `PyTorch` · `Torchvision` · `NumPy` · `Matplotlib` · `Pillow` · `CUDA`

## Project Structure

```text
cifar10-cnn-classifier/
├── model.py
├── train.py
├── transfer_model.py
├── transfer_train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Running the Project

```bash
git clone https://github.com/shaheerazam-ops/cifar10-cnn-classifier.git
cd cifar10-cnn-classifier
pip install -r requirements.txt
python train.py
```

For transfer learning:

```bash
python transfer_train.py
```

## Future Work

* Complete final evaluation of the fine-tuned ResNet18.
* Add a lightweight inference demo.
* Compare additional pretrained architectures.
* Experiment with learning-rate scheduling and full fine-tuning.

## Author

**Shaheer Azam Khan**

BS Software Engineering — University of Karachi
Aspiring AI/ML Engineer
