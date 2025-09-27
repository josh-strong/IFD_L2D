# BloodMNIST Dataset

## Overview
This directory contains the BloodMNIST dataset for blood cell classification from microscopy images.

## Dataset Details
- **Source**: MedMNIST collection (https://medmnist.com/)
- **Task**: Blood cell classification
- **Classes**: 8 different blood cell types
- **Image format**: 28x28 RGB images
- **Domain**: Medical microscopy imaging

## Directory Structure
After downloading, the data will be organized as:
```
blood_mnist/
├── bloodmnist.npz  # Downloaded by medmnist library
└── README.md       # This file
```

## Usage

### Loading the Dataset
The dataset is automatically downloaded and split when using the `load_bloodmnist()` function:

```python
from datasets import load_bloodmnist

# Load with expert split (4 datasets returned)
train_dataset, val_dataset, test_dataset, expert_dataset = load_bloodmnist(use_expert_split=True)

# Load without expert split (3 datasets returned, original behavior)
train_dataset, val_dataset, test_dataset = load_bloodmnist(use_expert_split=False)
```

## Split Information

### With Expert Split
- **Train**: ~60% of original train+val data
- **Val**: ~15% of original train+val data  
- **Expert**: ~25% of original train+val data
- **Test**: Original test set (unchanged)

The expert split is created by redistributing the original train and validation sets while maintaining stratification across the 8 blood cell classes and keeping the test set unchanged for consistency.

### Blood Cell Classes
The dataset contains 8 classes of blood cells:
1. Class 0: Basophil
2. Class 1: Eosinophil  
3. Class 2: Erythroblast
4. Class 3: Immature granulocytes
5. Class 4: Lymphocyte
6. Class 5: Monocyte
7. Class 6: Neutrophil
8. Class 7: Platelet

## Model Configuration
- **Backbone**: EfficientNet-B0 (3-channel RGB input)
- **Dropout**: 0.3 for regularization
- **Training**: Balanced hyperparameters for medium-sized dataset
- **Learning Rate Scheduler**: ReduceLROnPlateau with moderate settings

## Notes
- Dataset is automatically downloaded via the MedMNIST library
- Images are normalized to [-1, 1] range
- Expert split maintains class balance across all splits
- Compatible with the expert prediction generation system