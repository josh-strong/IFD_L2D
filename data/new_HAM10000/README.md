# New HAM10000 (DermaMNIST-Extended) Dataset

## Overview

This dataset is an improved version of the original HAM10000 dataset, addressing known issues and providing enhanced data quality for dermatological image classification research. The dataset contains **11,719 dermatoscopic images** across **7 skin condition classes**.

## Classes

The dataset includes the same 7 classes as the original HAM10000:

- **0: akiec** - Actinic keratoses and intraepithelial carcinoma / Bowen's disease
- **1: bcc** - Basal cell carcinoma  
- **2: bkl** - Benign keratosis-like lesions
- **3: df** - Dermatofibroma
- **4: mel** - Melanoma
- **5: nv** - Melanocytic nevi (largest class with ~66% of samples)
- **6: vasc** - Vascular lesions

## Dataset Files

### Original Files
- `dermamnist_extended_224.npz` - Original dataset with train/val/test splits
- `DermaMNIST-E.csv` - Metadata and labels for all samples

### Processed Files (after running preprocess.py)
- `dermamnist_extended_224_with_expert.npz` - Dataset with expert split included
- `data_splits_summary.csv` - Summary of class distributions across splits
- `preprocess.py` - Script to create expert data split

## Data Preprocessing

### Creating Expert Split

To create the expert data split (required for the ML pipeline):

```bash
cd data/new_HAM10000
python3 preprocess.py
```

This script:
1. Combines the original train and validation data (10,208 samples total)
2. Re-splits them into new train (60%), validation (15%), and expert (25%) sets
3. Maintains stratified class distribution across all splits
4. Saves the new data structure to `dermamnist_extended_224_with_expert.npz`

### Data Split Proportions

**Original Splits:**
- Train: 10,015 samples (85.4%)
- Val: 193 samples (1.6%) 
- Test: 1,511 samples (12.9%)

**With Expert Split (Updated Allocation):**
- Train: ~6,124 samples (~52.3%)
- Val: ~1,531 samples (~13.1%)
- Expert: ~2,553 samples (~21.8%)
- Test: 1,511 samples (12.9%) [unchanged]

## Image Properties

- **Format**: RGB images
- **Size**: 224×224 pixels
- **Data type**: uint8 (0-255 range)
- **Channels**: 3 (RGB)

## Usage

### Loading the Dataset

```python
from datasets import load_new_ham10000

# Load with expert split (default)
train_data, val_data, test_data, expert_data = load_new_ham10000(use_expert_split=True)

# Load without expert split (original 3-way split)
train_data, val_data, test_data = load_new_ham10000(use_expert_split=False)

# Load with data augmentation
train_data, val_data, test_data, expert_data = load_new_ham10000(data_aug=True)
```

### Training Example

```python
# Set dataset in config
config['dataset'] = 'new_ham10000'

# Train expert classifier
from train_expert_clf import train_and_compute_prototypicality
best_state, centroids, max_distances = train_and_compute_prototypicality(config)

# Generate expert predictions
from generate_exp_preds import ensure_expert_predictions_exist
ensure_expert_predictions_exist(config, expert_archetypes=['high_specialist'])
```

## Model Configuration

The dataset uses:
- **Backbone**: ResNet34 (same as original HAM10000)
- **Input channels**: 3 (RGB)
- **Number of classes**: 7
- **Normalization**: ImageNet statistics ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

## Improvements Over Original HAM10000

This dataset addresses several issues identified in the original HAM10000:
- Enhanced data quality and consistency
- Improved label accuracy and validation
- Better class balance considerations
- Updated metadata and documentation

## Dataset Statistics

| Split  | Total Samples | akiec | bcc | bkl | df | mel | nv   | vasc |
|--------|---------------|-------|-----|-----|----|----|------|------|
| Train  | 6,124         | 201   | 317 | 673 | 70 | 680| 4,096| 87   |
| Val    | 1,531         | 50    | 80  | 168 | 17 | 170| 1,024| 22   |
| Expert | 2,553         | 84    | 132 | 280 | 29 | 284| 1,708| 36   |
| Test   | 1,511         | 43    | 93  | 217 | 44 | 171| 908  | 35   |
| **Total** | **11,719** | **378** | **622** | **1,338** | **160** | **1,305** | **7,736** | **180** |

## Dependencies

- numpy
- pandas
- scikit-learn
- torch
- torchvision
- PIL

## Citation

Please cite the original papers when using this dataset:
- Original HAM10000 dataset paper
- DermaMNIST-Extended improvement paper (if available)

## Notes

- The test set remains unchanged from the original dataset to maintain benchmark comparability
- Expert split creation uses stratified sampling to preserve class distributions
- The dataset is designed to be a drop-in replacement for the original HAM10000 in existing pipelines