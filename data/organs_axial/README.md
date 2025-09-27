# Organs Axial Dataset

## Overview
This directory contains the Organs Axial dataset for organ classification from axial CT slices.

## Files
- `download_data.sh`: Script to download and extract the dataset
- `preprocess.py`: Script to create expert split from existing train/val data
- `organs_axial_data/`: Directory containing the dataset after download

## Usage

### 1. Navigate to the directory
```bash
cd data/organs_axial
```

### 2. Download Dataset
Run the download script to get the data:
```bash
bash download_data.sh
```

The script will:
- Download the dataset from Zenodo
- Extract it to `organs_axial_data/`
- Optionally create expert split

### 3. Create Expert Split (Optional)
If you didn't create the expert split during download, you can create it manually:
```bash
python3 preprocess.py
```

This will:
- Create `annotations_with_expert.csv` with the new expert split
- Create `splits_with_expert/` directory with new split files
- Maintain test set integrity while redistributing train/val into train/val/expert

### 4. Loading in Code
The dataset can be loaded using the `load_organs_axial()` function in `datasets.py`:

```python
# Load with expert split (4 datasets returned)
train_dataset, val_dataset, test_dataset, expert_dataset = load_organs_axial(
    root_dir="data/organs_axial/organs_axial_data/organs_axial/",
    use_expert_split=True
)

# Load without expert split (3 datasets returned, original behavior)
train_dataset, val_dataset, test_dataset = load_organs_axial(
    root_dir="data/organs_axial/organs_axial_data/organs_axial/",
    use_expert_split=False
)
```

## Split Information

### Original Splits
- Train: 871 samples (52.9%)
- Val: 156 samples (9.5%)  
- Test: 618 samples (37.6%)
- Total: 1645 samples

### With Expert Split (Updated Allocation)
- Train: ~616 samples (~37.5%)
- Val: ~154 samples (~9.4%)
- Expert: ~257 samples (~15.6%)
- Test: 618 samples (37.6%) - unchanged
- Total: 1645 samples

The expert split is created by redistributing the original train and validation sets while maintaining stratification across the 11 organ classes and keeping the test set unchanged for consistency. **The expert split has been enlarged** (from ~8% to ~15.6%) to provide more training data for expert modeling in medical imaging tasks.

## Dataset Details
- **Classes**: 11 organ types (heart, lungs, liver, spleen, pancreas, kidneys, bladder, femoral heads)
- **Image format**: TIFF files
- **Task**: Multi-class classification
- **Domain**: Abdominal CT scans
