#!/usr/bin/env python3
"""
Preprocessing script for new_HAM10000 (DermaMNIST-Extended) dataset.
Creates expert data split by combining train and val data, then re-splitting into
train, val, and expert sets while maintaining class distribution.

Usage:
    cd data/new_HAM10000
    python3 preprocess.py

Requirements:
    - dermamnist_extended_224.npz should be present
    - DermaMNIST-E.csv should be present
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter


def load_and_prepare_data():
    """Load the original data and prepare for re-splitting."""
    print("Loading original data...")
    
    # Load NPZ file
    npz_path = 'dermamnist_extended_224.npz'
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    data = np.load(npz_path)
    
    # Load CSV for reference (though we'll mainly use the NPZ labels)
    csv_path = 'DermaMNIST-E.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"CSV loaded: {len(df)} samples")
    
    # Extract train and val data
    train_images = data['train_images']  # (10015, 224, 224, 3)
    train_labels = data['train_labels'].flatten()  # (10015,)
    val_images = data['val_images']      # (193, 224, 224, 3)
    val_labels = data['val_labels'].flatten()    # (193,)
    test_images = data['test_images']    # (1511, 224, 224, 3)
    test_labels = data['test_labels'].flatten()  # (1511,)
    
    print(f"Original splits:")
    print(f"  Train: {len(train_images)} samples")
    print(f"  Val: {len(val_images)} samples")
    print(f"  Test: {len(test_images)} samples")
    
    # Combine train and val data for re-splitting
    combined_images = np.vstack([train_images, val_images])
    combined_labels = np.hstack([train_labels, val_labels])
    
    print(f"\nCombined train+val: {len(combined_images)} samples")
    print(f"Class distribution in combined data:")
    class_counts = Counter(combined_labels)
    for class_idx in sorted(class_counts.keys()):
        print(f"  Class {class_idx}: {class_counts[class_idx]} samples")
    
    return combined_images, combined_labels, test_images, test_labels


def create_expert_split(combined_images, combined_labels):
    """Create new train, val, and expert splits from combined data."""
    print("\nCreating expert split...")
    
    # Target proportions for train+val portion:
    # Train: ~60%, Val: ~15%, Expert: ~25% 
    train_prop = 0.60
    val_expert_prop = 0.40  # remaining 40% will be split between val and expert
    
    # First split: train vs (val + expert)
    train_images, val_expert_images, train_labels, val_expert_labels = train_test_split(
        combined_images, combined_labels,
        test_size=val_expert_prop,
        stratify=combined_labels,
        random_state=42
    )
    
    # Second split: val vs expert (from the remaining 40%)
    # Expert gets 25% / 40% = 62.5% of val_expert data
    val_images, expert_images, val_labels, expert_labels = train_test_split(
        val_expert_images, val_expert_labels,
        test_size=0.625,  # 25% / 40% = 62.5% of val_expert goes to expert
        stratify=val_expert_labels,
        random_state=42
    )
    
    print(f"\nNew splits created:")
    print(f"  Train: {len(train_images)} samples ({len(train_images)/len(combined_images)*100:.1f}%)")
    print(f"  Val: {len(val_images)} samples ({len(val_images)/len(combined_images)*100:.1f}%)")
    print(f"  Expert: {len(expert_images)} samples ({len(expert_images)/len(combined_images)*100:.1f}%)")
    
    # Verify class distribution is maintained
    print(f"\nClass distribution verification:")
    for split_name, labels in [("Train", train_labels), ("Val", val_labels), ("Expert", expert_labels)]:
        class_counts = Counter(labels)
        print(f"  {split_name}:")
        for class_idx in sorted(class_counts.keys()):
            print(f"    Class {class_idx}: {class_counts[class_idx]} samples")
    
    return train_images, train_labels, val_images, val_labels, expert_images, expert_labels


def save_new_splits(train_images, train_labels, val_images, val_labels, 
                   expert_images, expert_labels, test_images, test_labels):
    """Save the new data splits to NPZ file."""
    output_file = 'dermamnist_extended_224_with_expert.npz'
    
    print(f"\nSaving new splits to {output_file}...")
    
    # Reshape labels to match original format (N, 1)
    train_labels = train_labels.reshape(-1, 1)
    val_labels = val_labels.reshape(-1, 1)
    expert_labels = expert_labels.reshape(-1, 1)
    test_labels = test_labels.reshape(-1, 1)
    
    np.savez_compressed(
        output_file,
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        expert_images=expert_images,
        expert_labels=expert_labels,
        test_images=test_images,
        test_labels=test_labels
    )
    
    print(f"✓ New data splits saved to {output_file}")
    
    # Create a summary CSV for reference
    summary_file = 'data_splits_summary.csv'
    split_info = []
    
    for split_name, labels in [("train", train_labels.flatten()), 
                              ("val", val_labels.flatten()), 
                              ("expert", expert_labels.flatten()), 
                              ("test", test_labels.flatten())]:
        class_counts = Counter(labels)
        for class_idx in sorted(class_counts.keys()):
            split_info.append({
                'split': split_name,
                'class': class_idx,
                'count': class_counts[class_idx]
            })
    
    summary_df = pd.DataFrame(split_info)
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Summary saved to {summary_file}")


def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("New HAM10000 (DermaMNIST-Extended) Expert Split Creation")
    print("=" * 60)
    
    try:
        # Load and prepare data
        combined_images, combined_labels, test_images, test_labels = load_and_prepare_data()
        
        # Create expert split
        train_images, train_labels, val_images, val_labels, expert_images, expert_labels = create_expert_split(
            combined_images, combined_labels
        )
        
        # Save new splits
        save_new_splits(train_images, train_labels, val_images, val_labels, 
                       expert_images, expert_labels, test_images, test_labels)
        
        print("\n" + "=" * 60)
        print("✓ Expert split creation completed successfully!")
        print("✓ You can now use the new_ham10000 dataset with expert split")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()