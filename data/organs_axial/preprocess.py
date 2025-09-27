#!/usr/bin/env python3
"""
Preprocess organs_axial dataset to create expert split.
Similar to HAM10000 preprocessing but working with existing splits.
"""

import os
import pandas as pd
import numpy as np
from collections import Counter


def stratified_split_indices(all_indices: np.ndarray,
                             labels: np.ndarray,
                             test_size: float,
                             random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministically split indices into train/test while preserving label distribution.

    Args:
        all_indices: Array of indices to split.
        labels: Labels aligned with all_indices.
        test_size: Proportion assigned to the test split (between 0 and 1).
        random_state: Seed for deterministic shuffling.

    Returns:
        (train_indices, test_indices) as absolute indices into the original array.
    """
    rng = np.random.RandomState(random_state)

    # Group indices by label
    label_to_indices: dict[int, list[int]] = {}
    for idx, label in zip(all_indices.tolist(), labels.tolist()):
        label_to_indices.setdefault(int(label), []).append(idx)

    train_parts: list[int] = []
    test_parts: list[int] = []

    for label, label_indices in label_to_indices.items():
        label_indices = np.array(label_indices)
        rng.shuffle(label_indices)

        n_total = len(label_indices)
        n_test = int(round(n_total * test_size))
        n_test = max(0, min(n_total, n_test))

        test_split = label_indices[:n_test]
        train_split = label_indices[n_test:]

        test_parts.append(test_split)
        train_parts.append(train_split)

    # Concatenate and shuffle again for randomness
    train_indices = np.concatenate(train_parts) if len(train_parts) > 0 else np.array([], dtype=int)
    test_indices = np.concatenate(test_parts) if len(test_parts) > 0 else np.array([], dtype=int)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices

def create_expert_split():
    """
    Create expert split from existing train/val data, maintaining test split.
    """
    
    # Load the annotations
    data_path = "organs_axial_data/organs_axial"
    annotations_path = os.path.join(data_path, "annotations.csv")
    
    print("Loading annotations...")
    df = pd.read_csv(annotations_path)
    
    # Get current split counts
    print("Current split distribution:")
    print(df['split'].value_counts())
    
    # Keep test split unchanged
    test_df = df[df['split'] == 'test'].copy()
    
    # Combine train and val to redistribute  
    train_val_df = df[df['split'].isin(['train', 'val'])].copy()
    
    print(f"\nTotal train+val samples: {len(train_val_df)}")
    print(f"Test samples (unchanged): {len(test_df)}")
    
    # Get labels for stratification
    labels = train_val_df['tasks/organ label'].values
    
    print(f"\nLabel distribution in train+val:")
    print(Counter(labels))
    
    # Calculate new splits from the train+val data
    # NEW Target proportions for train+val portion (more data for expert):
    # Train: ~60%, Val: ~15%, Expert: ~25% 
    train_prop = 0.60
    val_expert_prop = 0.40  # 15% + 25% = 40%
    
    # First split: train vs (val + expert)
    indices = np.arange(len(train_val_df))
    train_indices, val_expert_indices = stratified_split_indices(
        all_indices=indices,
        labels=labels,
        test_size=val_expert_prop,
        random_state=42
    )
    
    # Second split: val vs expert (37.5% val, 62.5% expert of the val_expert portion)
    # This gives us 15% val (0.375 * 40%) and 25% expert (0.625 * 40%) of total
    val_expert_labels = labels[val_expert_indices]
    val_indices_relative, expert_indices_relative = stratified_split_indices(
        all_indices=np.arange(len(val_expert_indices)),
        labels=val_expert_labels,
        test_size=0.625,  # 25% / 40% = 62.5% of val_expert goes to expert
        random_state=42
    )
    
    # Convert back to absolute indices
    val_indices = val_expert_indices[val_indices_relative]
    expert_indices = val_expert_indices[expert_indices_relative]
    
    # Create new dataframes
    new_train_df = train_val_df.iloc[train_indices].copy()
    new_val_df = train_val_df.iloc[val_indices].copy()
    new_expert_df = train_val_df.iloc[expert_indices].copy()
    
    # Update split labels
    new_train_df['split'] = 'train'
    new_val_df['split'] = 'val'
    new_expert_df['split'] = 'expert'
    
    # Combine all splits
    final_df = pd.concat([new_train_df, new_val_df, new_expert_df, test_df], ignore_index=True)
    
    print(f"\nNew split distribution:")
    print(final_df['split'].value_counts())
    
    # Print label distribution for each split
    for split_name in ['train', 'val', 'expert', 'test']:
        split_data = final_df[final_df['split'] == split_name]
        print(f"\n{split_name.capitalize()} label distribution:")
        print(Counter(split_data['tasks/organ label'].values))
    
    # Save updated annotations
    output_path = os.path.join(data_path, "annotations_with_expert.csv")
    final_df.to_csv(output_path, index=False)
    print(f"\nSaved updated annotations to: {output_path}")
    
    # Create new split files
    splits_dir = os.path.join(data_path, "splits_with_expert")
    os.makedirs(splits_dir, exist_ok=True)
    
    for split_name in ['train', 'val', 'expert', 'test']:
        split_data = final_df[final_df['split'] == split_name]
        split_file = os.path.join(splits_dir, f"{split_name}.txt")
        
        with open(split_file, 'w') as f:
            for filepath in split_data['filepath']:
                f.write(f"{filepath}\n")
        
        print(f"Created split file: {split_file} ({len(split_data)} samples)")
    
    return final_df

if __name__ == "__main__":
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("Creating expert split for organs_axial dataset...")
    final_df = create_expert_split()
    print("\nPreprocessing completed successfully!")