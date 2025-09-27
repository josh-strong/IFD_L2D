import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import VisionDataset
import torchvision.datasets as datasets
from utils import ROOT
from torch.utils.data import Dataset
import medmnist
from medmnist import INFO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyVisionDataset(VisionDataset):
    def __init__(self, images, labels, transform, image_names=None, noise_levels=None):
        super().__init__(ROOT + '/data', transform=transform)
        # Apply transform to each image individually if it exists
        if self.transform:
            self.data = [self.transform(image) for image in images]
        else:
            self.data = images
        # Convert data to a tensor and stack them if they were transformed individually
        self.data = torch.stack(self.data) if self.transform else images
        self.targets = torch.as_tensor(labels, dtype=torch.int64)
        self.image_names = image_names
        self.noise_levels = noise_levels

    def __getitem__(self, index):
        if (self.image_names is not None) and (self.noise_levels is not None):  # for imagenet-16h
            img, target, noise_lvl, image_name = self.data[index], int(self.targets[index]), self.noise_levels[index]
            return img, target, noise_lvl, image_name
        else:
            img, target = self.data[index], int(self.targets[index])
            return img, target

    def __len__(self):
        return len(self.data)
    
class SyntheticHumanPredictionDataset(Dataset):
    def __init__(self, data):
        """
        Initializes the dataset with synthetic data.
        Args:
            data (dict): Dictionary containing 'xc', 'mc', and 'yc' tensors.
        """
        self.xc = data['xc']
        self.mc = data['mc']
        self.yc = data['yc']
        
    def __len__(self):
        # Assuming 'xc' is a tensor where the first dimension is the number of samples
        return self.xc.size(1)

    def __getitem__(self, idx):
        # Return the data at index idx
        return {
            'xc': self.xc[:,idx,:,:,:],
            'mc': self.mc[:,idx],
            'yc': self.yc[:,idx]
        }

def load_ham10000():
    LABEL_TO_CONDITION = {
        0: 'nv',   # Nevus
        1: 'mel',  # Melanoma
        2: 'bkl',  # Benign Keratosis
        3: 'bcc',  # Basal Cell Carcinoma
        4: 'akiec',# Actinic Keratosis
        5: 'vasc', # Vascular Lesions
        6: 'df',   # Dermatofibroma
    }
    ds_path = ROOT + '/data/HAM10000/'
    train_dataset = torch.load(ds_path + 'train_data.pt', weights_only=False)
    val_dataset = torch.load(ds_path + 'validation_data.pt', weights_only=False)
    test_dataset = torch.load(ds_path + 'test_data.pt', weights_only=False)
    expert_dataset = torch.load(ds_path + 'expert_data.pt', weights_only=False)

    transform_train = transforms.Compose([])
    transform_test = transforms.Compose([])

    images_train = train_dataset["data"]
    targets_train = train_dataset["labels"]

    images_val = val_dataset["data"]
    targets_val = val_dataset["labels"]

    images_test = test_dataset["data"]
    targets_test = test_dataset["labels"]

    images_expert = expert_dataset["data"]
    targets_expert = expert_dataset["labels"]


    train_dataset = MyVisionDataset(images_train, targets_train, transform_train)
    val_dataset = MyVisionDataset(images_val, targets_val, transform_test)
    test_dataset = MyVisionDataset(images_test, targets_test, transform_test)
    expert_dataset = MyVisionDataset(images_expert, targets_expert, transform_test)

    return train_dataset, val_dataset, test_dataset, expert_dataset

def load_bus(root_dir, transform=None):
    df = pd.read_csv(os.path.join(root_dir,"annotations.csv"))

    # Split the data into training and validation sets based on the 'split' column
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']  # Assuming you want to load a test dataset

    # Define the transformations (e.g., resize, normalization)
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
        ])

    # Helper function to load images and their labels
    def load_images_and_labels(data_frame):
        images = []
        labels = []
        for _, row in data_frame.iterrows():
            img_path = os.path.join(root_dir, row['filepath'])
            label = row['tasks/case category']
            
            # Open image using PIL and apply transformation
            image = Image.open(img_path)

            # Ensure the image is in a compatible format (PIL Image)
            if transform:
                image = transform(image)
            
            images.append(image)
            labels.append(label)
        
        # Convert lists to tensors
        images = torch.stack(images)
        labels = torch.tensor(labels)

        return images, labels

    # Load the images and labels for training, validation, and test datasets
    train_images, train_labels = load_images_and_labels(train_df)
    val_images, val_labels = load_images_and_labels(val_df)
    test_images, test_labels = load_images_and_labels(test_df)

    # Create MyVisionDataset objects for train, val, and test sets
    train_dataset = MyVisionDataset(train_images, train_labels, None)
    val_dataset = MyVisionDataset(val_images, val_labels, None)
    test_dataset = MyVisionDataset(test_images, test_labels, None)  # Test dataset

    return train_dataset, val_dataset, test_dataset

def load_organs_axial(root_dir, transform=None, use_expert_split=True):
    # Choose annotations file based on whether to use expert split
    if use_expert_split:
        annotations_file = "annotations_with_expert.csv"
    else:
        annotations_file = "annotations.csv"
    
    df = pd.read_csv(os.path.join(root_dir, annotations_file))

    # Split the data based on the 'split' column
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    # Expert split only exists if using the expert split annotations
    if use_expert_split:
        expert_df = df[df['split'] == 'expert']
    
    # Define the transformations (e.g., resize, normalization)
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
        ])

    # Helper function to load images and their labels
    def load_images_and_labels(data_frame):
        images = []
        labels = []
        for _, row in data_frame.iterrows():
            img_path = os.path.join(root_dir, row['filepath'])
            label = row['tasks/organ label']
            
            # Open image using PIL and apply transformation
            image = Image.open(img_path)

            # Ensure the image is in a compatible format (PIL Image)
            if transform:
                image = transform(image)
            
            images.append(image)
            labels.append(label)
        
        # Convert lists to tensors
        images = torch.stack(images)
        labels = torch.tensor(labels)

        return images, labels

    # Load the images and labels for training, validation, and test datasets
    train_images, train_labels = load_images_and_labels(train_df)
    val_images, val_labels = load_images_and_labels(val_df)
    test_images, test_labels = load_images_and_labels(test_df)

    # Create MyVisionDataset objects for train, val, and test sets
    train_dataset = MyVisionDataset(train_images, train_labels, None)
    val_dataset = MyVisionDataset(val_images, val_labels, None)
    test_dataset = MyVisionDataset(test_images, test_labels, None)

    # Return expert dataset if using expert split, otherwise return original 3-tuple
    expert_images, expert_labels = load_images_and_labels(expert_df)
    expert_dataset = MyVisionDataset(expert_images, expert_labels, None)
    return train_dataset, val_dataset, test_dataset, expert_dataset

def load_bcn(root_dir=ROOT + "/data/bcn", transform=None):
    df_train = pd.read_csv(os.path.join(root_dir, "BCN_20k_train.csv"))
    test_df = pd.read_csv(os.path.join(root_dir, "BCN_20k_test.csv"))

    # Create label mapping for diagnoses
    unique_diagnoses = sorted(df_train['diagnosis'].unique())
    diagnosis_to_label = {diagnosis: idx for idx, diagnosis in enumerate(unique_diagnoses)}

    # Split the train data into train and val
    train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['diagnosis'])

    # Define default transformations if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Standard size for many vision models
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                                std=[0.229, 0.224, 0.225])
        ])

    # Helper function to load images and labels
    def load_images_and_labels(data_frame):
        images = []
        labels = []
        for _, row in data_frame.iterrows():
            img_path = os.path.join(root_dir, 'bcn_images', row['bcn_filename'])
            label = diagnosis_to_label[row['diagnosis']]
            
            # Open and transform image
            image = Image.open(img_path).convert('RGB')
            if transform:
                image = transform(image)
            
            images.append(image)
            labels.append(label)
        
        # Convert lists to tensors
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        return images, labels

    # Load the images and labels for all splits
    train_images, train_labels = load_images_and_labels(train_df)
    val_images, val_labels = load_images_and_labels(val_df)
    test_images, test_labels = load_images_and_labels(test_df)

    # Create dataset objects
    train_dataset = MyVisionDataset(train_images, train_labels, None)
    val_dataset = MyVisionDataset(val_images, val_labels, None)
    test_dataset = MyVisionDataset(test_images, test_labels, None)

    print("Loaded train dataset")
    print("Loaded val dataset")
    print("Loaded test dataset")

    return train_dataset, val_dataset, test_dataset

def load_bloodmnist(data_aug=False, use_expert_split=True):
    # Load BloodMNIST dataset info
    info = INFO['bloodmnist']
    n_channels = info['n_channels']
    num_classes = len(info['label'])

    # Set download directory to organized data folder
    download_root = os.path.join(ROOT, 'data', 'blood_mnist')
    os.makedirs(download_root, exist_ok=True)

    # Normalization values specific to BloodMNIST
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1,1]

    # Define transformations
    if data_aug:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            normalize
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # Load dataset using medmnist with specified download root
    bloodmnist_train = medmnist.BloodMNIST(split='train', download=True, transform=transform_train, root=download_root)
    bloodmnist_val = medmnist.BloodMNIST(split='val', download=True, transform=transform_test, root=download_root)
    bloodmnist_test = medmnist.BloodMNIST(split='test', download=True, transform=transform_test, root=download_root)

    # Extract data and labels
    images_train, targets_train = bloodmnist_train.imgs, bloodmnist_train.labels.ravel()
    images_val, targets_val = bloodmnist_val.imgs, bloodmnist_val.labels.ravel()
    images_test, targets_test = bloodmnist_test.imgs, bloodmnist_test.labels.ravel()

    if use_expert_split:
        # Create expert split by redistributing train and val data
        # Combine train and val for redistribution
        combined_images = np.concatenate([images_train, images_val], axis=0)
        combined_targets = np.concatenate([targets_train, targets_val], axis=0)
        
        # Create indices for stratified split
        indices = np.arange(len(combined_images))
        
        # Split into train (60%), val+expert (40%)
        train_indices, val_expert_indices = train_test_split(
            indices,
            test_size=0.40,
            stratify=combined_targets,
            random_state=42
        )
        
        # Split val_expert into val (37.5%) and expert (62.5%) to get ~15% val, ~25% expert
        val_expert_targets = combined_targets[val_expert_indices]
        val_indices_relative, expert_indices_relative = train_test_split(
            np.arange(len(val_expert_indices)),
            test_size=0.625,  # 25% / 40% = 62.5% of val_expert goes to expert
            stratify=val_expert_targets,
            random_state=42
        )
        
        # Convert to absolute indices
        val_indices = val_expert_indices[val_indices_relative]
        expert_indices = val_expert_indices[expert_indices_relative]
        
        # Create new splits
        new_train_images = combined_images[train_indices]
        new_train_targets = combined_targets[train_indices]
        
        new_val_images = combined_images[val_indices]
        new_val_targets = combined_targets[val_indices]
        
        new_expert_images = combined_images[expert_indices]
        new_expert_targets = combined_targets[expert_indices]
        
        print(f"BloodMNIST splits - Train: {len(new_train_images)}, Val: {len(new_val_images)}, Expert: {len(new_expert_images)}, Test: {len(images_test)}")
        
        # Create custom VisionDataset objects
        train_dataset = MyVisionDataset(new_train_images, new_train_targets, transform_train)
        val_dataset = MyVisionDataset(new_val_images, new_val_targets, transform_test)
        expert_dataset = MyVisionDataset(new_expert_images, new_expert_targets, transform_test)
        test_dataset = MyVisionDataset(images_test, targets_test, transform_test)
        
        return train_dataset, val_dataset, test_dataset, expert_dataset
    else:
        # Original behavior: return 3 datasets without expert split
        train_dataset = MyVisionDataset(images_train, targets_train, transform_train)
        val_dataset = MyVisionDataset(images_val, targets_val, transform_test)
        test_dataset = MyVisionDataset(images_test, targets_test, transform_test)
        
        return train_dataset, val_dataset, test_dataset

def load_new_ham10000(data_aug=False, use_expert_split=True):
    """
    Load the new HAM10000 (DermaMNIST-Extended) dataset with expert split.
    This is an improved version of the original HAM10000 dataset.
    
    Args:
        data_aug (bool): Whether to apply data augmentation to training data
        use_expert_split (bool): Whether to use the 4-split (train/val/test/expert) or 3-split (train/val/test)
    
    Returns:
        If use_expert_split=True: (train_dataset, val_dataset, test_dataset, expert_dataset)
        If use_expert_split=False: (train_dataset, val_dataset, test_dataset)
    """
    # Dataset path
    dataset_path = os.path.join(ROOT, 'data', 'new_HAM10000')
    
    # Use preprocessed data with expert split if available, otherwise use original
    if use_expert_split and os.path.exists(os.path.join(dataset_path, 'dermamnist_extended_224_with_expert.npz')):
        npz_file = os.path.join(dataset_path, 'dermamnist_extended_224_with_expert.npz')
        has_expert_split = True
    else:
        npz_file = os.path.join(dataset_path, 'dermamnist_extended_224.npz')
        has_expert_split = False
        print("Warning: Expert split not found. Use `python3 preprocess.py` in data/new_HAM10000/ to create it.")
    
    # Load data
    data = np.load(npz_file)
    
    # Normalization values for RGB images (same as original HAM10000)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Define transformations
    if data_aug:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize
        ])
    
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        normalize
    ])
    
    # Extract data and convert labels to proper format
    images_train = data['train_images']
    targets_train = data['train_labels'].ravel()  # Flatten to 1D
    
    images_val = data['val_images']
    targets_val = data['val_labels'].ravel()
    
    images_test = data['test_images']
    targets_test = data['test_labels'].ravel()
    
    if has_expert_split and use_expert_split:
        images_expert = data['expert_images']
        targets_expert = data['expert_labels'].ravel()
        
        print(f"New HAM10000 splits - Train: {len(images_train)}, Val: {len(images_val)}, Expert: {len(images_expert)}, Test: {len(images_test)}")
        
        # Create custom VisionDataset objects
        train_dataset = MyVisionDataset(images_train, targets_train, transform_train)
        val_dataset = MyVisionDataset(images_val, targets_val, transform_test)
        expert_dataset = MyVisionDataset(images_expert, targets_expert, transform_test)
        test_dataset = MyVisionDataset(images_test, targets_test, transform_test)
        
        return train_dataset, val_dataset, test_dataset, expert_dataset
    else:
        print(f"New HAM10000 splits - Train: {len(images_train)}, Val: {len(images_val)}, Test: {len(images_test)}")
        
        # Create custom VisionDataset objects
        train_dataset = MyVisionDataset(images_train, targets_train, transform_train)
        val_dataset = MyVisionDataset(images_val, targets_val, transform_test)
        test_dataset = MyVisionDataset(images_test, targets_test, transform_test)
        
        return train_dataset, val_dataset, test_dataset


def load_oct(data_aug=False, use_expert_split=True, subsample_ratio=0.1):
    # Load OCTMNIST dataset info
    info = INFO['octmnist']
    n_channels = info['n_channels']
    num_classes = len(info['label'])

    # Set download directory to organized data folder
    download_root = os.path.join(ROOT, 'data', 'oct')
    os.makedirs(download_root, exist_ok=True)

    # Normalization values specific to OCTMNIST
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1,1]

    # Define transformations
    if data_aug:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            normalize
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # Load dataset using medmnist with specified download root
    octmnist_train = medmnist.OCTMNIST(split='train', download=True, transform=transform_train, root=download_root)
    octmnist_val = medmnist.OCTMNIST(split='val', download=True, transform=transform_test, root=download_root)
    octmnist_test = medmnist.OCTMNIST(split='test', download=True, transform=transform_test, root=download_root)

    # Extract data and labels
    images_train, targets_train = octmnist_train.imgs, octmnist_train.labels.ravel()
    images_val, targets_val = octmnist_val.imgs, octmnist_val.labels.ravel()
    images_test, targets_test = octmnist_test.imgs, octmnist_test.labels.ravel()

    # Apply subsampling if specified (maintain original 10% default)
    if subsample_ratio < 1.0:
        images_train, _, targets_train, _ = train_test_split(
            images_train, targets_train, train_size=subsample_ratio, stratify=targets_train, random_state=42
        )
        images_val, _, targets_val, _ = train_test_split(
            images_val, targets_val, train_size=subsample_ratio, stratify=targets_val, random_state=42
        )

    if use_expert_split:
        # Create expert split by redistributing train and val data
        # Combine train and val for redistribution
        combined_images = np.concatenate([images_train, images_val], axis=0)
        combined_targets = np.concatenate([targets_train, targets_val], axis=0)
        
        # Create indices for stratified split
        indices = np.arange(len(combined_images))
        
        # Split into train (60%), val+expert (40%)
        train_indices, val_expert_indices = train_test_split(
            indices,
            test_size=0.40,
            stratify=combined_targets,
            random_state=42
        )
        
        # Split val_expert into val (37.5%) and expert (62.5%) to get ~15% val, ~25% expert
        val_expert_targets = combined_targets[val_expert_indices]
        val_indices_relative, expert_indices_relative = train_test_split(
            np.arange(len(val_expert_indices)),
            test_size=0.625,  # 25% / 40% = 62.5% of val_expert goes to expert
            stratify=val_expert_targets,
            random_state=42
        )
        
        # Convert to absolute indices
        val_indices = val_expert_indices[val_indices_relative]
        expert_indices = val_expert_indices[expert_indices_relative]
        
        # Create new splits
        new_train_images = combined_images[train_indices]
        new_train_targets = combined_targets[train_indices]
        
        new_val_images = combined_images[val_indices]
        new_val_targets = combined_targets[val_indices]
        
        new_expert_images = combined_images[expert_indices]
        new_expert_targets = combined_targets[expert_indices]
        
        print(f"OCTMNIST splits - Train: {len(new_train_images)}, Val: {len(new_val_images)}, Expert: {len(new_expert_images)}, Test: {len(images_test)}")
        
        # Create custom VisionDataset objects
        train_dataset = MyVisionDataset(new_train_images, new_train_targets, transform_train)
        val_dataset = MyVisionDataset(new_val_images, new_val_targets, transform_test)
        expert_dataset = MyVisionDataset(new_expert_images, new_expert_targets, transform_test)
        test_dataset = MyVisionDataset(images_test, targets_test, transform_test)
        
        return train_dataset, val_dataset, test_dataset, expert_dataset
    else:
        # Original behavior: return 3 datasets without expert split
        train_dataset = MyVisionDataset(images_train, targets_train, transform_train)
        val_dataset = MyVisionDataset(images_val, targets_val, transform_test)
        test_dataset = MyVisionDataset(images_test, targets_test, transform_test)
        
        return train_dataset, val_dataset, test_dataset

def load_cifar10(data_aug=False, use_expert_split=True, subsample_ratio=0.05, return_full_test=False):
    """
    Load CIFAR-10 dataset with train/val/test/expert splits.
    Memory-efficient implementation with subsampling to avoid OOM issues.
    Optionally returns the full (non-subsampled) test set if return_full_test is True.
    
    Args:
        data_aug (bool): Whether to apply data augmentation to training set
        use_expert_split (bool): Whether to include expert split
        subsample_ratio (float): Ratio of data to use (0.3 = 30% of original data)
        return_full_test (bool): If True, return the full (non-subsampled) test set
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, expert_dataset) if use_expert_split=True
               (train_dataset, val_dataset, test_dataset) if use_expert_split=False
               If return_full_test=True, test_dataset is the full test set (not subsampled)
    """
    # Define data directory
    data_dir = os.path.join(ROOT, 'data', 'cifar10')
    os.makedirs(data_dir, exist_ok=True)
    
    # CIFAR-10 normalization values
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Define transformations
    if data_aug:
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match other datasets
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Load original CIFAR-10 datasets
    cifar10_train_full = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
    cifar10_test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=None)
    
    # Get indices and labels (memory efficient)
    train_labels = np.array(cifar10_train_full.targets)
    test_labels = np.array(cifar10_test.targets)
    
    # Apply subsampling to reduce memory usage
    if subsample_ratio < 1.0:
        # print(f"Subsampling CIFAR-10 to {subsample_ratio*100}% of original data for memory efficiency")
        # Subsample train data
        train_indices_full, _, train_labels_subsampled, _ = train_test_split(
            np.arange(len(train_labels)), train_labels, 
            train_size=subsample_ratio, stratify=train_labels, random_state=42
        )
        # Subsample test data (unless return_full_test is True)
        if not return_full_test:
            test_indices_full, _, test_labels_subsampled, _ = train_test_split(
                np.arange(len(test_labels)), test_labels,
                train_size=subsample_ratio, stratify=test_labels, random_state=42
            )
        else:
            test_indices_full = np.arange(len(test_labels))
            test_labels_subsampled = test_labels
    else:
        train_indices_full = np.arange(len(train_labels))
        test_indices_full = np.arange(len(test_labels))
        train_labels_subsampled = train_labels
        test_labels_subsampled = test_labels

    # Define memory-efficient dataset classes
    class CIFAR10SplitDataset(torch.utils.data.Dataset):
        def __init__(self, cifar_dataset, indices, targets, transform):
            self.cifar_dataset = cifar_dataset
            self.indices = indices
            self.targets = targets
            self.transform = transform
            # Pre-load all images for compatibility with existing code
            self.data = self._load_all_images()
        
        def _load_all_images(self):
            """Load all images into memory for compatibility with context_data.py"""
            images = []
            for idx in self.indices:
                img, _ = self.cifar_dataset[idx]
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            return torch.stack(images)
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    class CIFAR10TestDataset(torch.utils.data.Dataset):
        def __init__(self, cifar_dataset, transform, indices=None):
            self.cifar_dataset = cifar_dataset
            self.transform = transform
            self.indices = indices if indices is not None else np.arange(len(cifar_dataset))
            # Pre-load all images for compatibility with existing code
            self.data = self._load_all_images()
            self.targets = torch.tensor([cifar_dataset.targets[i] for i in self.indices])
        
        def _load_all_images(self):
            """Load all images into memory for compatibility with context_data.py"""
            images = []
            for i in self.indices:
                img, _ = self.cifar_dataset[i]
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            return torch.stack(images)
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    if use_expert_split:
        # Create train/val/expert split from subsampled train data (60%/15%/25%)
        train_prop = 0.60
        val_expert_prop = 0.40  # 15% + 25% = 40%
        
        # First split: train vs (val + expert)
        train_indices_relative, val_expert_indices_relative = train_test_split(
            np.arange(len(train_labels_subsampled)),
            test_size=val_expert_prop,
            stratify=train_labels_subsampled,
            random_state=42
        )
        
        # Get labels for splits
        targets_train_split = train_labels_subsampled[train_indices_relative]
        targets_val_expert = train_labels_subsampled[val_expert_indices_relative]
        
        # Second split: val vs expert (15% vs 25% of original = 37.5% vs 62.5% of val_expert)
        val_indices_relative, expert_indices_relative = train_test_split(
            np.arange(len(targets_val_expert)),
            test_size=0.625,  # 25% / 40% = 62.5% of val_expert goes to expert
            stratify=targets_val_expert,
            random_state=42
        )
        
        # Get final label splits
        targets_val_split = targets_val_expert[val_indices_relative]
        targets_expert_split = targets_val_expert[expert_indices_relative]
        
        # Convert to absolute indices in the original dataset
        train_indices = train_indices_full[train_indices_relative]
        val_indices = train_indices_full[val_expert_indices_relative[val_indices_relative]]
        expert_indices = train_indices_full[val_expert_indices_relative[expert_indices_relative]]
        
        # Create datasets
        train_dataset = CIFAR10SplitDataset(cifar10_train_full, train_indices, targets_train_split, transform_train)
        val_dataset = CIFAR10SplitDataset(cifar10_train_full, val_indices, targets_val_split, transform_test)
        expert_dataset = CIFAR10SplitDataset(cifar10_train_full, expert_indices, targets_expert_split, transform_test)
        
        # Create test dataset (full or subsampled)
        if return_full_test:
            test_dataset = CIFAR10TestDataset(cifar10_test, transform_test, np.arange(len(cifar10_test)))
        else:
            test_dataset = CIFAR10TestDataset(cifar10_test, transform_test, test_indices_full)
        
        print(f"CIFAR-10 splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Expert: {len(expert_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset, expert_dataset
    else:
        # Original behaviour: create train/val split from subsampled train data
        train_indices_relative, val_indices_relative = train_test_split(
            np.arange(len(train_labels_subsampled)),
            test_size=0.2,
            stratify=train_labels_subsampled,
            random_state=42
        )
        
        # Get label splits
        targets_train_split = train_labels_subsampled[train_indices_relative]
        targets_val_split = train_labels_subsampled[val_indices_relative]
        
        # Convert to absolute indices
        train_indices = train_indices_full[train_indices_relative]
        val_indices = train_indices_full[val_indices_relative]
        
        # Create memory-efficient datasets
        train_dataset = CIFAR10SplitDataset(cifar10_train_full, train_indices, targets_train_split, transform_train)
        val_dataset = CIFAR10SplitDataset(cifar10_train_full, val_indices, targets_val_split, transform_test)
        if return_full_test:
            test_dataset = CIFAR10TestDataset(cifar10_test, transform_test, np.arange(len(cifar10_test)))
        else:
            test_dataset = CIFAR10TestDataset(cifar10_test, transform_test, test_indices_full)
        
        return train_dataset, val_dataset, test_dataset

def load_imagenet16_greyscale():
    """
    Loads the greyscale ImageNet-16 dataset from .npz files.
    Args:
        npz_dir (str): Directory containing imagenet16h_train_split.npz, imagenet16h_val_split.npz, imagenet16h_expert_split.npz
    Returns:
        train_dataset, val_dataset, test_dataset, expert_dataset (MyVisionDataset)
    """

    npz_dir_h = os.path.join(ROOT, 'data/imagenet-16h')
    # Use context manager to load arrays
    with np.load(os.path.join(npz_dir_h, 'imagenet16h_train_split_v1.npz')) as train:
        train_images = train['images']
        train_labels = train['labels']
        train_noise_levels = train['noise_levels']
        train_image_names = train['image_names']
    with np.load(os.path.join(npz_dir_h, 'imagenet16h_val_split_v1.npz')) as val:
        val_images = val['images']
        val_labels = val['labels']
        val_noise_levels = val['noise_levels']
        val_image_names = val['image_names']
    with np.load(os.path.join(npz_dir_h, 'imagenet16h_test_split_v1.npz')) as test:
        test_images = test['images']
        test_labels = test['labels']
        test_noise_levels = test['noise_levels']
        test_image_names = test['image_names']

    # Define transforms for greyscale images

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    # Datasets
    train_dataset = MyVisionDataset([Image.fromarray(img) for img in train_images], train_labels, transform, train_image_names, train_noise_levels)
    val_dataset = MyVisionDataset([Image.fromarray(img) for img in val_images], val_labels, transform, val_image_names, val_noise_levels)
    test_dataset = MyVisionDataset([Image.fromarray(img) for img in test_images], test_labels, transform, test_image_names, test_noise_levels)
    return train_dataset, val_dataset, test_dataset, None

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import ROOT

def load_cifar10_new(split_proportions=(0.6, 0.2, 0.2), data_aug=False, seed=42, max_total_images=None):
    """
    Load CIFAR-10 using only the 'test' split, and create new train/val/test splits.
    This is because only the 'test' split contains real human annotations.
    The returned splits are mutually exclusive and cover the full 'test' set.

    Args:
        split_proportions (tuple): Proportions for (train, val, test) splits. Must sum to 1.0.
        data_aug (bool): Whether to apply data augmentation to the training set.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    assert abs(sum(split_proportions) - 1.0) < 1e-6, "Split proportions must sum to 1.0"
    train_prop, val_prop, test_prop = split_proportions

    # Define data directory
    data_dir = os.path.join(ROOT, 'data', 'cifar10')
    os.makedirs(data_dir, exist_ok=True)

    # CIFAR-10 normalisation values (ImageNet-style)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Define transformations
    if data_aug:
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # Load only the 'test' split of CIFAR-10
    cifar10_test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=None)
    all_indices = np.arange(len(cifar10_test))
    all_labels = np.array(cifar10_test.targets)

    # Optional stratified subsample BEFORE splitting, preserving original test indices
    if max_total_images is not None and max_total_images < len(all_indices):
        subset_indices, _, subset_labels, _ = train_test_split(
            all_indices, all_labels,
            train_size=max_total_images,
            stratify=all_labels,
            random_state=seed
        )
        all_indices = subset_indices
        all_labels = subset_labels

    # Stratified split into train/val/test using only the 'test' split
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        all_indices, all_labels,
        train_size=train_prop,
        stratify=all_labels,
        random_state=seed
    )
    # Now split temp_indices into val and test
    val_relative_prop = val_prop / (val_prop + test_prop)
    val_indices, test_indices, val_labels, test_labels = train_test_split(
        temp_indices, temp_labels,
        train_size=val_relative_prop,
        stratify=temp_labels,
        random_state=seed
    )

    # Define a dataset wrapper with .data attribute (tensor of images)
    class CIFAR10Subset(torch.utils.data.Dataset):
        def __init__(self, cifar_dataset, indices, transform):
            self.cifar_dataset = cifar_dataset
            self.indices = indices
            self.transform = transform
            # Store targets as a tensor
            self.targets = torch.tensor([cifar_dataset.targets[i] for i in self.indices], dtype=torch.long)
            # Pre-load and transform all images into a tensor for .data attribute
            imgs = []
            for i in self.indices:
                img, _ = cifar_dataset[i]
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            self.data = torch.stack(imgs)  # shape: (N, C, H, W)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            # Return pre-transformed image and label
            return self.data[idx], self.targets[idx]

    train_dataset = CIFAR10Subset(cifar10_test, train_indices, transform_train)
    val_dataset = CIFAR10Subset(cifar10_test, val_indices, transform_eval)
    test_dataset = CIFAR10Subset(cifar10_test, test_indices, transform_eval)

    print(f"CIFAR-10 (from test split) - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, None